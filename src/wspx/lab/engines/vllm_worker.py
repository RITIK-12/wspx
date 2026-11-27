"""wspx's vLLM worker, started by the lab with vLLM's own Python.

vLLM pins its own torch and transformers, so this file imports nothing from wspx. The lab
runs ``<vllm python> vllm_worker.py --port P --token-file F -- <vllm engine args>`` and
talks to it on localhost only:

- ``POST /generate`` takes token ids (or a rendered prompt with its images and videos) and
  streams one JSON line per finished request, so the lab can report progress;
- ``POST /weights/{init,start,update,finish}`` take the lab's edited tensors over CUDA IPC
  through vLLM's own weight-transfer engine, under the checkpoint's tensor names, one layer
  per update (vLLM reloads layer by layer and keeps the layers not sent; ``wspx_vllm_ext``,
  copied next to this file, lets strict model loaders take part of a checkpoint);
- ``GET /ready``, ``GET /info`` and ``GET /checksums`` say what it is serving.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import math
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

# IPC handles arrive pickled (vLLM's documented HTTP form); the worker listens on localhost
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

STATE: dict[str, Any] = {"engine": None, "error": None, "started": time.time()}


def _parser() -> Any:
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:  # older layouts
        from vllm.utils import FlexibleArgumentParser
    return FlexibleArgumentParser()


def _engine(engine_argv: list[str]) -> Any:
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    parser = AsyncEngineArgs.add_cli_args(_parser())
    args = AsyncEngineArgs.from_cli_args(parser.parse_args(engine_argv))
    return AsyncLLMEngine.from_engine_args(args)


def _gpu() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "uuid": str(props.uuid),
        "capability": f"{props.major}.{props.minor}",
        "memory": props.total_memory,
    }


def _image(data: str) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(data))) as im:
        return im.convert("RGB")


def _video(spec: dict[str, Any]) -> Any:
    import numpy as np

    frames = np.load(io.BytesIO(base64.b64decode(spec["npy"])))
    fps = float(spec.get("fps") or 2.0)
    n = int(frames.shape[0])
    meta = {
        "fps": fps,
        "total_num_frames": n,
        "duration": n / fps,
        "frames_indices": list(range(n)),
        "video_backend": "wspx",
    }
    return (frames, meta)


def _prompt(req: dict[str, Any]) -> Any:
    from vllm import TokensPrompt

    if "ids" in req:
        return TokensPrompt(prompt_token_ids=[int(t) for t in req["ids"]])
    data: dict[str, Any] = {}
    if req.get("images"):
        data["image"] = [_image(i) for i in req["images"]]
    if req.get("videos"):
        data["video"] = [_video(v) for v in req["videos"]]
    out: dict[str, Any] = {"prompt": req["prompt"]}
    if data:
        out["multi_modal_data"] = data
    return out


def _sampling(engine: Any, body: dict[str, Any]) -> Any:
    from vllm import SamplingParams

    # the checkpoint's generation config (repetition penalty and the like), as transformers
    # applies it; then the call's own settings
    defaults = dict(engine.model_config.get_diff_sampling_param() or {})
    defaults.pop("max_new_tokens", None)
    temperature = float(body.get("temperature", 0.0))
    params = {
        **{k: v for k, v in defaults.items() if k in SamplingParams.__struct_fields__},
        "max_tokens": int(body.get("max_tokens", 64)),
        "temperature": temperature,
        "top_p": float(body.get("top_p", 1.0)) if temperature > 0 else 1.0,
        "top_k": 0,
        "seed": int(body.get("seed", 0)),
        "skip_special_tokens": False,
    }
    if body.get("watch"):
        params["logprob_token_ids"] = [int(t) for t in body["watch"]]
    return SamplingParams(**params)


def _row(i: int, out: Any, watch: list[int]) -> dict[str, Any]:
    first = out.outputs[0]
    watched: list[list[float]] = []
    for step in first.logprobs or []:
        watched.append([float(step[w].logprob) if w in step else -math.inf for w in watch])
    spec = getattr(first, "spec_decode_metrics", None)
    return {
        "i": i,
        "ids": [int(t) for t in first.token_ids],
        "watched": watched if watch else [],
        "finish": str(first.finish_reason or "stop"),
        "prompt_tokens": len(out.prompt_token_ids or []),
        "spec": dict(vars(spec)) if spec is not None and hasattr(spec, "__dict__") else None,
    }


def build_app(engine_argv: list[str], token: str) -> Any:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        try:
            STATE["engine"] = _engine(engine_argv)
        except Exception as exc:  # reported on /ready, so the lab can show why
            STATE["error"] = f"{type(exc).__name__}: {exc}"
        yield
        engine = STATE.get("engine")
        if engine is not None:
            engine.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def auth(request: Request, call_next: Any) -> Any:
        if request.headers.get("authorization") != f"Bearer {token}":
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    def engine() -> Any:
        if STATE["error"]:
            raise HTTPException(500, STATE["error"])
        if STATE["engine"] is None:
            raise HTTPException(503, "loading")
        return STATE["engine"]

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        if STATE["error"]:
            raise HTTPException(500, STATE["error"])
        if STATE["engine"] is None:
            raise HTTPException(503, f"loading for {time.time() - STATE['started']:.0f}s")
        return await info()

    @app.get("/info")
    async def info() -> dict[str, Any]:
        import vllm

        e = engine()
        cfg = e.model_config
        return {
            "vllm": vllm.__version__,
            "model": cfg.model,
            "dtype": str(cfg.dtype).removeprefix("torch."),
            "quantization": cfg.quantization,
            "max_model_len": cfg.max_model_len,
            "weight_version": await e.get_weight_version(),
            "gpu": _gpu(),
            "speculative": e.vllm_config.speculative_config is not None,
            "reload_mode": os.environ.get("VLLM_RELOAD_MODE", "auto"),
            "extension": e.vllm_config.parallel_config.worker_extension_cls,
        }

    @app.post("/generate")
    async def generate(request: Request) -> StreamingResponse:
        e = engine()
        body = await request.json()
        watch = [int(t) for t in body.get("watch", [])]
        params = _sampling(e, body)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def one(i: int, req: dict[str, Any]) -> None:
            final = None
            try:
                async for out in e.generate(_prompt(req), params, request_id=uuid.uuid4().hex):
                    final = out
                await queue.put(_row(i, final, watch))
            except Exception as exc:  # one bad request does not lose the others
                await queue.put({"i": i, "error": f"{type(exc).__name__}: {exc}"})

        requests = body.get("requests", [])
        started = time.perf_counter()
        tasks = [asyncio.create_task(one(i, r)) for i, r in enumerate(requests)]

        async def lines() -> AsyncIterator[bytes]:
            for _ in tasks:
                yield (json.dumps(await queue.get()) + "\n").encode()
            yield (
                json.dumps({"done": True, "seconds": time.perf_counter() - started}) + "\n"
            ).encode()

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.post("/weights/{phase}")
    async def weights(phase: str, request: Request) -> dict[str, Any]:
        from vllm.distributed.weight_transfer.base import (
            WeightTransferInitRequest,
            WeightTransferUpdateRequest,
        )

        e = engine()
        body = await request.json() if phase in ("init", "update", "finish") else {}
        if phase == "init":
            await e.init_weight_transfer_engine(
                WeightTransferInitRequest(init_info={"packed": bool(body.get("packed"))})
            )
        elif phase == "start":
            await e.start_weight_update()
        elif phase == "update":
            await e.update_weights(WeightTransferUpdateRequest(update_info=body["update_info"]))
        elif phase == "finish":
            await e.finish_weight_update(body.get("version"))
            # cached KV blocks (and encoded images) were computed with the old weights; a
            # prompt sharing a prefix with an earlier one must not reuse them
            for reset in ("reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache"):
                fn = getattr(e, reset, None)
                if fn is not None:
                    with contextlib.suppress(Exception):  # a cache this build lacks
                        await fn()
        else:
            raise HTTPException(404, f"no weight phase {phase!r}")
        return {"ok": True}

    @app.get("/checksums")
    async def checksums() -> Any:
        return await engine().compute_weight_checksums()

    return app


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    ours, engine_argv = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-file", required=True)
    args = parser.parse_args(ours)
    with open(args.token_file) as fh:
        token = fh.read().strip()

    import uvicorn

    uvicorn.run(build_app(engine_argv, token), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
