"""The vLLM engine: a worker process beside the lab that serves the lab's weights.

The worker (:mod:`wspx.lab.engines.vllm_worker`) runs with vLLM's own Python, since vLLM
pins its own torch. Before each generation the lab pushes every layer an edit, checkout,
discard or merge changed since the last push, under the checkpoint's tensor names
(:mod:`wspx.lab.checkpoint`), through vLLM's weight-transfer engine over CUDA IPC. vLLM
fuses and shards them itself. A whole layer goes at once: vLLM reloads per module, and a
module that receives only some of its tensors would be left with the rest uninitialized.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import pickle
import secrets
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import torch

from wspx.config import VllmSection
from wspx.errors import WspxError
from wspx.lab import analysis, checkpoint
from wspx.lab.engines import Engine, Rows, agreement_prompts
from wspx.lab.media import is_media
from wspx.lab.text import Prompt, Text
from wspx.lab.tree import TreeState, WorkingTree

WORKER = Path(__file__).with_name("vllm_worker.py")
VLLM_PYTHON = os.environ.get("WSPX_VLLM_PYTHON", "/opt/vllm/bin/python")
READY_TIMEOUT_S = 45 * 60
BUFFER_BYTES = 512 * 2**20
AGREEMENT_TOKENS = 16
AGREEMENT_TOP = 20

# Engine arguments vLLM's own recipes give a model type (``recipes/models/...``).
RECIPES: dict[str, tuple[list[str], dict[str, str]]] = {
    "inkling_mm_model": (
        ["--trust-remote-code", "--tokenizer-mode", "inkling"],
        {"VLLM_USE_V2_MODEL_RUNNER": "1"},
    ),
    "inkling_model": (
        ["--trust-remote-code", "--tokenizer-mode", "inkling"],
        {"VLLM_USE_V2_MODEL_RUNNER": "1"},
    ),
}
UNSERVED: dict[str, str] = {}
"""Model types vLLM cannot serve, with why (Inkling-2-Mini needs the pinned vLLM build of
the vLLM lab image; vLLM 0.34.0 itself does not have it)."""


class Transport:
    """How tensors reach the worker. Over CUDA IPC normally; tests swap in their own."""

    def send(
        self, http: httpx.Client, chunks: Iterable[dict[str, torch.Tensor]], version: str
    ) -> int:
        raise NotImplementedError


@dataclass
class IPCTransport(Transport):
    """vLLM's documented "trainer without vLLM over HTTP" IPC contract, packed: tensors are
    copied into one exported GPU buffer and sent a chunk at a time."""

    device: torch.device
    buffer_bytes: int = BUFFER_BYTES
    _inited: bool = False

    def send(
        self, http: httpx.Client, chunks: Iterable[dict[str, torch.Tensor]], version: str
    ) -> int:
        from torch.multiprocessing.reductions import reduce_tensor

        uuid = str(torch.cuda.get_device_properties(self.device).uuid)
        if not self._inited:
            _post(http, "/weights/init", {"packed": True})
            self._inited = True
        _post(http, "/weights/start", {})
        keep: list[torch.Tensor] = []  # exported memory lives until finish returns
        buffer: torch.Tensor | None = None
        handle = ""
        names: list[str] = []
        dtypes: list[str] = []
        shapes: list[list[int]] = []
        sizes: list[int] = []
        sent = 0

        def flush() -> None:
            if not names:
                return
            torch.cuda.synchronize(self.device)  # the copies must land before the worker reads
            info = {
                "names": list(names),
                "dtype_names": list(dtypes),
                "shapes": list(shapes),
                "ipc_handles_pickled": handle,
                "tensor_sizes": list(sizes),
            }
            _post(http, "/weights/update", {"update_info": info})
            for done in (names, dtypes, shapes, sizes):
                done.clear()

        for chunk in chunks:
            for name, tensor in chunk.items():
                t = tensor.detach().to(self.device).contiguous()
                nbytes = t.numel() * t.element_size()
                if buffer is None or nbytes > buffer.numel():
                    flush()
                    buffer = torch.empty(
                        max(self.buffer_bytes, nbytes), dtype=torch.uint8, device=self.device
                    )
                    keep.append(buffer)
                    exported = {uuid: reduce_tensor(buffer)[1]}  # type: ignore[no-untyped-call]
                    handle = base64.b64encode(pickle.dumps(exported)).decode()
                offset = sum(sizes)
                if names and offset + nbytes > buffer.numel():  # close before overflowing
                    flush()
                    offset = 0
                buffer[offset : offset + nbytes].copy_(t.view(-1).view(torch.uint8))
                names.append(name)
                dtypes.append(str(t.dtype).removeprefix("torch."))
                shapes.append(list(t.shape))
                sizes.append(nbytes)
                sent += nbytes
        flush()
        _post(http, "/weights/finish", {"version": version})
        del keep, buffer
        torch.cuda.ipc_collect()  # type: ignore[no-untyped-call]
        return sent


def _post(http: httpx.Client, path: str, body: dict[str, Any]) -> Any:
    r = http.post(path, json=body)
    if r.status_code != 200:
        raise WspxError(f"vLLM worker {path}: {r.status_code} {r.text[:400]}")
    return r.json()


@dataclass
class Plan:
    """Where each engine runs and how much of its GPU vLLM may take."""

    hf_device: str | None
    """``device_map`` for transformers (``None``: spread over every GPU)."""
    vllm_gpu: int
    memory: float
    lines: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return "engine plan:\n" + "\n".join(f"  {line}" for line in self.lines)


def plan(
    settings: VllmSection,
    *,
    checkpoint_bytes: int,
    gpus: list[tuple[str, int]] | None = None,
    hf_bytes: int | None = None,
) -> Plan:
    """Decide where vLLM runs before transformers loads (``gpus``: name and memory per GPU).

    With two or more GPUs, transformers keeps the first and vLLM the last; with one, vLLM
    gets what transformers leaves, which is checked again once it has loaded."""
    if gpus is None:
        if not torch.cuda.is_available():
            raise WspxError("the vLLM engine needs a GPU; set [lab] gpu, or use engine = 'hf'")
        gpus = [
            (torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory)
            for i in range(torch.cuda.device_count())
        ]
    need = int(checkpoint_bytes * 1.1) + 4 * 2**30  # weights, graphs, a minimal KV cache
    gb = 2**30
    if len(gpus) >= 2:
        name, total = gpus[-1]
        memory = settings.memory or 0.90
        p = Plan("cuda:0", len(gpus) - 1, memory)
        p.lines = [
            f"transformers on GPU 0 ({gpus[0][0]}, {gpus[0][1] / gb:.0f} GB)",
            f"vLLM on GPU {len(gpus) - 1} ({name}): {memory:.0%} of {total / gb:.0f} GB, "
            f"checkpoint {checkpoint_bytes / gb:.1f} GB",
        ]
        if memory * total < need:
            raise WspxError(
                f"vLLM needs about {need / gb:.0f} GB on GPU {len(gpus) - 1} and may use "
                f"{memory * total / gb:.0f} GB; raise [lab.vllm] memory or use a larger GPU"
            )
        return p
    name, total = gpus[0]
    hf = hf_bytes if hf_bytes is not None else checkpoint_bytes
    headroom = 0.08 * total + 3 * gb  # activations for probes, allocator slack
    room = (total - hf - headroom) / total
    memory = settings.memory or round(max(room, 0.0), 2)
    p = Plan(None, 0, memory)
    p.lines = [
        f"one GPU ({name}, {total / gb:.0f} GB) shared by both engines",
        f"transformers holds about {hf / gb:.1f} GB; vLLM may use {memory:.0%} "
        f"({memory * total / gb:.0f} GB) for a {checkpoint_bytes / gb:.1f} GB checkpoint",
    ]
    if memory * total < need or memory > room + 1e-6:
        raise WspxError(
            f"both engines do not fit on one {total / gb:.0f} GB {name}: transformers needs "
            f"about {hf / gb:.0f} GB and vLLM about {need / gb:.0f} GB more; use two GPUs "
            "([lab] gpu = 'H100:2'), a larger GPU, or engine = 'hf'"
        )
    return p


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class VLLMEngine(Engine):
    """Generation by a vLLM worker that holds the same weights as the lab."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        tree: WorkingTree,
        ir: Any,
        source: Path,
        settings: VllmSection,
        plan: Plan | None = None,
        http: httpx.Client | None = None,
        transport: Transport | None = None,
        state_key: Callable[[TreeState], str] = lambda s: "",
        python: str = VLLM_PYTHON,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        log: Path | None = None,
    ) -> None:
        super().__init__(model, name="vllm")
        self.tree, self.ir, self.source, self.settings = tree, ir, Path(source), settings
        self.state_key = state_key
        self.src = checkpoint.read_source(self.source)
        self.held = TreeState()
        """The state vLLM's weights are at (it loads the base checkpoint)."""
        self.syncs: list[dict[str, Any]] = []
        self.full_resync = False
        """Push every layer each time: for vLLM models whose loader takes only whole models."""
        self.agreement: dict[str, Any] | None = None
        self.proc: subprocess.Popen[bytes] | None = None
        self.log = log
        self.plan = plan
        if http is None:
            if plan is None:
                raise WspxError("the vLLM engine needs a plan or a worker connection")
            http = self._launch(python, dtype, trust_remote_code)
        self.http = http
        self.transport = transport or IPCTransport(
            torch.device("cuda", plan.vllm_gpu if plan else 0)
        )
        self.info = self._ready()
        self._names = sorted(
            [n for n, _ in model.named_parameters()] + sorted(self._buffers()),
            key=checkpoint._natural,
        )
        self._groups = checkpoint.groups(self._names)

    # ------------------------------------------------------------------ process

    def _launch(self, python: str, dtype: str, trust_remote_code: bool) -> httpx.Client:
        assert self.plan is not None
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if model_type in UNSERVED:
            raise WspxError(f"engine = 'vllm' cannot serve this model: {UNSERVED[model_type]}")
        if not Path(python).exists():
            raise WspxError(
                f"no vLLM environment at {python}: the vLLM engine needs the vLLM lab image "
                "(or WSPX_VLLM_PYTHON pointing at a Python with vllm installed)"
            )
        recipe_args, recipe_env = RECIPES.get(model_type or "", ([], {}))
        port = free_port()
        token = secrets.token_hex(16)
        folder = Path(tempfile.mkdtemp(prefix="wspx-vllm-"))
        token_file = folder / "token"
        token_file.write_text(token)
        token_file.chmod(0o600)
        argv = [
            "--model", str(self.source),
            "--dtype", dtype,
            "--max-model-len", str(self.settings.max_model_len),
            "--max-num-seqs", str(self.settings.max_num_seqs),
            "--gpu-memory-utilization", f"{self.plan.memory:.3f}",
            "--weight-transfer-config", json.dumps({"backend": "ipc"}),
            "--logprobs-mode", "processed_logprobs",
            "--generation-config", "auto",
            "--seed", "0",
            *(["--trust-remote-code"] if trust_remote_code else []),
            *recipe_args,
            *(self._speculative() if self.settings.speculative else []),
            *self.settings.args,
        ]  # fmt: skip
        env = {
            **os.environ,
            **recipe_env,
            "CUDA_VISIBLE_DEVICES": str(self.plan.vllm_gpu),
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        env.pop("PYTHONPATH", None)  # the lab's packages must not shadow vLLM's
        self.log = self.log or folder / "vllm.log"
        out = self.log.open("ab")
        cmd = [
            python,
            str(WORKER),
            "--port",
            str(port),
            "--token-file",
            str(token_file),
            "--",
            *argv,
        ]
        self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
        return httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            headers={"authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(None, connect=10.0),
        )

    def _speculative(self) -> list[str]:
        if self.src is None or not self.src.extra_files:
            raise WspxError(
                "[lab.vllm] speculative = true needs draft heads next to the checkpoint "
                "(such as mtp.safetensors), and this one has none"
            )
        return ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 1})]

    def _ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + READY_TIMEOUT_S
        last = ""
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise WspxError(f"the vLLM worker exited ({self.proc.returncode}):\n{self._tail()}")
            try:
                r = self.http.get("/ready", timeout=10.0)
            except httpx.TransportError as exc:
                last = str(exc)
            else:
                if r.status_code == 200:
                    info: dict[str, Any] = r.json()
                    return info
                if r.status_code == 500:
                    raise WspxError(
                        f"vLLM could not start: {r.json().get('detail')}\n{self._tail()}"
                    )
                last = r.text
            time.sleep(2.0)
        raise WspxError(f"vLLM did not become ready in {READY_TIMEOUT_S // 60} min ({last})")

    def _tail(self, n: int = 40) -> str:
        if self.log is None or not self.log.exists():
            return ""
        return "\n".join(self.log.read_text(errors="replace").splitlines()[-n:])

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.http.close()

    # ------------------------------------------------------------------ weights

    def _buffers(self) -> set[str]:
        from wspx.lab.export import _persistent_buffers

        return _persistent_buffers(self.model)

    def dirty(self, state: TreeState) -> list[str]:
        """In-memory tensors whose value under ``state`` differs from what vLLM holds."""
        held = self.held
        out = []
        for name in sorted(held.tensors() | state.tensors()):
            if (
                held.merged.get(name, ()) != state.merged.get(name, ())
                or held.adapter.get(name, ()) != state.adapter.get(name, ())
                or (name in held.promote) != (name in state.promote)
            ):
                out.append(name)
        return out

    def _value(self, name: str, state: TreeState, live: dict[str, torch.Tensor]) -> torch.Tensor:
        """What vLLM should hold for ``name``: the live tensor, plus adapter deltas (which
        transformers runs as a side path) and rounded back from promote mode's fp32."""
        if name in state.adapter or name in state.promote:
            extra = [self.tree.delta(o) for o in state.adapter.get(name, ())]
            return self.tree.preview(
                name, extra, dtype=self.tree.dtype_of(name), device=live[name].device
            )
        return live[name]

    def sync(self) -> dict[str, Any] | None:
        """Push every layer whose tensors changed since the last push, then check agreement."""
        state = self.tree.state
        changed = self.dirty(state)
        if not changed:
            return None
        served = self.tree.served
        if (
            served is None
            and self.src is not None
            and self.src.quantized
            and getattr(self.model, "hf_quantizer", None) is None
        ):
            method = (self.src.quantization or {}).get("quant_method", "quantized")
            raise WspxError(
                f"vLLM serves this checkpoint {method} in a format wspx cannot write edits "
                "in, so the edit cannot reach vLLM; use engine = 'hf'"
            )
        if served is not None:
            for name in changed:
                shape = tuple(dict(self.model.named_parameters())[name].shape)
                if name in state.promote and served.quantized(name, shape):
                    raise WspxError(
                        f"{name} is kept exact in bf16 (promote mode), and vLLM holds it in the "
                        "checkpoint's quantized format; edit it merged, or use engine = 'hf'"
                    )
        params = dict(self.model.named_parameters())
        packed = [
            n
            for n in changed
            if n in params and not params[n].dtype.is_floating_point and n in state.adapter
        ]
        if packed:  # an int4/int8 weight plus a float delta is not a weight vLLM can load
            raise WspxError(
                f"{packed[0]} is stored as {params[packed[0]].dtype} integers, and an adapter "
                "edit to it cannot be merged into what vLLM holds; use engine = 'hf' for "
                "adapter edits on this checkpoint"
            )
        touched = set(changed)
        groups = (
            list(self._groups)
            if self.full_resync
            else [g for g in self._groups if touched & set(g)]
        )
        live: dict[str, torch.Tensor] = dict(self.model.named_parameters())
        live.update(dict(self.model.named_buffers()))
        started = time.perf_counter()

        def chunks() -> Iterator[dict[str, torch.Tensor]]:
            for group in groups:
                plain, stored = {}, {}
                for n in group:
                    shape = tuple(live[n].shape)
                    if served is not None and served.quantized(n, shape):
                        # the checkpoint's own format: the edited codes, or the source's bytes
                        device = live[n].device
                        stored.update(served.stored(n) or served.source_parts(n, shape, device))
                    else:
                        plain[n] = self._value(n, state, live)
                yield {**checkpoint.to_checkpoint(self.model, plain), **stored}

        version = self.state_key(state) or f"sync-{len(self.syncs) + 1}"
        try:
            sent = self.transport.send(self.http, chunks(), version)
        except WspxError as exc:
            if self.full_resync or "Missing checkpoint tensors" not in str(exc):
                raise
            # seen with vLLM's Inkling2: its loader takes only a whole model per update. The
            # failed update left vLLM mid-reload; a full one (start → update → finish) mends it
            self.full_resync = True
            groups = list(self._groups)
            sent = self.transport.send(self.http, chunks(), version)
        self.held = state
        record = {
            "version": version,
            "tensors": len(changed),
            "layers": len(groups),
            "bytes": sent,
            "seconds": round(time.perf_counter() - started, 3),
        }
        self.syncs.append(record)
        return record

    # ------------------------------------------------------------------ generation

    def _requests(self, text: Text, prompts: Sequence[Prompt], batch: Any) -> list[dict[str, Any]]:
        out = []
        for row, prompt in enumerate(prompts):
            if is_media(prompt) and text.media is not None:
                rendered, images, videos = text.media._render(text.media.resolve(prompt))
                out.append(
                    {
                        "prompt": rendered,
                        "images": [_png(im) for im in images],
                        "videos": [_npy(frames, meta) for frames, meta in videos],
                        "expect": int(batch.mask[row].sum()),
                    }
                )
            else:
                keep = batch.mask[row].bool()
                out.append({"ids": [int(t) for t in batch.ids[row][keep].tolist()]})
        return out

    def complete_raw(
        self,
        requests: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        watch: Sequence[int] = (),
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        """Send requests to the worker; rows in request order, and the wall time."""
        body = {
            "requests": requests,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "watch": [int(w) for w in watch],
        }
        rows: dict[int, dict[str, Any]] = {}
        seconds = 0.0
        with self.http.stream("POST", "/generate", json=body) as r:
            if r.status_code != 200:
                r.read()
                raise WspxError(f"vLLM worker /generate: {r.status_code} {r.text[:400]}")
            for line in r.iter_lines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("done"):
                    seconds = float(item.get("seconds", 0.0))
                    continue
                if "error" in item:
                    raise WspxError(f"vLLM could not generate prompt {item['i']}: {item['error']}")
                rows[int(item["i"])] = item
                if progress is not None:
                    progress(len(rows), len(requests))
        missing = [i for i in range(len(requests)) if i not in rows]
        if missing:
            raise WspxError(f"the vLLM worker returned no answer for prompts {missing[:5]}")
        return [rows[i] for i in range(len(requests))], seconds

    def generate_scored(
        self,
        text: Text,
        prompts: Sequence[Prompt],
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        watch: Sequence[int] = (),
        progress: Callable[[int, int], None] | None = None,
    ) -> Rows:
        self.ensure(text)
        return self._rows(
            text,
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            watch=watch,
            progress=progress,
        )

    def complete(
        self,
        text: Text,
        prompts: Sequence[Prompt],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        progress: Callable[[int, int], None] | None = None,
        batch_size: int = 8,
    ) -> list[tuple[list[int], str, int]]:
        """All prompts at once: vLLM batches continuously, and progress counts answers."""
        del batch_size
        self.ensure(text)
        stops = analysis._stop_ids(self.model, text)
        rows = self._rows(
            text,
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            watch=(),
            progress=progress,
        )
        out = []
        for (ids, _), raw in zip(rows, self.last_rows, strict=True):
            cut = next((i for i, t in enumerate(ids) if t in stops), None)
            finish = "length" if raw.get("finish") == "length" else "stop"
            out.append((ids[:cut] if cut is not None else ids, finish, int(raw["prompt_tokens"])))
        return out

    def ensure(self, text: Text) -> None:
        """vLLM holds the live weights; after a push, both engines are compared again."""
        pushed = self.sync()
        if (pushed or self.agreement is None) and self.settings.agreement_prompts:
            self.check_agreement(
                text, agreement_prompts(text, self.ir, self.settings.agreement_prompts)
            )

    def _rows(
        self,
        text: Text,
        prompts: Sequence[Prompt],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        watch: Sequence[int],
        progress: Callable[[int, int], None] | None = None,
    ) -> Rows:
        batch = text.batch(prompts, "cpu")  # the same rendering transformers gets, recorded
        requests = self._requests(text, prompts, batch)
        rows, seconds = self.complete_raw(
            requests,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            watch=watch,
            progress=progress,
        )
        for req, row in zip(requests, rows, strict=True):
            if "expect" in req and req["expect"] != row["prompt_tokens"]:
                note = (
                    f"a media prompt is {row['prompt_tokens']} tokens in vLLM and "
                    f"{req['expect']} in transformers: their processors differ"
                )
                if note not in self.notes:
                    self.notes.append(note)
        out = [(list(r["ids"]), [list(s) for s in r.get("watched", [])]) for r in rows]
        if text.io is not None and batch.records:
            text.io.outputs(
                batch.records,
                [ids for ids, _ in out],
                settings={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "engine": "vllm",
                },
                stop_ids=analysis._stop_ids(self.model, text),
                seconds=seconds,
            )
        self.last_rows = rows
        return out

    # ------------------------------------------------------------------ agreement

    def check_agreement(self, text: Text, prompts: Sequence[Prompt]) -> dict[str, Any]:
        """Greedy answers from both engines on the same prompts: how many match exactly,
        how long they agree, and the KL of the first token over transformers' top 20."""
        self.sync()
        batch = text.batch(prompts, next(self.model.parameters()).device)
        with torch.no_grad():
            logits = analysis.forward(self.model, self.ir, batch.ids, batch.mask, batch.extra)
        top = sorted({int(i) for row in logits[:, -1].topk(AGREEMENT_TOP).indices for i in row})
        hf = analysis.generate_scored(
            self.model, text, prompts, max_tokens=AGREEMENT_TOKENS, watch=top
        )
        vl = self._rows(
            text,
            prompts,
            max_tokens=AGREEMENT_TOKENS,
            temperature=0.0,
            top_p=1.0,
            seed=0,
            watch=top,
        )
        exact, prefix, kls = 0, [], []
        for (a, wa), (b, wb) in zip(hf, vl, strict=True):
            exact += a == b
            same = next(
                (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                min(len(a), len(b)),
            )
            prefix.append(same / max(len(a), len(b), 1))
            if wa and wb:
                kls.append(_kl(wa[0], wb[0]))
        result = {
            "prompts": len(prompts),
            "exact": exact,
            "agree": round(sum(prefix) / max(len(prefix), 1), 3),
            "kl_first": round(sum(kls) / len(kls), 6) if kls else None,
            "weights": self.weights(),
        }
        self.agreement = result
        return result

    # ------------------------------------------------------------------ provenance

    def provenance(self) -> dict[str, Any]:
        gpu = self.info.get("gpu") or {}
        return {
            "engine": "vllm",
            "vllm": self.info.get("vllm"),
            "gpu": gpu.get("name"),
            "quantization": self.info.get("quantization"),
            "weights": self.weights(),
            "agreement": self.agreement,
        }

    def label(self) -> str:
        return f"vLLM {self.info.get('vllm', '?')}"

    def weights(self) -> str:
        """Which weights vLLM holds: ``base``, or the key of the state last pushed."""
        return (self.state_key(self.held) or "edited") if self.held.tensors() else "base"

    def describe(self) -> str:
        """One status line: where it runs, what it holds, and how well it agrees."""
        gpu = (self.info.get("gpu") or {}).get("name") or "GPU"
        line = f"engine: {self.label()} on {gpu}, weights {self.weights()}"
        if self.syncs:
            last = self.syncs[-1]
            line += f" (last sync {last['layers']} layer(s), {last['bytes'] / 2**20:.0f} MB in {last['seconds']:.1f}s)"
        a = self.agreement
        if a:
            line += f"; agrees with transformers on {a['exact']}/{a['prompts']} greedy answers"
            if a.get("kl_first") is not None:
                line += f", first-token KL {a['kl_first']:.2g}"
        if self.plan is not None:
            line += "\n  " + self.plan.describe().replace("\n", "\n  ")
        return line


def _kl(p_log: Sequence[float], q_log: Sequence[float]) -> float:
    """KL(p ‖ q) over the watched tokens, each side renormalized over them."""
    p = torch.tensor(list(p_log), dtype=torch.float64)
    q = torch.tensor(list(q_log), dtype=torch.float64)
    finite = torch.isfinite(p) & torch.isfinite(q)
    if not finite.any():
        return math.inf
    p, q = p[finite], q[finite]
    p, q = p - torch.logsumexp(p, 0), q - torch.logsumexp(q, 0)
    return float((p.exp() * (p - q)).sum())


def _png(image: Any) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _npy(frames: Any, meta: Any) -> dict[str, Any]:
    import numpy as np

    buf = io.BytesIO()
    np.save(buf, np.asarray(frames))
    return {"npy": base64.b64encode(buf.getvalue()).decode(), "fps": getattr(meta, "fps", None)}
