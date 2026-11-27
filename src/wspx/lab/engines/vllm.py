"""The vLLM engine: a worker process beside the lab that serves the lab's weights.

The worker (:mod:`wspx.lab.engines.vllm_worker`) runs with vLLM's own Python, since vLLM
pins its own torch. Before each generation the lab pushes every layer an edit, checkout,
discard or merge changed since the last push, under the checkpoint's tensor names
(:mod:`wspx.lab.checkpoint`) and in the checkpoint's own format (quantized tensors as
their codes and scales, :mod:`wspx.lab.served`), through vLLM's weight-transfer engine
over CUDA IPC. vLLM reloads layer by layer and keeps every layer a push leaves out, so a
push costs the layers it changes, not the model. A whole layer goes at once: a module that
received only some of its tensors would be left with the rest uninitialized.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import os
import pickle
import re
import secrets
import signal
import socket
import subprocess
import sys
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
from wspx.lab.offload import device_of, release_gpu
from wspx.lab.text import Prompt, Text
from wspx.lab.tree import TreeState, WorkingTree

WORKER = Path(__file__).with_name("vllm_worker.py")
EXTENSION = Path(__file__).with_name("vllm_ext.py")
EXTENSION_CLS = "wspx_vllm_ext.WspxWorkerExtension"
VLLM_PYTHON = os.environ.get("WSPX_VLLM_PYTHON", "/opt/vllm/bin/python")
READY_TIMEOUT_S = 45 * 60
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


Chunks = Iterable[dict[str, torch.Tensor]]


class Transport:
    """How tensors reach the worker. Over CUDA IPC normally; tests swap in their own.

    Before the update finishes, while vLLM reports modules it has given only part of their
    tensors: ``supply`` sends the checkpoint tensors vLLM names as completing them, and
    failing that, ``more`` sends further chunks, in order. Returns the bytes sent and how
    many of ``more`` it took."""

    def send(
        self,
        http: httpx.Client,
        chunks: Chunks,
        version: str,
        more: Sequence[Callable[[], Chunks]] = (),
        supply: Callable[[list[str]], Chunks] | None = None,
    ) -> tuple[int, int]:
        raise NotImplementedError


@dataclass
class IPCTransport(Transport):
    """vLLM's "trainer without vLLM over HTTP" IPC contract, unpacked: every tensor of a
    chunk (one layer) goes as its own CUDA IPC handle in one update, so each update is one
    ``load_weights`` call on whole layers."""

    device: torch.device
    _inited: bool = False
    reported: list[str] = field(default_factory=list)
    """The modules vLLM last reported as given only part of their tensors."""

    def send(
        self,
        http: httpx.Client,
        chunks: Chunks,
        version: str,
        more: Sequence[Callable[[], Chunks]] = (),
        supply: Callable[[list[str]], Chunks] | None = None,
    ) -> tuple[int, int]:
        uuid = str(torch.cuda.get_device_properties(self.device).uuid)
        if not self._inited:
            _post(http, "/weights/init", {"packed": False})
            self._inited = True
        _post(http, "/weights/start", {})
        sent = used = rounds = 0
        try:
            sent += self._updates(http, chunks, uuid)
            while True:
                missing, need = _incomplete(http)
                if not missing:
                    break
                self.reported = missing
                if need and supply is not None and rounds < 4:
                    sent += self._updates(http, supply(need), uuid)
                    rounds += 1
                    continue
                if used == len(more):
                    raise WspxError(
                        "vLLM has only part of the tensors of "
                        f"{', '.join(missing[:3])}{' …' if len(missing) > 3 else ''}"
                    )
                sent += self._updates(http, more[used](), uuid)
                used += 1
        except BaseException:
            # end vLLM's update so it serves again; what it holds is now unknown
            with contextlib.suppress(Exception):
                _post(http, "/weights/finish", {"version": f"{version}-failed"})
            raise
        _post(http, "/weights/finish", {"version": version})
        torch.cuda.ipc_collect()  # type: ignore[no-untyped-call]
        return sent, used

    def _updates(self, http: httpx.Client, chunks: Chunks, uuid: str) -> int:
        from torch.multiprocessing.reductions import reduce_tensor

        sent = 0
        for chunk in chunks:
            if not chunk:
                continue
            keep: list[torch.Tensor] = []  # exported memory lives until the update returns
            names: list[str] = []
            dtypes: list[str] = []
            shapes: list[list[int]] = []
            handles: list[dict[str, Any]] = []
            for name, tensor in chunk.items():
                t = tensor.detach().to(self.device, non_blocking=True).contiguous()
                keep.append(t)
                names.append(name)
                dtypes.append(str(t.dtype).removeprefix("torch."))
                shapes.append(list(t.shape))
                handles.append({uuid: reduce_tensor(t)[1]})  # type: ignore[no-untyped-call]
                sent += t.numel() * t.element_size()
            torch.cuda.synchronize(self.device)  # the copies land before the worker reads
            info = {
                "names": names,
                "dtype_names": dtypes,
                "shapes": shapes,
                "ipc_handles_pickled": base64.b64encode(pickle.dumps(handles)).decode(),
            }
            _post(http, "/weights/update", {"update_info": info})
            del keep, handles
            # vLLM has copied this layer in: free what was exported for it now, not when the
            # whole push ends (twelve layers of Inkling held at once ran vLLM out of memory)
            torch.cuda.ipc_collect()  # type: ignore[no-untyped-call]
        return sent


def _incomplete(http: httpx.Client) -> tuple[list[str], list[str]]:
    """Modules the running update has given only part of their tensors, and the checkpoint
    tensors that complete them."""
    r = http.get("/weights/incomplete")
    if r.status_code != 200:  # a worker without the check: nothing to say
        return [], []
    body = r.json()
    return list(body.get("modules", [])), list(body.get("need", []))


def _post(http: httpx.Client, path: str, body: dict[str, Any]) -> Any:
    r = http.post(path, json=body)
    if r.status_code != 200:
        raise WspxError(f"vLLM worker {path}: {r.status_code} {_detail(r)}")
    return r.json()


def _detail(r: httpx.Response) -> str:
    """What went wrong in the worker, as it said it (seen live: a bare "Internal Server
    Error" hid a CUDA out-of-memory error inside vLLM)."""
    try:
        body = r.json()
    except ValueError:
        return r.text[:1500]
    detail = body.get("detail") if isinstance(body, dict) else None
    return str(detail if detail is not None else body)[:1500]


@dataclass
class Plan:
    """Where each engine runs and how much of its GPU vLLM may take."""

    hf_device: str | None
    """``device_map`` for transformers (``None``: spread over every GPU; ``cpu``: host memory,
    streamed to the GPU a layer at a time)."""
    vllm_gpu: int
    memory: float
    lines: list[str] = field(default_factory=list)
    stream: bool = False
    """transformers' copy stays in host memory and runs a layer at a time on the GPU."""

    def describe(self) -> str:
        return "engine plan:\n" + "\n".join(f"  {line}" for line in self.lines)


def plan(
    settings: VllmSection,
    *,
    checkpoint_bytes: int,
    gpus: list[tuple[str, int]] | None = None,
    hf_bytes: int | None = None,
    resident_bytes: int | None = None,
    weightless: bool = False,
    layer_bytes: int = 0,
) -> Plan:
    """Decide where vLLM runs before transformers loads (``gpus``: name and memory per GPU).

    With two or more GPUs, transformers keeps the first and vLLM the last. With one, both
    share it when transformers' copy (``hf_bytes``) leaves vLLM room for a useful KV cache;
    otherwise transformers' copy stays in host memory and only ``resident_bytes`` (what lies
    outside its layer stacks, and a layer in flight) sit on the GPU. ``weightless``: the lab
    holds no copy at all (it reads the checkpoint), so vLLM gets all but ``resident_bytes``.
    ``layer_bytes`` (the checkpoint's largest layer) is kept free for pushes on a shared GPU:
    the lab's copy of the layer in flight, and vLLM's own while it reloads it."""
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
    # activations for probes, allocator slack, and a layer being pushed: the lab's copy and
    # vLLM's while it reloads it (vLLM's share does not cover what a reload allocates)
    headroom = 0.08 * total + 3 * gb + 3 * layer_bytes
    if weightless:
        lab = resident_bytes or 0
        room = (total - lab - headroom) / total
        memory = settings.memory or _floor(room)
        if memory * total < need or memory > room + 1e-6:
            raise WspxError(
                f"vLLM needs about {need / gb:.0f} GB of the {total / gb:.0f} GB {name} and may "
                f"use {memory * total / gb:.0f} GB beside the lab's {lab / gb:.1f} GB; use a "
                "larger GPU or two ([lab] gpu = 'H100:2')"
            )
        p = Plan(None, 0, memory, stream=True)
        p.lines = [
            f"one GPU ({name}, {total / gb:.0f} GB): vLLM holds the model, and may use "
            f"{memory:.0%} ({memory * total / gb:.0f} GB) for a {checkpoint_bytes / gb:.1f} GB "
            "checkpoint",
            f"the lab holds no copy: {lab / gb:.1f} GB on the GPU (what lies outside the layer "
            "stacks, a layer in flight); it reads the checkpoint a layer at a time for probes",
        ]
        return p
    hf = hf_bytes if hf_bytes is not None else checkpoint_bytes
    useful = need + max(0.0, 0.15 * total - 4 * gb)  # a KV cache for batched evals
    room = (total - hf - headroom) / total
    if room * total >= useful or (resident_bytes is None and room * total >= need):
        memory = settings.memory or _floor(room)
        p = Plan(None, 0, memory)
        p.lines = [
            f"one GPU ({name}, {total / gb:.0f} GB) shared by both engines",
            f"transformers holds about {hf / gb:.1f} GB; vLLM may use {memory:.0%} "
            f"({memory * total / gb:.0f} GB) for a {checkpoint_bytes / gb:.1f} GB checkpoint",
        ]
        if memory * total < need or memory > room + 1e-6:
            raise WspxError(
                f"both engines do not fit on one {total / gb:.0f} GB {name}: transformers "
                f"needs about {hf / gb:.0f} GB and vLLM about {need / gb:.0f} GB more; use "
                "two GPUs ([lab] gpu = 'H100:2'), a larger GPU, or engine = 'hf'"
            )
        return p
    if resident_bytes is not None:
        streamed = (total - resident_bytes - headroom) / total
        memory = settings.memory or _floor(streamed)
        if memory * total >= need and memory <= streamed + 1e-6:
            p = Plan("cpu", 0, memory, stream=True)
            p.lines = [
                f"one GPU ({name}, {total / gb:.0f} GB): vLLM may use {memory:.0%} "
                f"({memory * total / gb:.0f} GB) for a {checkpoint_bytes / gb:.1f} GB checkpoint",
                f"transformers' {hf / gb:.1f} GB copy stays in host memory and runs a layer at "
                f"a time on the GPU ({resident_bytes / gb:.1f} GB stays there)",
            ]
            return p
    raise WspxError(
        f"both engines do not fit on one {total / gb:.0f} GB {name}: vLLM needs about "
        f"{need / gb:.0f} GB, and transformers {hf / gb:.0f} GB (or "
        f"{(resident_bytes or hf) / gb:.0f} GB streamed from host memory); use two GPUs "
        "([lab] gpu = 'H100:2'), a larger GPU, or engine = 'hf'"
    )


def _floor(share: float) -> float:
    """A share of the GPU in hundredths, never above ``share`` (rounding up overcommits)."""
    return max(math.floor(share * 100) / 100, 0.0)


def supported_archs(python: str = VLLM_PYTHON) -> set[str] | None:
    """Architectures this vLLM build serves (``None``: it could not be asked). Read from
    ``wspx-archs.json`` beside the vLLM environment when the image wrote one, else asked."""
    baked = Path(python).parent.parent / "wspx-archs.json"
    cache = Path(tempfile.gettempdir()) / "wspx-vllm-archs.json"
    for f in (baked, cache):
        if f.is_file():
            try:
                return set(json.loads(f.read_text()))
            except ValueError:
                pass
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = (
        "import json; from vllm import ModelRegistry; "
        "print(json.dumps(sorted(ModelRegistry.get_supported_archs())))"
    )
    try:
        out = subprocess.run(
            [python, "-P", "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            check=False,
        )
        archs = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    with contextlib.suppress(OSError):
        cache.write_text(json.dumps(archs))
    return set(archs)


def can_serve(source: Path, *, python: str = VLLM_PYTHON) -> tuple[bool, str]:
    """Whether vLLM can serve this checkpoint with wspx's edits reaching it, and if not why."""
    if not torch.cuda.is_available():
        return False, "there is no GPU"
    if not Path(python).exists():
        return False, "this lab image has no vLLM (the vLLM lab image does)"
    config_file = Path(source) / "config.json"
    if not config_file.is_file():
        return False, "the model has no config.json (a custom module)"
    config = json.loads(config_file.read_text())
    model_type = config.get("model_type")
    if model_type in UNSERVED:
        return False, UNSERVED[model_type]
    archs = list(config.get("architectures") or [])
    known = supported_archs(python)
    if known is not None and archs and not any(a in known for a in archs):
        return False, f"this vLLM has no {archs[0]}"
    from wspx.lab import served

    why = served.unwritable(checkpoint.read_source(Path(source)))
    if why:
        return False, why
    return True, ""


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
        self.stale = False
        """A push failed partway, so what vLLM holds is unknown: the next push sends every
        layer."""
        self.survival: dict[str, Any] = {}
        """Adapter edits on tensors stored quantized, as vLLM serves them (merged)."""
        self._by_layer: dict[str, list[str]] | None = None
        self._producers: dict[str, str] | None = None
        """Checkpoint tensor → the lab tensor that produces it."""
        self.with_outside = False
        """Pushes also send what lies outside the layer stacks (a model whose vLLM loader
        maps such tensors into every layer), learned from vLLM on the first push."""
        self.agreement: dict[str, Any] | None = None
        self.proc: subprocess.Popen[bytes] | None = None
        self.log = log
        self.plan = plan
        self.restarts = 0
        self._launch_args = (python, dtype, trust_remote_code)
        self._closing = False
        if http is None:
            if plan is None:
                raise WspxError("the vLLM engine needs a plan or a worker connection")
            http = self._launch(python, dtype, trust_remote_code)
        self.http = http
        self.transport = transport or IPCTransport(
            torch.device("cuda", plan.vllm_gpu if plan else 0)
        )
        self.info = self._ready()
        self._buffer_names = set(self._buffers())
        self._names = sorted(
            [n for n, _ in model.named_parameters()] + sorted(self._buffer_names),
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
            "--worker-extension-cls", EXTENSION_CLS,
            "--logprobs-mode", "processed_logprobs",
            "--generation-config", "auto",
            "--seed", "0",
            *(["--trust-remote-code"] if trust_remote_code else []),
            *recipe_args,
            *(self._speculative() if self.settings.speculative else []),
            *self.settings.args,
        ]  # fmt: skip
        # The worker and its extension run from a folder of their own: next to this file,
        # ``import vllm`` would find wspx's ``vllm.py``, not vLLM.
        worker = folder / "wspx_vllm_worker.py"
        worker.write_text(WORKER.read_text())
        (folder / "wspx_vllm_ext.py").write_text(EXTENSION.read_text())
        env = {
            **os.environ,
            **recipe_env,
            "CUDA_VISIBLE_DEVICES": str(self.plan.vllm_gpu),
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            # layer by layer: an update holds only the layers that changed, and vLLM puts
            # back the ones it leaves out (direct reload takes whole models only)
            "VLLM_RELOAD_MODE": "layerwise",
            "HF_HUB_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            # only the extension: the lab's packages must not shadow vLLM's
            "PYTHONPATH": str(folder),
        }
        self.log = self.log or folder / "vllm.log"
        out = self.log.open("ab")
        cmd = [
            python,
            "-P",
            str(worker),
            "--port",
            str(port),
            "--token-file",
            str(token_file),
            "--",
            *argv,
        ]
        # a session of its own: closing it ends vLLM's engine process too, not only the worker
        self.proc = subprocess.Popen(
            cmd, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        if not self._closing:
            import atexit

            atexit.register(self.close)  # a lab that ends any way frees vLLM's GPU memory
            self._closing = True
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
        proc = self.proc
        if proc is not None and proc.poll() is None:
            _signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _signal_group(proc, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=30)
        if proc is not None:
            _signal_group(proc, signal.SIGKILL)  # its engine process, if it outlived it
        self.http.close()

    def _failed(self, exc: BaseException) -> str:
        """Why a push failed: the worker's own error, and what vLLM logged about it."""
        lines = [
            line.strip()
            for line in self._tail(400).splitlines()
            if re.search(r"Error|error|out of memory|Traceback", line)
        ][-6:]
        text = f"the push into vLLM failed: {exc}"
        if lines:
            text += "\nvLLM's log:\n  " + "\n  ".join(line[:300] for line in lines)
        if "out of memory" in (text.lower()):
            text += (
                "\n(vLLM ran out of GPU memory reloading the layers: lower [lab.vllm] memory "
                "to leave the lab more room)"
            )
        return text

    def restart(self, why: str) -> None:
        """Start vLLM again from the checkpoint: after a push failed partway, what it holds
        is unknown (and a failed reload can leave its memory taken). It then holds the base,
        so the next push sends only what the session has edited."""
        if self.proc is None:
            raise WspxError(f"vLLM needs a restart ({why}), and this lab did not start it")
        self.close()
        _wait_for_gpu_memory(self.plan)
        self.http = self._launch(*self._launch_args)
        if isinstance(self.transport, IPCTransport):
            self.transport._inited = False
        self.info = self._ready()
        self.held = TreeState()
        self.stale = False
        self.agreement = None
        self.restarts += 1
        self.notes.append(f"vLLM was restarted ({why}); it holds the base checkpoint again")

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

    def _value(self, name: str, state: TreeState) -> torch.Tensor:
        """What vLLM should hold for ``name``: the live tensor, plus adapter deltas (which
        transformers runs as a side path) and rounded back from promote mode's fp32."""
        if name in self._buffer_names:
            buffer: torch.Tensor = self.model.get_buffer(name)
            return buffer
        if name in state.adapter or name in state.promote:
            extra = [self.tree.delta(o) for o in state.adapter.get(name, ())]
            return self.tree.preview(
                name, extra, dtype=self.tree.dtype_of(name), device=self.tree.device(name)
            )
        live: torch.Tensor = self.tree.params[name]
        return live

    def _supply(self, need: list[str], state: TreeState) -> Chunks:
        """The checkpoint tensors ``need`` names, at their current values: converted from
        the tensors the lab holds that produce them, or as the source stores them."""
        served = self.tree.served
        if self._producers is None:
            self._producers = {}
            for n in self._names:
                shape = self._shape(n)
                for ck in checkpoint.names_in_checkpoint(self.model, n, shape):
                    fmt = served.formats.get(ck) if served is not None else None
                    for part in fmt.parts if fmt is not None else [ck]:
                        self._producers.setdefault(part, n)
        by_lab: dict[str, set[str]] = {}
        stored: list[str] = []
        for ck in need:
            producer = self._producers.get(ck)
            if producer is None:
                stored.append(ck)
            else:
                by_lab.setdefault(producer, set()).add(ck)
        out: dict[str, torch.Tensor] = {}
        for n, cks in by_lab.items():
            shape = self._shape(n)
            if served is not None and served.quantized(n, shape):
                parts = (
                    self._stored_value(n, state)
                    or served.stored(n)
                    or served.source_parts(n, shape, torch.device("cpu"))
                )
                out.update({k: v for k, v in parts.items() if k in cks})
                continue
            try:
                converted = checkpoint.to_checkpoint(self.model, {n: self._value(n, state)})
            except Exception:  # a conversion that needs others' tensors: as stored
                stored += sorted(cks)
                continue
            converted, _ = checkpoint.as_stored(self.src, converted)
            for k in cks:
                if k in converted:
                    out[k] = converted[k]
                else:
                    stored.append(k)
        out.update(self._source_tensors([k for k in stored if k not in out]))
        return [out]

    def _source_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        if self.src is None or not names:
            return {}
        from safetensors import safe_open

        out: dict[str, torch.Tensor] = {}
        for fname in sorted(self.src.weight_files):
            with safe_open(str(self.src.path / fname), framework="pt") as fh:
                keys = set(fh.keys())
                for k in names:
                    if k in keys and k not in out:
                        out[k] = fh.get_tensor(k)
        return out

    def _rest_of_layers(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The checkpoint tensors of the layers in ``batch`` that the lab does not hold
        (static activation scales, KV-cache scales and the like), as the source stores them:
        vLLM reprocesses a module only once every tensor of it has arrived."""
        if self.src is None or not self.src.tensors:
            return {}
        if self._by_layer is None:
            self._by_layer = {}
            for name in self.src.tensors:
                m = checkpoint._LAYER.match(name)
                if m:
                    self._by_layer.setdefault(m.group(1), []).append(name)
        layers = {m.group(1) for k in batch if (m := checkpoint._LAYER.match(k))}
        missing = [k for layer in layers for k in self._by_layer.get(layer, []) if k not in batch]
        return self._source_tensors(missing)

    def _shape(self, name: str) -> tuple[int, ...]:
        if name in self._buffer_names:
            return tuple(self.model.get_buffer(name).shape)
        return self.tree.shape(name)

    def _stored_value(self, name: str, state: TreeState) -> dict[str, torch.Tensor] | None:
        """A tensor stored quantized that has adapter edits: transformers runs them as a
        side path, and vLLM gets the weight with them merged, in the stored format."""
        served = self.tree.served
        if served is None or name not in state.adapter:
            return None
        value = self._value(name, state)
        encoded = served.encode(name, self.tree.base(name).to(value.device), value)
        if encoded is None:
            return None
        parts, survival, _ = encoded
        self.survival[name] = survival
        out: dict[str, torch.Tensor] = parts
        return out

    def sync(self, *, retry: bool = False) -> dict[str, Any] | None:
        """Push every layer whose tensors changed since the last push. A push that fails
        restarts vLLM (it then holds the base) and is tried once more from there."""
        if self.stale and self.proc is not None and not retry:
            self.restart("an earlier push failed partway")
        state = self.tree.state
        changed = self.dirty(state)
        if not changed and not self.stale:
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
                shape = self.tree.shape(name)
                if name in state.promote and served.quantized(name, shape):
                    raise WspxError(
                        f"{name} is kept exact in bf16 (promote mode), and vLLM holds it in the "
                        "checkpoint's quantized format; edit it merged, or use engine = 'hf'"
                    )
        params = self.tree.meta
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
        groups = list(self._groups) if self.stale else [g for g in self._groups if touched & set(g)]
        release_gpu()  # what probes left cached: a push needs it
        outside = [g for g in self._groups if not checkpoint._LAYER.match(g[0])]
        if self.with_outside:  # this model maps tensors kept outside its layers into them
            groups += [g for g in outside if g not in groups]
        started = time.perf_counter()

        def chunks(groups: list[list[str]]) -> Iterator[dict[str, torch.Tensor]]:
            for group in groups:
                plain: dict[str, torch.Tensor] = {}
                stored: dict[str, torch.Tensor] = {}
                for n in group:
                    shape = self._shape(n)
                    if served is not None and served.quantized(n, shape):
                        # the checkpoint's own format: the edited codes, or the source's bytes
                        stored.update(
                            self._stored_value(n, state)
                            or served.stored(n)
                            or served.source_parts(n, shape, torch.device("cpu"))
                        )
                    else:
                        plain[n] = self._value(n, state)
                # vLLM loads only the dtype the checkpoint stores (transformers widens some)
                as_stored, _ = checkpoint.as_stored(
                    self.src, checkpoint.to_checkpoint(self.model, plain)
                )
                batch = {**as_stored, **stored}
                yield {**batch, **self._rest_of_layers(batch)}

        sent_groups = [list(g) for g in groups]

        def these(level: Callable[[], list[list[str]]]) -> Callable[[], Chunks]:
            def go() -> Chunks:
                # never a group already sent in this update: vLLM has processed it into its
                # kernel layout, and a second copy of its checkpoint tensors does not fit
                gs = [g for g in level() if g not in sent_groups]
                sent_groups.extend(gs)
                return chunks(gs)

            return go

        more = [these(lambda: outside), these(lambda: list(self._groups))]
        version = self.state_key(state) or f"sync-{len(self.syncs) + 1}"
        try:
            sent, used = self.transport.send(
                self.http, chunks(groups), version, more, lambda need: self._supply(need, state)
            )
        except Exception as exc:
            self.stale = True
            if self.proc is None or retry:  # not ours to restart, or it failed again
                raise WspxError(self._failed(exc)) from exc
            first = self._failed(exc)
            print(f"wspx lab: {first}; restarting vLLM", file=sys.stderr, flush=True)
            self.restart(f"a push failed: {str(exc)[:200]}")
            try:
                return self.sync(retry=True)
            except WspxError as again:
                raise WspxError(f"{first}\nafter restarting vLLM, again: {again}") from exc
        finally:
            release_gpu()
        if used:
            self.with_outside = True
            seen = ", ".join(getattr(self.transport, "reported", [])[:3])
            if used == 1:
                self.notes.append(
                    "vLLM maps tensors kept outside the layer stack into every layer"
                    + (f" ({seen})" if seen else "")
                    + ": each push sends them too"
                )
        self.stale = False
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
        again = pushed is not None and self.settings.check_pushes
        if (again or self.agreement is None) and self.settings.agreement_prompts:
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
        batch = text.batch(prompts, device_of(self.model))
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


def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    """Signal the worker's process group: vLLM's engine process is a child of the worker."""
    if proc.pid <= 0:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, sig)


def _wait_for_gpu_memory(plan: Plan | None, timeout: float = 120.0) -> None:
    """After vLLM stops, until its share of the GPU is free again (the driver releases a
    process's memory a moment after it exits)."""
    if plan is None or not torch.cuda.is_available():
        return
    device = torch.device("cuda", plan.vllm_gpu)
    want = plan.memory * torch.cuda.get_device_properties(device).total_memory * 0.95
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        free, _ = torch.cuda.mem_get_info(device)
        if free >= want:
            return
        time.sleep(1.0)


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
