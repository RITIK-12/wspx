from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from wspx import project
from wspx.errors import ConfigError, WspxError
from wspx.project import Project


def make_project(root: Path, exclude: list[str] | None = None) -> Project:
    toml = '[project]\nname = "demo"\n'
    if exclude:
        toml += f"exclude = {json.dumps(exclude)}\n"
    (root / "wspx.toml").write_text(toml)
    for rel, content in {
        "data/loops.jsonl": b'{"prompt": "x"}\n',
        "data/scratch/big.jsonl": b"y\n",
        "wtests/test_math.py": b"X = 1\n",
        ".git/HEAD": b"ref: refs/heads/main\n",
        "__pycache__/a.cpython-312.pyc": b"\0",
        "model.safetensors": b"\0" * 16,
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    found = Project.find(root / "data")  # found from a subdirectory too
    assert found is not None
    return found


def test_files_skip_version_control_caches_and_weights(tmp_path: Path) -> None:
    p = make_project(tmp_path)
    assert p.name == "demo"
    assert p.root == tmp_path.resolve()
    assert [str(f) for f in p.files()] == [
        "data/loops.jsonl",
        "data/scratch/big.jsonl",
        "wspx.toml",
        "wtests/test_math.py",
    ]


def test_project_excludes_add_to_the_defaults(tmp_path: Path) -> None:
    p = make_project(tmp_path, exclude=["data/scratch"])
    assert "data/scratch/big.jsonl" not in [str(f) for f in p.files()]


def test_pack_is_reproducible_and_round_trips(tmp_path: Path) -> None:
    p = make_project(tmp_path)
    first = p.pack()
    os.utime(tmp_path / "data" / "loops.jsonl", (1, 1))
    assert p.pack() == first  # timestamps and owners never change the bytes
    out = project.extract(first, tmp_path / "out")
    assert (out / "data" / "loops.jsonl").read_bytes() == b'{"prompt": "x"}\n'
    assert not (out / ".git").exists()


def test_oversized_projects_name_their_largest_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = make_project(tmp_path)
    monkeypatch.setattr(project, "MAX_BYTES", 8)
    with pytest.raises(ConfigError, match=r"data/loops\.jsonl.*\[project\] exclude"):
        p.pack()


def _archive(*members: tarfile.TarInfo) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for info in members:
            tar.addfile(info, io.BytesIO(b"x" * info.size) if info.isfile() else None)
    return buf.getvalue()


def _file(name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = 1
    return info


def _link(name: str, target: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info


@pytest.mark.parametrize(
    "data",
    [
        _archive(_file("../escape.txt")),
        _archive(_file("/etc/passwd")),
        _archive(_link("data", "/etc")),
        b"not an archive",
    ],
    ids=["parent", "absolute", "symlink", "garbage"],
)
def test_unsafe_snapshots_are_refused(tmp_path: Path, data: bytes) -> None:
    with pytest.raises(WspxError):
        project.extract(data, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_no_project_outside_one(tmp_path: Path) -> None:
    assert Project.find(tmp_path) is None
