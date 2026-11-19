import threading
from pathlib import Path

import pytest

from wspx.errors import IntegrityError, RefConflict, StoreError
from wspx.store import Store, canonical_json, oid_of, short


class TestObjectStore:
    def test_put_get(self, tmp_path: Path) -> None:
        store = Store(tmp_path)
        oid = store.objects.put(b"hello")
        assert oid == oid_of(b"hello")
        assert store.objects.get(oid) == b"hello"
        assert store.objects.put(b"hello") == oid
        assert list(store.objects) == [oid]

    def test_json_is_canonical(self, tmp_path: Path) -> None:
        store = Store(tmp_path)
        a = store.objects.put_json({"b": 1, "a": [1, 2]})
        b = store.objects.put_json({"a": [1, 2], "b": 1})
        assert a == b
        assert store.objects.get_json(a) == {"a": [1, 2], "b": 1}
        with pytest.raises(ValueError):  # noqa: PT011
            canonical_json({"x": float("nan")})

    def test_corruption_detected(self, tmp_path: Path) -> None:
        store = Store(tmp_path)
        oid = store.objects.put(b"payload")
        path = store.objects._path(oid)
        path.chmod(0o644)
        path.write_bytes(b"tampered")
        with pytest.raises(IntegrityError):
            store.objects.get(oid)

    def test_prefix_resolution(self, tmp_path: Path) -> None:
        store = Store(tmp_path)
        oid = store.objects.put(b"x")
        assert store.objects.resolve_prefix(short(oid)) == oid
        assert store.objects.resolve_prefix(oid) == oid
        with pytest.raises(StoreError):
            store.objects.resolve_prefix("zz")

    def test_path_escape(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="escapes"):
            Store(tmp_path).path("..", "etc")


class TestRefs:
    def test_update_and_cas(self, tmp_path: Path) -> None:
        refs = Store(tmp_path).refs
        assert refs.read("heads/main") is None
        refs.update("heads/main", "sha256:" + "a" * 64, expected=None)
        with pytest.raises(RefConflict):
            refs.update("heads/main", "sha256:" + "b" * 64, expected=None)
        refs.update("heads/main", "sha256:" + "b" * 64, expected="sha256:" + "a" * 64)
        assert refs.read("heads/main") == "sha256:" + "b" * 64
        assert refs.list() == {"heads/main": "sha256:" + "b" * 64}

    def test_symbolic(self, tmp_path: Path) -> None:
        refs = Store(tmp_path).refs
        refs.set_symbolic("sessions/s_1/HEAD", "heads/s_1")
        assert refs.read("sessions/s_1/HEAD") is None
        refs.update("sessions/s_1/HEAD", "sha256:" + "c" * 64)
        assert refs.read("heads/s_1") == "sha256:" + "c" * 64
        assert refs.symbolic_target("sessions/s_1/HEAD") == "heads/s_1"

    def test_concurrent_updates_serialise(self, tmp_path: Path) -> None:
        refs = Store(tmp_path).refs
        refs.update("heads/n", "0")
        errors: list[Exception] = []

        def bump() -> None:
            for _ in range(50):
                while True:
                    current = refs.read("heads/n")
                    try:
                        refs.update("heads/n", str(int(current or 0) + 1), expected=current)
                        break
                    except RefConflict:
                        continue
                    except Exception as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)
                        return

        threads = [threading.Thread(target=bump) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert refs.read("heads/n") == "200"
