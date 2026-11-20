"""The repository: commits, trees and patch records over a :class:`Store`."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from wspx.addressing import is_valid_ref_name
from wspx.errors import StoreError, VCSError
from wspx.store import Store, is_oid, short

HEADS = "heads/"
_REV_RE = re.compile(r"^(?P<base>[^~^]+)(?P<ops>(?:~\d*|\^\d*)*)$")


def short_id(oid: str) -> str:
    return short(oid, 7)


@dataclass
class PatchRecord:
    """A staged edit. Its id (``p_<8 hex>``) is derived from its content hash."""

    op: str
    args: dict[str, Any]
    precision: str
    deltas: list[dict[str, Any]]
    """``[{"tensor", "kind", "blob", "rank"?}]``."""
    created: float = field(default_factory=time.time)
    call: str | None = None
    script: str | None = None
    """Hash of the script that produced it, when it came from ``run``."""
    oid: str = ""

    @property
    def handle(self) -> str:
        return "p_" + short(self.oid, 8)

    @property
    def tensors(self) -> list[str]:
        return sorted({d["tensor"] for d in self.deltas})

    def to_dict(self) -> dict[str, Any]:
        out = {
            "type": "patch",
            "op": {"name": self.op, "args": self.args},
            "precision": self.precision,
            "deltas": self.deltas,
            "created": self.created,
        }
        if self.call:
            out["call"] = self.call
        if self.script:
            out["script"] = self.script
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any], oid: str = "") -> PatchRecord:
        return cls(
            op=data["op"]["name"],
            args=data["op"].get("args", {}),
            precision=data.get("precision", "merged"),
            deltas=list(data["deltas"]),
            created=data.get("created", 0.0),
            call=data.get("call"),
            script=data.get("script"),
            oid=oid,
        )


@dataclass
class Commit:
    parents: list[str]
    base: str
    tree: str
    patches: list[str]
    """Patch-record ids introduced by this commit."""
    result_hashes: dict[str, str]
    message: str
    author: dict[str, Any] = field(default_factory=dict)
    checks: str | None = None
    evals: list[str] = field(default_factory=list)
    time: float = field(default_factory=time.time)
    forced: bool = False
    merge: dict[str, Any] | None = None
    status: str = "ok"
    """``ok`` or ``needs-review`` (a merge whose interference check failed)."""
    oid: str = ""

    @property
    def short(self) -> str:
        return short_id(self.oid)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": "commit",
            "parents": self.parents,
            "base": self.base,
            "tree": self.tree,
            "patches": self.patches,
            "result_hashes": self.result_hashes,
            "message": self.message,
            "author": self.author,
            "evals": self.evals,
            "time": self.time,
            "status": self.status,
        }
        if self.checks:
            out["checks"] = self.checks
        if self.forced:
            out["forced"] = True
        if self.merge is not None:
            out["merge"] = self.merge
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any], oid: str) -> Commit:
        if data.get("type") != "commit":
            raise VCSError(f"{short_id(oid)} is a {data.get('type')}, not a commit")
        return cls(
            parents=list(data["parents"]),
            base=data["base"],
            tree=data["tree"],
            patches=list(data.get("patches", [])),
            result_hashes=dict(data.get("result_hashes", {})),
            message=data.get("message", ""),
            author=data.get("author", {}),
            checks=data.get("checks"),
            evals=list(data.get("evals", [])),
            time=data.get("time", 0.0),
            forced=data.get("forced", False),
            merge=data.get("merge"),
            status=data.get("status", "ok"),
            oid=oid,
        )


EMPTY_TREE: dict[str, Any] = {"type": "tree", "merged": {}, "adapter": {}, "promote": []}


def normalize_tree(tree: dict[str, Any] | None) -> dict[str, Any]:
    """The canonical form of a tree: sorted, empty entries dropped, no type tag."""
    tree = tree or {}
    return {
        "merged": {k: sorted(v) for k, v in sorted((tree.get("merged") or {}).items()) if v},
        "adapter": {k: sorted(v) for k, v in sorted((tree.get("adapter") or {}).items()) if v},
        "promote": sorted(tree.get("promote") or []),
    }


def tree_key(tree: dict[str, Any] | None) -> str:
    """Short hash of the weights a tree describes: the eval-cache key (same weights, same key)."""
    import hashlib

    from wspx.store import canonical_json

    return hashlib.sha256(canonical_json(normalize_tree(tree))).hexdigest()[:16]


class Repo:
    def __init__(self, store: Store, base: str) -> None:
        self.store = store
        self.base = base
        self._commits: dict[str, Commit] = {}

    # ------------------------------------------------------------------ objects

    def put_tree(self, tree: dict[str, Any]) -> str:
        return self.store.objects.put_json({**EMPTY_TREE, **tree, "type": "tree"})

    def get_tree(self, oid: str | None) -> dict[str, Any]:
        if oid is None:
            return dict(EMPTY_TREE)
        data = self.store.objects.get_json(oid)
        if data.get("type") != "tree":
            raise VCSError(f"{short_id(oid)} is not a tree")
        return dict(data)

    def put_patch(self, record: PatchRecord) -> PatchRecord:
        record.oid = self.store.objects.put_json(record.to_dict())
        return record

    def get_patch(self, ref: str) -> PatchRecord:
        oid = self._expand(ref.removeprefix("p_"))
        return PatchRecord.from_dict(self.store.objects.get_json(oid), oid)

    def get_commit(self, oid: str) -> Commit:
        if oid not in self._commits:
            self._commits[oid] = Commit.from_dict(self.store.objects.get_json(oid), oid)
        return self._commits[oid]

    def tree_of(self, rev: str | None) -> dict[str, Any]:
        if rev is None:
            return dict(EMPTY_TREE)
        oid = self.resolve(rev)
        return self.get_tree(self.get_commit(oid).tree) if oid else dict(EMPTY_TREE)

    # ------------------------------------------------------------------ refs

    def head_ref(self, session: str) -> str:
        return f"sessions/{session}/HEAD"

    def init_session(
        self, session: str, branch: str | None = None, *, start: str | None = None
    ) -> str:
        """Point the session's HEAD at ``branch`` (created at ``start`` if missing)."""
        branch = branch or session
        self._check_branch(branch)
        ref = HEADS + branch
        if start is not None and self.store.refs.read(ref) is None:
            self.store.refs.update(ref, self.resolve(start) or "", expected=None)
        self.store.refs.set_symbolic(self.head_ref(session), ref)
        return branch

    def current_branch(self, session: str) -> str | None:
        target = self.store.refs.symbolic_target(self.head_ref(session))
        return target.removeprefix(HEADS) if target and target.startswith(HEADS) else None

    def head(self, session: str) -> str | None:
        value = self.store.refs.read(self.head_ref(session))
        return value or None

    def branches(self) -> dict[str, str]:
        """Branches on this model. Refs are shared by every model in the store, so others' are
        left out (a commit records the model it was made on)."""
        out = {}
        for ref, oid in self.store.refs.list(HEADS).items():
            if not oid:
                continue
            try:
                if self.base and self.get_commit(oid).base != self.base:
                    continue
            except StoreError:
                continue
            out[ref.removeprefix(HEADS)] = oid
        return out

    def create_branch(self, name: str, at: str | None) -> str:
        self._check_branch(name)
        oid = self.resolve(at) if at else None
        if self.store.refs.exists(HEADS + name):
            raise VCSError(
                f"branch {name!r} already exists (branch names are shared by every session in "
                "this deployment, whatever the model); pick another name"
            )
        self.store.refs.update(HEADS + name, oid or "", expected=None)
        return oid or ""

    def delete_branch(self, name: str) -> None:
        self.store.refs.delete(HEADS + name)

    def switch(self, session: str, branch: str) -> str | None:
        if not self.store.refs.exists(HEADS + branch):
            raise VCSError(f"no branch {branch!r}")
        self.store.refs.set_symbolic(self.head_ref(session), HEADS + branch)
        return self.head(session)

    def fast_forward(self, session: str, oid: str) -> None:
        """Move the session's branch (or detached HEAD) on to ``oid``, a descendant of HEAD."""
        self.store.refs.update(self.head_ref(session), oid, expected=self.head(session))

    def detach(self, session: str, oid: str) -> None:
        self.store.refs.delete(self.head_ref(session))
        self.store.refs.update(self.head_ref(session), oid)

    @staticmethod
    def _check_branch(name: str) -> None:
        if not is_valid_ref_name(name) or name.startswith(("p_", "t_", "HEAD")):
            raise VCSError(f"invalid branch name {name!r}")

    # ------------------------------------------------------------------ commit

    def commit(
        self,
        session: str,
        *,
        tree: dict[str, Any],
        patches: list[str],
        result_hashes: dict[str, str],
        message: str,
        author: dict[str, Any] | None = None,
        checks: str | None = None,
        evals: list[str] | None = None,
        forced: bool = False,
        extra_parents: list[str] | None = None,
        merge: dict[str, Any] | None = None,
        status: str = "ok",
    ) -> Commit:
        if not message.strip():
            raise VCSError("commit message is empty")
        parent = self.head(session)
        parents = ([parent] if parent else []) + list(extra_parents or [])
        commit = Commit(
            parents=parents,
            base=self.base,
            tree=self.put_tree(tree),
            patches=patches,
            result_hashes=result_hashes,
            message=message.strip(),
            author=author or {},
            checks=checks,
            evals=evals or [],
            forced=forced,
            merge=merge,
            status=status,
        )
        commit.oid = self.store.objects.put_json(commit.to_dict())
        self.store.refs.update(self.head_ref(session), commit.oid, expected=parent)
        self._commits[commit.oid] = commit
        return commit

    # ------------------------------------------------------------------ revisions

    def resolve(self, rev: str, session: str | None = None) -> str | None:
        """``HEAD``, ``HEAD~2``, ``HEAD^2``, a branch, a (short) commit id or ``base``."""
        rev = rev.strip()
        if rev in ("base", "BASE"):
            return None
        m = _REV_RE.match(rev)
        if not m:
            raise VCSError(f"bad revision {rev!r}")
        oid = self._resolve_base(m.group("base"), session)
        for op in re.findall(r"[~^]\d*", m.group("ops")):
            n = int(op[1:]) if len(op) > 1 else 1
            if oid is None:
                raise VCSError(f"{rev}: walks past the base")
            if op[0] == "~":
                for _ in range(n):
                    if oid is None:
                        raise VCSError(f"{rev}: walks past the base")
                    parents = self.get_commit(oid).parents
                    oid = parents[0] if parents else None  # the root's parent is the base
            else:
                parents = self.get_commit(oid).parents
                if n == 0:
                    continue
                if n > len(parents):
                    raise VCSError(f"{rev}: commit has {len(parents)} parent(s)")
                oid = parents[n - 1]
        return oid

    def _resolve_base(self, name: str, session: str | None) -> str | None:
        if name == "HEAD":
            if session is None:
                raise VCSError("HEAD needs a session")
            return self.head(session)
        if is_valid_ref_name(name) and self.store.refs.exists(HEADS + name):
            return self.store.refs.read(HEADS + name)
        if is_oid(name):
            return name
        try:
            return self._expand(name)
        except StoreError:
            raise VCSError(f"unknown revision {name!r}") from None

    def _expand(self, prefix: str) -> str:
        return self.store.objects.resolve_prefix(prefix)

    # ------------------------------------------------------------------ history

    def walk(self, start: str | None, *, first_parent: bool = False) -> Iterator[Commit]:
        """Commits reachable from ``start``, newest first."""
        seen: set[str] = set()
        frontier = [start] if start else []
        while frontier:
            frontier.sort(key=lambda o: -self.get_commit(o).time)
            oid = frontier.pop(0)
            if oid in seen:
                continue
            seen.add(oid)
            commit = self.get_commit(oid)
            yield commit
            parents = commit.parents[:1] if first_parent else commit.parents
            frontier.extend(p for p in parents if p not in seen)

    def log(self, rev: str | None, n: int = 20) -> list[Commit]:
        out = []
        for commit in self.walk(rev):
            out.append(commit)
            if len(out) >= n:
                break
        return out

    def merge_base(self, a: str | None, b: str | None) -> str | None:
        if a is None or b is None:
            return None
        ancestors = {c.oid for c in self.walk(a)}
        for commit in self.walk(b):
            if commit.oid in ancestors:
                return commit.oid
        return None

    def is_ancestor(self, a: str | None, b: str | None) -> bool:
        if a is None:
            return True
        return b is not None and any(c.oid == a for c in self.walk(b))

    def introduced(
        self,
        since: str | None,
        until: str | None,
        sections: tuple[str, ...] = ("merged", "adapter"),
    ) -> dict[str, list[str]]:
        """Delta blobs per tensor in ``until``'s tree that are not in ``since``'s tree."""
        before = self.get_tree(self.get_commit(since).tree) if since else EMPTY_TREE
        after = self.get_tree(self.get_commit(until).tree) if until else EMPTY_TREE
        out: dict[str, list[str]] = {}
        for section in sections:
            for tensor, blobs in after.get(section, {}).items():
                old = set(before.get(section, {}).get(tensor, []))
                new = [b for b in blobs if b not in old]
                if new:
                    out.setdefault(tensor, []).extend(new)
        return out

    def revert_tree(self, session: str, rev: str) -> dict[str, Any]:
        """HEAD's tree without the deltas ``rev`` introduced (exact: sums restart from base)."""
        oid = self.resolve(rev, session)
        if oid is None:
            raise VCSError("nothing to revert at the base")
        commit = self.get_commit(oid)
        removed = self.introduced(commit.parents[0] if commit.parents else None, oid)
        tree = self.tree_of_oid(self.head(session))
        for section in ("merged", "adapter"):
            for tensor, blobs in list(tree.get(section, {}).items()):
                keep = [b for b in blobs if b not in set(removed.get(tensor, []))]
                if keep:
                    tree[section][tensor] = keep
                else:
                    del tree[section][tensor]
        touched_left = set(tree.get("merged", {})) | set(tree.get("adapter", {}))
        tree["promote"] = [t for t in tree.get("promote", []) if t in touched_left]
        return tree

    def tree_of_oid(self, oid: str | None) -> dict[str, Any]:
        if oid is None:
            return dict(EMPTY_TREE)
        return {
            k: (dict(v) if isinstance(v, dict) else v)
            for k, v in self.get_tree(self.get_commit(oid).tree).items()
        }

    def blame(self, start: str | None, tensor: str, touches: Any) -> list[Commit]:
        """Commits whose patches touched ``tensor`` where ``touches(delta_entry)`` says so."""
        out = []
        for commit in self.walk(start):
            for pid in commit.patches:
                record = self.get_patch(pid)
                if any(d["tensor"] == tensor and touches(d) for d in record.deltas):
                    out.append(commit)
                    break
        return out
