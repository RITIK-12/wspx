from pathlib import Path

import pytest

from wspx.errors import RefConflict, VCSError
from wspx.store import Store
from wspx.vcs import PatchRecord, Repo


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    r = Repo(Store(tmp_path), base="fp:base")
    r.init_session("s1", "main")
    return r


def commit(repo: Repo, msg: str, tree: dict, session: str = "s1", **kw: object) -> str:  # type: ignore[type-arg]
    return repo.commit(session, tree=tree, patches=[], result_hashes={}, message=msg, **kw).oid  # type: ignore[arg-type]


def test_unborn_then_commits(repo: Repo) -> None:
    assert repo.head("s1") is None
    assert repo.current_branch("s1") == "main"
    a = commit(repo, "first", {"merged": {"w": ["sha256:" + "a" * 64]}})
    b = commit(repo, "second", {"merged": {"w": ["sha256:" + "a" * 64, "sha256:" + "b" * 64]}})
    assert repo.head("s1") == b
    assert repo.branches() == {"main": b}
    assert [c.message for c in repo.log(b)] == ["second", "first"]
    assert repo.resolve("HEAD~1", "s1") == a
    assert repo.resolve("HEAD^", "s1") == a
    assert repo.resolve("HEAD~2", "s1") is None
    assert repo.resolve(b[7:14]) == b
    assert repo.resolve("main") == b
    with pytest.raises(VCSError):
        repo.resolve("HEAD~3", "s1")


def test_commit_is_content_addressed_and_cas_protected(repo: Repo) -> None:
    oid = commit(repo, "x", {})
    data = repo.store.objects.get_json(oid)
    assert data["type"] == "commit"
    assert data["base"] == "fp:base"
    stale = repo.head("s1")
    commit(repo, "y", {})  # another writer advances the branch…
    repo.head = lambda _session: stale  # type: ignore[method-assign]  # …after this one read HEAD
    with pytest.raises(RefConflict):
        commit(repo, "z", {})


def test_branches_merge_base_and_introduced(repo: Repo) -> None:
    blob = lambda c: "sha256:" + c * 64  # noqa: E731
    base = commit(repo, "base", {"merged": {"w": [blob("a")]}})
    repo.create_branch("feature", base)
    repo.init_session("s2", "feature")
    ours = commit(repo, "ours", {"merged": {"w": [blob("a"), blob("b")]}})
    theirs = commit(
        repo, "theirs", {"merged": {"w": [blob("a"), blob("c")], "v": [blob("d")]}}, session="s2"
    )
    assert repo.merge_base(ours, theirs) == base
    assert repo.introduced(base, theirs) == {"w": [blob("c")], "v": [blob("d")]}
    assert repo.is_ancestor(base, ours)
    assert not repo.is_ancestor(ours, theirs)


def test_revert_removes_exactly_what_a_commit_added(repo: Repo) -> None:
    blob = lambda c: "sha256:" + c * 64  # noqa: E731
    commit(repo, "a", {"merged": {"w": [blob("a")]}, "promote": []})
    mid = commit(
        repo, "b", {"merged": {"w": [blob("a"), blob("b")], "v": [blob("c")]}, "promote": ["v"]}
    )
    commit(
        repo,
        "c",
        {"merged": {"w": [blob("a"), blob("b"), blob("e")], "v": [blob("c")]}, "promote": ["v"]},
    )
    tree = repo.revert_tree("s1", mid)
    assert tree["merged"] == {"w": [blob("a"), blob("e")]}
    assert tree["promote"] == []


def test_patch_records(repo: Repo) -> None:
    rec = repo.put_patch(
        PatchRecord(
            "ablate",
            {"targets": "L0.mlp.down"},
            "merged",
            [{"tensor": "w", "kind": "set", "blob": "sha256:" + "a" * 64}],
        )
    )
    assert rec.handle.startswith("p_")
    again = repo.get_patch(rec.handle)
    assert again.op == "ablate"
    assert again.tensors == ["w"]


def test_branch_names(repo: Repo) -> None:
    for bad in ["p_x", "HEAD", "a..b", ""]:
        with pytest.raises(VCSError):
            repo.create_branch(bad, None)
    repo.create_branch("exp/one", None)
    with pytest.raises(VCSError, match="exists"):
        repo.create_branch("exp/one", None)
