from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
import torch

transformers = pytest.importorskip("transformers")

import wspx  # noqa: E402
from wspx.errors import CheckFailed, VCSError  # noqa: E402
from wspx.lab.sdk import activate, open_lab  # noqa: E402
from wspx.lab.session import Lab  # noqa: E402
from wspx.lab.tensors import tensor_hash  # noqa: E402


def tiny_llama(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(
        vocab_size=96,
        hidden_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=64,
        num_key_value_heads=2,
        head_dim=12,
    )
    return transformers.AutoModelForCausalLM.from_config(cfg).eval()


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    (tmp_path / "work").mkdir()
    return open_lab(tiny_llama(), store=tmp_path / "store", workdir=tmp_path / "work", branch="exp")


def hashes(lab: Lab) -> dict[str, str]:
    return {n: tensor_hash(p) for n, p in lab.model.named_parameters()}


POS = [[1, 9, 2, i] for i in range(10, 22)]
NEG = [[1, 9, 2, i] for i in range(60, 72)]


class TestWorkflow:
    def test_grep_contrast_edit_check_commit(self, lab: Lab) -> None:
        info = lab.arch().summary
        assert "LlamaForCausalLM" in info
        assert "branch exp" in info
        listed = lab.arch("L0").summary
        assert "L0.mlp.down_proj" in listed
        assert "L0.mlp.down " not in listed  # each tensor once, by its key

        hits = lab.grep("colnorm", scope="L*.mlp.down", top=5, name="big")
        assert hits.handle.startswith("@r1:")
        assert len(hits) == 5
        assert "saved as @r1" in hits.summary

        d = lab.contrast(POS, NEG, name="loop", layer=1)
        assert d.handle.startswith("dir:loop(contrast.L1-resid.n24")
        assert "top writers" in d.summary

        p = lab.edit.project_out("dir:loop", ["L1.mlp.down", "L2.mlp.down"], cols=hits)
        assert p.handle.startswith("p_")
        # only the hit columns of those tensors changed
        w = lab.model.model.layers[1].mlp.down_proj.weight.detach().float()
        v = lab.direction("dir:loop")
        cols = [h.unit for h in hits if h.param == "model.layers.1.mlp.down_proj.weight"]
        if cols:
            assert torch.allclose((v / v.norm()) @ w[:, cols], torch.zeros(len(cols)), atol=5e-3)

        report = lab.check(p)
        assert report.levels == ["static", "numeric", "smoke"]
        assert "canary_kl" in report.stats

        out = lab.commit("Project out loop direction in L1/L2 down_proj")
        assert out.summary.startswith("committed ")
        assert "Project out loop" in lab.log().summary
        assert lab.status().data["staged"] == []

    def test_noop_edit_is_refused(self, lab: Lab) -> None:
        with pytest.raises(CheckFailed) as err:
            lab.edit.scale("L0.mlp.down", factor=1.0)
        assert "E105" in str(err.value)

    def test_checkout_and_revert_are_exact(self, lab: Lab) -> None:
        base = hashes(lab)
        lab.edit.ablate("L0.mlp.down[:, 3]")
        lab.commit("Ablate unit 3")
        edited = hashes(lab)
        assert edited != base
        lab.checkout("base", force=True)
        assert hashes(lab) == base
        lab.checkout("exp")
        assert hashes(lab) == edited
        lab.revert("HEAD")
        assert hashes(lab) == base  # S4: revert restores the base hash

    def test_history_order_does_not_change_bits(self, tmp_path: Path) -> None:
        """The same deltas staged in either order give bit-identical weights (S4)."""
        lab = open_lab(tiny_llama(), store=tmp_path / "ref", workdir=tmp_path)
        a = lab.edit.scale("L1.mlp.down[:, 0:8]", factor=1.25)
        b = lab.edit.ablate("L1.mlp.down[5]")
        forward = hashes(lab)
        lab.tree.checkout(lab.tree.state.__class__())
        lab.tree.stage([lab._load_delta(d["blob"]) for d in b.record.deltas])
        lab.tree.stage([lab._load_delta(d["blob"]) for d in a.record.deltas])
        assert hashes(lab) == forward

    def test_protected_branch_and_staged_guard(self, lab: Lab) -> None:
        lab.edit.ablate("L0.mlp.down[:, 1]")
        with pytest.raises(VCSError, match="staged"):
            lab.checkout("base")
        lab.context["caller"] = "driver"
        lab.config = lab.config.model_copy(
            update={"policy": lab.config.policy.model_copy(update={"protected_branches": ["exp"]})}
        )
        with pytest.raises(VCSError, match="protected"):
            lab.commit("not allowed")


class TestMerge:
    def test_orthogonal_edits_sum_and_conflicts_need_a_strategy(self, lab: Lab) -> None:
        lab.edit.ablate("L0.mlp.down[:, 2]")
        lab.commit("base edit")
        lab.branch("other", switch=False)
        e = torch.eye(48)
        lab.edit.project_out(e[0], "L2.mlp.down")
        lab.commit("ours: L2")
        lab.checkout("other")
        lab.edit.project_out(e[1], "L1.mlp.down")
        lab.commit("theirs: L1")
        lab.checkout("exp")
        out = lab.merge("other")
        assert "merged other" in out.summary
        assert out.data["merge"]["conflicts"] == []

        lab.branch("clash", switch=False)
        lab.edit.project_out(e[3], "L0.attn.o")
        lab.commit("ours: o along e3")
        lab.checkout("clash")
        lab.edit.project_out(e[3], "L0.attn.o", strength=0.5)
        lab.commit("theirs: o along e3 too")
        lab.checkout("exp")
        conflict = lab.merge("clash")
        assert not conflict.ok
        assert "needs a strategy" in conflict.summary
        resolved = lab.merge("clash", strategies={"L0.attn.o": "orthogonalize"})
        assert "merged clash" in resolved.summary
        assert resolved.data["merge"]["conflicts"][0]["strategy"] == "orthogonalize"


class TestSuites:
    def test_user_suites_paired_cached_and_heldout(self, lab: Lab) -> None:
        tests = lab.workdir / "wtests"
        tests.mkdir()
        (tests / "suites.py").write_text(
            textwrap.dedent("""
            from wspx.testing import suite, expect

            @suite(split="dev", seed=0)
            def logits_ok(model):
                for i in range(6):
                    yield expect.logit_diff(model.logit_diff([1, 2, i + 3], 5, 6), at_least=-100.0)

            @suite(split="heldout", seed=1)
            def secret(model):
                for i in range(4):
                    yield expect.true(model.kl_vs_base([1, i + 2]) < 1.0, "close to base")
        """)
        )
        (tests / "simple.yaml").write_text(
            "name: echo\nitems:\n  - prompt: [1, 2, 3]\n    expect: {contains: '#'}\n"
        )
        first = lab.test("logits_ok")
        assert "6/6 passed" in first.summary
        again = lab.test("logits_ok")
        assert "cached" in again.summary
        assert "1/1 passed" in lab.test("echo").summary
        lab.edit.ablate("L0.mlp.down[:, 0:4]")
        paired = lab.test("logits_ok", vs="base")
        assert "vs " in paired.summary
        assert "CI" in paired.summary
        held = lab.test("secret")
        assert "held-out" in held.summary
        assert "checks" not in held.data["runs"][0]
        assert "failures" not in held.data["runs"][0]
        assert lab.heldout_queries == 1

    def test_builtin_canary(self, lab: Lab) -> None:
        assert "canary [dev] 32/32" in lab.test("canary").summary

    def test_the_eval_cache_is_per_model(self, lab: Lab, tmp_path: Path) -> None:
        """Seen live: every model's unedited base shared one cache key, so a test on a new
        model returned another model's cached run."""
        assert "cached" not in lab.test("canary").summary
        (tmp_path / "other").mkdir()
        other = open_lab(
            tiny_llama(seed=1), store=tmp_path / "store", workdir=tmp_path / "other", branch="o"
        )
        assert other.ir.fingerprint != lab.ir.fingerprint
        assert "cached" not in other.test("canary").summary
        assert "cached" in lab.test("canary").summary

    def test_diff_shows_current_suites_and_never_held_out_ones(self, lab: Lab) -> None:
        """Seen live: vcs diff listed another project's held-out suite to the driver, and three
        versions of one suite with different scores."""
        tests = lab.workdir / "wtests"
        tests.mkdir()
        source = textwrap.dedent("""
            from wspx.testing import suite, expect

            @suite(split="dev", seed=0)
            def near(model):
                for i in range(4):
                    yield expect.kl(model.kl_vs_base([1, i + 2]), at_most={limit})

            @suite(split="heldout", seed=1)
            def secret(model):
                for i in range(4):
                    yield expect.true(model.kl_vs_base([1, i + 2]) < 1.0, "close to base")
        """)
        (tests / "suites.py").write_text(source.format(limit="1.0"))
        lab.edit.scale("L0.mlp.down[:, 0:4]", factor=0.5)
        lab.test("near", vs="base")
        lab.test("secret", vs="base")
        asked = lab.heldout_queries
        lines = lab.diff("base").data["behaviour"]
        assert any(line.startswith("  behaviour near:") for line in lines)
        assert not any("secret" in line for line in lines)
        assert "1 held-out or blind suite(s) not shown" in lines[-1]
        assert lab.heldout_queries == asked
        (tests / "suites.py").write_text(source.format(limit="2.0"))  # a new version of `near`
        lines = lab.diff("base").data["behaviour"]
        assert not any("near" in line for line in lines)

    def test_bisect_finds_the_breaking_commit(self, lab: Lab) -> None:
        (lab.workdir / "wtests").mkdir()
        (lab.workdir / "wtests" / "near.py").write_text(
            textwrap.dedent("""
            from wspx.testing import suite, expect

            @suite
            def near_base(model):
                yield expect.kl(model.kl_vs_base([1, 2, 3, 4]), at_most=1e-3)
        """)
        )
        lab.edit.scale("L0.mlp.down[:, 0]", factor=1.001)
        lab.commit("tiny")
        lab.edit.scale("unembed", factor=50.0)  # sharpens every next-token distribution
        lab.commit("break everything")
        lab.edit.ablate("L2.mlp.down[:, 7]")
        lab.commit("after")
        out = lab.bisect("near_base")
        assert "break everything" in out.summary


class TestTools:
    def test_traces_show_generate(self, lab: Lab) -> None:
        fwd = lab.trace([1, 2, 3, 4])
        assert fwd.handles[0].startswith("t_")
        lab.edit.scale("L1.mlp.down", factor=0.0 + 1e-9)
        diff = lab.trace([1, 2, 3, 4], vs="base")
        assert "first divergence: L1" in diff.summary
        attr = lab.trace([1, 2, 3, 4], reference=[1, 2, 3, 5])
        assert "attribution" in attr.summary
        assert "norm" in lab.show("L0.mlp.down[:, 1]", values=5).summary
        assert lab.generate([1, 2, 3], max_tokens=3).data["ids"][0]

    def test_blame_and_diff(self, lab: Lab) -> None:
        lab.edit.ablate("L0.mlp.down[:, 3]")
        lab.commit("ablate col 3")
        lab.edit.ablate("L0.mlp.down[:, 9]")
        lab.commit("ablate col 9")
        blame = lab.blame("L0.mlp.down[:, 3]").summary
        assert "ablate col 3" in blame
        assert "col 9" not in blame
        diff = lab.diff("HEAD~2", "HEAD")
        assert "L0.mlp.down" in diff.summary
        assert diff.data["tensors"][0]["touched"] == 96

    def test_run_script_uses_the_session(self, lab: Lab) -> None:
        (lab.workdir / "edit.py").write_text(
            textwrap.dedent("""
            import wspx
            m = wspx.session()
            p = m.edit.ablate("L2.mlp.down[:, 0]")
            print("staged", p.handle)
        """)
        )
        out = lab.run("edit.py")
        assert out.ok, out.summary
        assert "staged p_" in out.summary
        assert lab.staged[-1].script == out.data["script"]

    def test_export(self, lab: Lab, tmp_path: Path) -> None:
        lab.edit.project_out(torch.ones(48), "L1.mlp.down")
        lab.commit("edit")
        merged = lab.export("merged", "out/merged")
        assert "exported merged" in merged.summary
        peft = lab.export("peft", "out/adapter")
        assert "exported peft" in peft.summary
        bundle = lab.export("bundle", "out/b.tar.gz")
        assert bundle.data["objects"] >= 3

    def test_resume_restores_everything(self, tmp_path: Path) -> None:
        lab = open_lab(tiny_llama(), store=tmp_path / "store", workdir=tmp_path)
        lab.grep("colnorm", scope="L*.mlp.down", top=3)
        lab.contrast(POS, NEG, name="loop", layer=1)
        lab.edit.ablate("L0.mlp.down[:, 4]")
        lab.commit("one")
        lab.edit.project_out("dir:loop", "L2.mlp.down")
        before = hashes(lab)
        # a fresh process: same store, same session, base weights
        again = open_lab(tiny_llama(), store=tmp_path / "store", workdir=tmp_path)
        assert hashes(again) == before
        assert [p.handle for p in again.staged] == [p.handle for p in lab.staged]
        assert torch.equal(again.direction("dir:loop").cpu(), lab.direction("dir:loop").cpu())
        assert again.handles.get("@r1") is not None


def test_wspx_session_entry_point(tmp_path: Path) -> None:
    lab = open_lab(tiny_llama(), store=tmp_path / "s", workdir=tmp_path)
    previous = activate(lab)
    try:
        assert wspx.session() is lab
    finally:
        activate(previous)


def test_grep_holds_one_tensor_at_a_time(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seen live: wide greps ran a 24 GB card out of memory holding every fp32 block."""
    import weakref

    from wspx.lab import grep as grep_mod

    alive: list[weakref.ref[torch.Tensor]] = []
    most = 0
    units = grep_mod._units

    def counted(*args: Any, **kw: Any) -> Any:
        nonlocal most
        for batch in units(*args, **kw):
            alive.append(weakref.ref(batch.vectors))
            most = max(most, sum(r() is not None for r in alive))
            yield batch

    monkeypatch.setattr(grep_mod, "_units", counted)
    hits = lab.grep("colnorm > 1σ", scope="L*.mlp.down,L*.attn.o", top=3)
    assert len(alive) >= 2 * 2 * lab.ir.num_blocks  # every tensor, once per pass
    assert most <= 2  # the one being scored, and the last one until it is released
    assert len(hits) == 3


def test_contrast_keeps_its_direction_when_ranking_writers_fails(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    grep = Lab.grep

    def failing(self: Lab, expression: str | None = None, **kw: Any) -> Any:
        if expression and expression.startswith("abs(cos(dir:"):
            raise RuntimeError("CUDA out of memory")
        return grep(self, expression, **kw)

    monkeypatch.setattr(Lab, "grep", failing)
    d = lab.contrast(
        [[1, i, 2, 10] for i in range(20, 24)], [[1, i, 2, 60] for i in range(20, 24)], name="kept"
    )
    assert "dir:kept" in d.summary
    assert "not ranked (RuntimeError: CUDA out of memory)" in d.summary
    assert "kept" in lab.directions
