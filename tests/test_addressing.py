import pytest

from wspx.addressing import (
    Address,
    HandleKind,
    HandleRef,
    Index,
    LayerSel,
    Named,
    Range,
    ScopeKind,
    Token,
    is_valid_ref_name,
    make_slug,
    parse_address,
    parse_handle,
)
from wspx.errors import AddressError, HandleError


class TestShortAddresses:
    def test_single_layer(self) -> None:
        a = parse_address("L19.mlp.down")
        assert a.kind is ScopeKind.SHORT
        assert a.layers == LayerSel(19, 19)
        assert a.path == ("mlp", "down")
        assert a.slice is None
        assert str(a) == "L19.mlp.down"
        assert not a.has_glob

    def test_range_and_glob(self) -> None:
        a = parse_address("L8-12.attn.o")
        assert a.layers == LayerSel(8, 12)
        assert 8 in a.layers
        assert 12 in a.layers
        assert 13 not in a.layers
        assert a.has_glob
        star = parse_address("L*.mlp.down")
        assert star.layers is not None
        assert star.layers.is_all
        assert 999 in star.layers

    def test_sub_parts(self) -> None:
        assert parse_address("L7.attn.q.h3").path == ("attn", "q", "h3")
        assert parse_address("L12.moe.e5.up").path == ("moe", "e5", "up")
        assert parse_address("L0.lstm.ih.forget").path == ("lstm", "ih", "forget")

    def test_reversed_range_rejected(self) -> None:
        with pytest.raises(AddressError, match="reversed"):
            parse_address("L12-8.mlp.down")

    def test_a_bare_layer_is_the_layer_itself(self) -> None:
        addr = parse_address("L3")
        assert addr.kind is ScopeKind.SHORT
        assert addr.path == ()
        assert parse_address("L3[oc=5]").slice is not None


class TestSlices:
    def test_column(self) -> None:
        a = parse_address("L19.mlp.down[:, 8812]")
        assert a.slice == (Range(None, None, None), Index(8812))
        assert str(a) == "L19.mlp.down[:, 8812]"

    def test_ranges_and_steps(self) -> None:
        a = parse_address("L1.attn.q[0:128, -4:]")
        assert a.slice == (Range(0, 128), Range(-4, None))
        assert parse_address("L1.attn.q[::2]").slice == (Range(None, None, 2),)

    def test_zero_step_rejected(self) -> None:
        with pytest.raises(AddressError, match="zero"):
            parse_address("L1.attn.q[::0]")

    def test_token_selector(self) -> None:
        a = parse_address('embed[" Wait"]')
        assert a.kind is ScopeKind.SPECIAL
        assert a.slice == (Token(" Wait"),)
        assert str(a) == 'embed[" Wait"]'

    def test_token_with_brackets_and_commas(self) -> None:
        assert parse_address('unembed["],["]').slice == (Token("],["),)
        assert parse_address("unembed['it\\'s']").slice == (Token("it's"),)

    def test_named_axis(self) -> None:
        assert parse_address("L3.conv[oc=12]").slice == (Named("oc", Index(12)),)
        assert parse_address("L3.conv[oc=0:4]").slice == (Named("oc", Range(0, 4)),)

    def test_handle_in_slice(self) -> None:
        a = parse_address("L19.mlp.down[:, @r1:top5]")
        item = a.slice[1] if a.slice else None
        assert isinstance(item, HandleRef)
        assert item.handle.key == "@r1"

    @pytest.mark.parametrize(
        "text", ["L1.mlp.down[", "L1.mlp.down[1]x", "L1.mlp.down[1,,2]", "L1.mlp.down[a]", "[1]"]
    )
    def test_malformed(self, text: str) -> None:
        with pytest.raises(AddressError):
            parse_address(text)


class TestRawNames:
    def test_raw_parameter(self) -> None:
        a = parse_address("model.layers.19.mlp.down_proj.weight[:, 11]")
        assert a.kind is ScopeKind.RAW
        assert a.raw == "model.layers.19.mlp.down_proj.weight"
        assert a.slice == (Range(), Index(11))

    def test_raw_glob(self) -> None:
        a = parse_address("model.layers.*.mlp.down_proj.weight")
        assert a.has_glob

    def test_roundtrip(self) -> None:
        for text in ["final_norm", "unembed", "L*.mlp.down[:, 3]", "a.b.c"]:
            assert str(parse_address(text)) == text

    def test_with_slice(self) -> None:
        a = parse_address("L1.mlp.down").with_slice((Index(2),))
        assert isinstance(a, Address)
        assert str(a) == "L1.mlp.down[2]"


class TestHandles:
    def test_result(self) -> None:
        h = parse_handle("@r1:writes-wait.mlp-down.top5")
        assert h.kind is HandleKind.RESULT
        assert h.key == "@r1"
        assert h.label == "writes-wait.mlp-down.top5"
        assert str(h) == "@r1:writes-wait.mlp-down.top5"
        assert parse_handle("@r007").key == "@r7"

    def test_direction(self) -> None:
        h = parse_handle("dir:loop(contrast.L19-resid.n200)")
        assert h.kind is HandleKind.DIRECTION
        assert h.key == "dir:loop"
        assert h.label == "contrast.L19-resid.n200"
        assert parse_handle("dir:loop").label is None

    def test_patch_and_trace(self) -> None:
        assert parse_handle("p_8f2a:proj").kind is HandleKind.PATCH
        assert parse_handle("t_1c0d").key == "t_1c0d"

    def test_revisions(self) -> None:
        for text in ["HEAD", "HEAD~3", "HEAD^", "main", "5be01c7", "feature/loops"]:
            assert parse_handle(text).kind is HandleKind.REVISION

    def test_revision_disallowed(self) -> None:
        with pytest.raises(HandleError):
            parse_handle("main", allow_revision=False)

    @pytest.mark.parametrize("text", ["", "@x", "dir:", "..", "a..b", "x.lock"])
    def test_invalid(self, text: str) -> None:
        with pytest.raises(HandleError):
            parse_handle(text)


def test_ref_names() -> None:
    assert is_valid_ref_name("heads/main")
    assert is_valid_ref_name("sessions/s_81f/HEAD")
    for bad in ["", "/x", "x/", "a..b", ".hidden", "x.lock", "a b", "a@{1}"]:
        assert not is_valid_ref_name(bad)


def test_make_slug() -> None:
    assert make_slug('writes(" Wait")', "L*.mlp.down", "top5") == "writes-wait.l-mlp-down.top5"
    assert make_slug("", None) == "unnamed"
    assert len(make_slug("x" * 200)) <= 60
