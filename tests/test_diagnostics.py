import pytest

from wspx.addressing import HandleKind
from wspx.diagnostics import Code, Fix, Report, Severity, diag
from wspx.errors import HandleError
from wspx.handles import HandleTable


class TestDiagnostics:
    def test_render_matches_compiler_style(self) -> None:
        d = diag(
            Code.SUB_ULP,
            "38% of delta magnitude below bf16 resolution",
            target="L19.mlp.down",
            detail=["median |w| 0.021 → ULP ≈ 1.2e-4"],
            fixes=[Fix(title="scale the edit ≥ 2×", action={"scale_at_least": 2})],
        )
        assert d.severity is Severity.WARNING
        assert d.render().splitlines() == [
            "W203 L19.mlp.down: 38% of delta magnitude below bf16 resolution",
            "     median |w| 0.021 → ULP ≈ 1.2e-4",
            "  fix 1  scale the edit ≥ 2×",
        ]

    def test_report(self) -> None:
        report = Report(levels=["static"])
        assert report.ok
        report.diagnostics.append(diag(Code.NO_OP, "edit changes nothing"))
        assert not report.ok
        assert report.errors[0].code == "E105"
        assert "blocked" in report.summary()


class TestHandleTable:
    def test_register_and_resolve(self) -> None:
        table = HandleTable()
        e1 = table.register_result(("writes-wait", "mlp-down", "top5"), description="5 cols")
        e2 = table.register_result(("x",), name="loop_writers")
        assert (e1.key, e2.key) == ("@r1", "@r2")
        entry, warnings = table.resolve("@r1:writes-wait.mlp-down.top5")
        assert entry is e1
        assert not warnings
        _, warnings = table.resolve("@r2:loop_writers")
        assert not warnings
        _, warnings = table.resolve("@r1:something-else")
        assert warnings
        assert "@r1" in warnings[0]

    def test_missing(self) -> None:
        with pytest.raises(HandleError, match="does not exist"):
            HandleTable().resolve("@r9")

    def test_roundtrip_keeps_counter(self) -> None:
        table = HandleTable()
        table.register_result(("a",))
        table.register(HandleKind.DIRECTION, "dir:loop", "contrast.n200")
        restored = HandleTable.load(table.dump())
        assert restored.register_result(("b",)).key == "@r2"
        assert "dir:loop" in restored.table()
