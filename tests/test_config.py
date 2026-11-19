from pathlib import Path

import pytest

from wspx.config import WspxConfig, find_config, format_duration, parse_duration
from wspx.errors import ConfigError


class TestDurations:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [("15m", 900), ("90s", 90), ("1h30m", 5400), ("2d", 172800), ("250ms", 0.25), (42, 42)],
    )
    def test_parse(self, text: str | int, seconds: float) -> None:
        assert parse_duration(text) == seconds

    @pytest.mark.parametrize("text", ["", "15", "15x", "m15", "1h 30q", -1, True])
    def test_invalid(self, text: object) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            parse_duration(text)  # type: ignore[arg-type]

    def test_format(self) -> None:
        assert format_duration(900) == "15m"
        assert format_duration(5400) == "1h30m"
        assert format_duration(0.25) == "250ms"


class TestConfig:
    def test_defaults(self) -> None:
        cfg = WspxConfig()
        assert cfg.lab.grace == 900
        assert cfg.loop_guard.enabled
        assert cfg.loop_guard.repeat_threshold == 3
        assert cfg.policy.protected_branches == ["main"]

    def test_doc_example(self) -> None:
        cfg = WspxConfig.from_toml(
            """
            [model]
            id = "Qwen/Qwen3.5-4B"
            revision = "3f2a9c1"
            dtype = "bfloat16"
            trust_remote_code = false

            [driver]
            model = "anthropic:claude-opus-5-5"

            [lab]
            gpu = "H100"
            grace = "15m"
            keep_warm = "auto"

            [loop_guard]
            repeat_threshold = 3

            [check]
            max_rel_norm = 0.05
            max_canary_kl = 0.10
            subulp_error_at = 0.5
            warn_dense_fraction = 0.05

            [tests]
            paths = ["wtests/"]

            [policy]
            protected_branches = ["main"]
            """
        )
        assert cfg.model.id == "Qwen/Qwen3.5-4B"
        assert cfg.driver.model == "anthropic:claude-opus-5-5"
        assert cfg.lab.gpu == "H100"

    def test_loop_guard_off(self) -> None:
        assert not WspxConfig.from_toml('loop_guard = "off"').loop_guard.enabled

    def test_limits_section_rejected(self) -> None:
        with pytest.raises(ConfigError, match="wspx limits set"):
            WspxConfig.from_toml("[limits]\ndollars = 5")

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"lab\.gpus"):
            WspxConfig.from_toml("[lab]\ngpus = 'H100'")

    def test_bad_toml(self) -> None:
        with pytest.raises(ConfigError):
            WspxConfig.from_toml("[lab")

    def test_discover(self, tmp_path: Path) -> None:
        (tmp_path / "wspx.toml").write_text('[driver]\nmodel = "openai:gpt-6"\n')
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_config(nested) == tmp_path / "wspx.toml"
        assert WspxConfig.discover(nested).driver.model == "openai:gpt-6"
