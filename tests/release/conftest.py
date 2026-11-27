"""The release matrix runs only when asked (``WSPX_RELEASE=1``): it loads real models on a
GPU, starts vLLM and calls real driver APIs, so it costs money. ``modal run
tests/release/modal_matrix.py`` runs it on Modal; see that file."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("WSPX_RELEASE") == "1":
        return
    skip = pytest.mark.skip(reason="the release matrix runs with WSPX_RELEASE=1 (GPUs, real APIs)")
    for item in items:
        if "release" in item.keywords:
            item.add_marker(skip)
