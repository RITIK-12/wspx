"""Tests and evals for weights (§5.7): suites are code.

.. code-block:: python

    from wspx.testing import suite, expect, load_jsonl

    @suite(split="dev", seed=0)
    def math_finishes(model):
        for item in load_jsonl("data/math_dev.jsonl"):
            out = model.generate(item["prompt"], max_tokens=4096)
            yield expect.regex(out, item["answer_regex"])

Simple cases can be YAML (``name``, ``split``, ``items: [{prompt, expect}]``).
Runs are paired (identical items and seeds on both sides), cached by the
weights they ran on, and held-out suites only ever return aggregates to the
driver.
"""

from wspx.testing.api import Check, Suite, expect, load_jsonl, suite

__all__ = ["Check", "Suite", "expect", "load_jsonl", "suite"]
