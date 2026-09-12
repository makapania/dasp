"""MultiGroupEPO must produce identical output regardless of PYTHONHASHSEED.

`MultiGroupEPO.fit` used to seed its pseudo-interferent RNG with `hash(label)`.
Python randomizes `str` hashing per process, and those draws feed the SVD that
builds the projection matrix applied to spectra in `transform()` -- so the
returned scientific data differed between runs of the same analysis. Measured
before the fix: max transformed-value difference of ~1.98 between two hash seeds.

These tests run in SUBPROCESSES on purpose. `PYTHONHASHSEED` is read once at
interpreter start, so setting `os.environ` inside a running test proves nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

# Emits the fitted projection and a transformed matrix as JSON so two
# interpreters started with different hash seeds can be compared exactly.
PROBE = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    sys.path.insert(0, {src!r})
    from spectral_predict.contaminant_analysis import MultiGroupEPO

    rng = np.random.RandomState(0)          # inputs identical across runs
    n_wl = 40
    base = np.linspace(0.1, 0.9, n_wl)
    clean = base + rng.normal(0, 0.01, size=(12, n_wl))
    groups = {{
        "glyptal":  base + 0.20 + rng.normal(0, 0.01, size=(8, n_wl)),
        "shellac":  base + 0.35 + rng.normal(0, 0.01, size=(8, n_wl)),
        "\\u00e9poxy-\\u00fcber": base + 0.05 + rng.normal(0, 0.01, size=(8, n_wl)),
    }}
    if {reverse!r}:
        groups = dict(reversed(list(groups.items())))

    epo = MultiGroupEPO(n_components_per_group=2)
    epo.fit(clean, groups)
    out = epo.transform(clean)
    print(json.dumps({{
        "labels": list(epo.group_labels_),
        "transformed": np.asarray(out, dtype=float).ravel().tolist(),
    }}))
    """
)


def _run(hashseed: str, reverse: bool = False) -> dict:
    import os
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[1] / "src")
    code = PROBE.format(src=src, reverse=reverse)
    # Inherit the full environment and override only the seed. A minimal env
    # breaks Winsock initialisation on Windows (asyncio's _overlapped fails with
    # WinError 10106), which looks like a test failure but is not one.
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        pytest.fail(f"probe failed (PYTHONHASHSEED={hashseed}):\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_transform_is_identical_across_hash_seeds():
    """The whole point: same inputs, different hash salt, same numbers out."""
    a = _run("1")
    b = _run("2")

    assert a["labels"] == b["labels"]
    np.testing.assert_array_equal(
        np.array(a["transformed"]),
        np.array(b["transformed"]),
        err_msg="MultiGroupEPO output depends on PYTHONHASHSEED",
    )


def test_group_insertion_order_does_not_change_output():
    """Group order set the interferent-library row order, and so the SVD input.

    Two callers passing the same groups built in a different order got different
    projections (~5e-15, small but real). Sorting makes the result a function of
    the groups rather than of how the caller happened to assemble the dict.
    """
    forward = _run("1", reverse=False)
    backward = _run("1", reverse=True)

    assert forward["labels"] == backward["labels"], "group_labels_ must be canonical"
    np.testing.assert_array_equal(
        np.array(forward["transformed"]),
        np.array(backward["transformed"]),
        err_msg="MultiGroupEPO output depends on group insertion order",
    )


def test_group_labels_are_sorted():
    """Sorted order is the contract; assert it directly rather than by inference.

    Uses reverse=True deliberately. In forward order the probe's labels already
    happen to be in sorted order, so asserting against that passes even with the
    bug present -- a vacuous test. Feeding them in reverse is what actually
    distinguishes "sorted" from "whatever the caller passed".
    """
    result = _run("1", reverse=True)
    assert result["labels"] == sorted(result["labels"])
