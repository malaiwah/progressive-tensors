"""Tests for fq_eps: loading, K2-abort logic, budget solve properties."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_eps  # noqa: E402

L_IDS, E = [3, 4], 8


def write_work(root: Path, k: int, eps_fn, counts_fn):
    wd = root / f"work-k{k}-tr3"
    wd.mkdir(parents=True, exist_ok=True)
    for L in L_IDS:
        doc = {"layer": L, "bits": k,
               "expert_rel_rt_mse": [eps_fn(k, L, e) for e in range(E)],
               "expert_routed_count": [counts_fn(L, e) for e in range(E)]}
        (wd / f"layer-{L:03d}.done.json").write_text(json.dumps(doc))


@pytest.fixture()
def hetero(tmp_path):
    # expert 0 is hot and very sensitive; ε halves per +1 bit
    write_work(tmp_path, 3, lambda k, L, e: (0.4 if e == 0 else 0.02),
               lambda L, e: 10000 if e == 0 else 100)
    write_work(tmp_path, 4, lambda k, L, e: (0.2 if e == 0 else 0.01),
               lambda L, e: 10000 if e == 0 else 100)
    return tmp_path


def test_load_and_shapes(hetero):
    eps, phi, layers = fq_eps.load_eps(hetero, [3, 4])
    assert layers == L_IDS and eps[3].shape == (2, E) == phi.shape


def test_k2_abort_fires_on_homogeneous(tmp_path):
    write_work(tmp_path, 3, lambda k, L, e: 0.02, lambda L, e: 100)
    write_work(tmp_path, 4, lambda k, L, e: 0.01, lambda L, e: 100)
    eps, phi, layers = fq_eps.load_eps(tmp_path, [3, 4])
    a = fq_eps.analyze(eps, phi, layers)
    assert a["k2_abort"]["fires"] is True


def test_k2_abort_quiet_on_heterogeneous(hetero):
    eps, phi, layers = fq_eps.load_eps(hetero, [3, 4])
    a = fq_eps.analyze(eps, phi, layers)
    assert a["k2_abort"]["fires"] is False


def test_budget_solve_targets_hot_sensitive_expert(hetero):
    eps, phi, layers = fq_eps.load_eps(hetero, [3, 4])
    s = fq_eps.budget_solve(eps, phi, layers, frac=2 / (2 * E))
    assert s["budget_experts"] == 2
    # both budget slots go to expert 0 of each layer -> 1 per layer
    assert list(s["n_k4_per_layer"].values()) == [1, 1]
    assert s["advantage_vs_uniform_pct"] >= 0


def test_budget_conservation(hetero):
    eps, phi, layers = fq_eps.load_eps(hetero, [3, 4])
    for frac in (0.25, 0.5, 1.0):
        s = fq_eps.budget_solve(eps, phi, layers, frac)
        assert sum(s["n_k4_per_layer"].values()) == s["budget_experts"]
