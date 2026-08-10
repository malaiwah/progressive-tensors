#!/usr/bin/env python3
"""fq_eps — Phase 0c analysis: ε curves, sensitivity variance, budget solve.

Consumes the encoder's per-layer done-JSONs (one work dir per K) and
produces the knob inputs of 01-artifacts-policy-stats.md §6:

- eps[L][e][K]   from `expert_rel_rt_mse` (per-expert round-trip MSE),
- φ[L][e]        from `expert_routed_count` (capture routing mass),
- per-layer sensitivity variance of the benefit signal Δε·φ,
- K2 abort check (04-milestones.md): homogeneous sensitivity across
  experts ⇒ per-expert allocation is pointless ⇒ pivot to layer-level,
- the global budget solve → n_k4_per_layer at given bpw budget points.

Usage: fq_eps.py --work-root /home/mbelleau/fq-0c --ks 2,3,4,5 \
                 --out <dir> [--budgets 0.25,0.42,0.5]
(budgets = fraction of experts at the upper tier, model-wide)

NumPy is an *extra*, not a runtime dependency.  The base install exists so
that someone who only wants to fetch, verify and assemble segments does not
have to build a numeric stack for it — everything on that path is stdlib plus
pynacl.  But the wheel still declares the `fq-eps` console script, so a base
install used to ship a command that died with `ModuleNotFoundError: numpy`
the instant it was run: a broken command, not a missing feature.  NumPy is
therefore imported on demand, inside the functions that do arithmetic, and a
base install gets `--help`, a clean diagnostic, and the name of the extra to
install.  CI installs the bare wheel and runs every entry point to keep it
that way.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

np = None  # bound by _require_numpy(); see the module docstring


class MissingNumPy(ImportError):
    """NumPy is in the [numeric] extra and this install does not have it."""


def _require_numpy():
    """Import NumPy on first use, or explain precisely how to get it."""
    global np
    if np is None:
        try:
            import numpy
        except ImportError as exc:                      # pragma: no cover
            raise MissingNumPy(
                "fq-eps needs NumPy, which the base install deliberately does "
                "not pull in (fetch/verify/assemble are stdlib + pynacl).\n"
                "Install the extra:  pip install 'progressive-tensors[numeric]'"
            ) from exc
        np = numpy
    return np


def load_eps(work_root: Path, ks: list[int]) -> tuple[dict, "np.ndarray", list[int]]:
    """Return (eps: {k: [L,E]}, phi: [L,E], layers)."""
    _require_numpy()
    eps: dict[int, list] = {}
    phi_rows, layers = [], []
    for k in ks:
        wd = work_root / f"work-k{k}-tr3"
        rows = {}
        for p in sorted(wd.glob("layer-*.done.json")):
            d = json.loads(p.read_text())
            rows[d["layer"]] = (d["expert_rel_rt_mse"], d["expert_routed_count"])
        eps[k] = rows
    layers = sorted(set.intersection(*(set(eps[k]) for k in ks)))
    if not layers:
        raise SystemExit("no common completed layers across all Ks")
    phi = np.array([eps[ks[0]][L][1] for L in layers], dtype=np.float64)
    out = {k: np.array([eps[k][L][0] for L in layers], dtype=np.float64)
           for k in ks}
    return out, phi, layers


def analyze(eps: dict, phi: "np.ndarray", layers: list[int],
            lo: int = 3, hi: int = 4) -> dict:
    """Benefit signal, variance table, K2 verdict for the (lo→hi) upgrade."""
    _require_numpy()
    delta = eps[lo] - eps[hi]                      # [L,E] error reduction
    phi_n = phi / np.maximum(phi.sum(axis=1, keepdims=True), 1)
    benefit = delta * phi_n                        # spec D5: ε-gap × mass
    per_layer = []
    for i, L in enumerate(layers):
        b = benefit[i]
        per_layer.append({
            "layer": L,
            "delta_eps_mean": float(delta[i].mean()),
            "delta_eps_cv": float(delta[i].std() / max(abs(delta[i].mean()), 1e-12)),
            "benefit_sum": float(b.sum()),
            "benefit_gini": gini(b),
            "top16_benefit_share": float(
                np.sort(b)[-16:].sum() / max(b.sum(), 1e-18)),
        })
    cvs = [r["delta_eps_cv"] for r in per_layer]
    ginis = [r["benefit_gini"] for r in per_layer]
    # K2 abort: homogeneous sensitivity = low spread AND low concentration.
    k2_fires = (np.median(cvs) < 0.1) and (np.median(ginis) < 0.2)
    return {"per_layer": per_layer,
            "k2_abort": {"fires": bool(k2_fires),
                         "median_delta_eps_cv": float(np.median(cvs)),
                         "median_benefit_gini": float(np.median(ginis))}}


def gini(x: "np.ndarray") -> float:
    _require_numpy()
    x = np.sort(np.clip(x, 0, None))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def budget_solve(eps: dict, phi: "np.ndarray", layers: list[int],
                 frac: float, lo: int = 3, hi: int = 4) -> dict:
    """Global greedy solve: spend the model-wide upper-tier budget on the
    highest benefit (l,e) pairs → n_k4_per_layer (00-overview: 0c sets N_L)."""
    _require_numpy()
    L, E = phi.shape
    budget = int(round(frac * L * E))
    delta = eps[lo] - eps[hi]
    phi_n = phi / np.maximum(phi.sum(axis=1, keepdims=True), 1)
    benefit = (delta * phi_n).flatten()
    order = np.argsort(-benefit)[:budget]
    counts = np.zeros(L, dtype=int)
    for idx in order:
        counts[idx // E] += 1
    total_benefit = float(benefit[order].sum())
    uniform = uniform_baseline(delta * phi_n, budget, L, E)
    return {
        "budget_frac": frac,
        "budget_experts": budget,
        "n_k4_per_layer": {str(layers[i]): int(counts[i]) for i in range(L)},
        "benefit_captured": total_benefit,
        "benefit_uniform_baseline": uniform,
        "advantage_vs_uniform_pct": (
            100.0 * (total_benefit - uniform) / max(uniform, 1e-18)),
    }


def uniform_baseline(benefit_le, budget: int, L: int, E: int) -> float:
    """Spend the SAME total budget as the solve, distributed as evenly as
    integer counts allow (largest-remainder: the budget % L leftover slots
    go one each to the first layers). Discarding the remainder would give
    the baseline fewer slots than the solve and inflate the reported
    advantage — peer-review finding, 2026-08-10."""
    _require_numpy()
    base, rem = divmod(budget, L)
    total = 0.0
    for i in range(L):
        take = min(base + (1 if i < rem else 0), E)
        row = np.sort(benefit_le[i])[::-1]
        total += float(row[:take].sum())
    return total

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--ks", default="2,3,4,5")
    ap.add_argument("--budgets", default="0.25,0.42,0.5")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ks = [int(k) for k in args.ks.split(",")]
    try:
        _require_numpy()
    except MissingNumPy as e:
        print(e, file=sys.stderr)
        return 2
    eps, phi, layers = load_eps(args.work_root, ks)

    result = {"layers": layers, "ks": ks,
              "mean_eps_per_k": {str(k): float(eps[k].mean()) for k in ks}}
    result["analysis_k3_to_k4"] = analyze(eps, phi, layers, 3, 4)
    if 5 in eps:
        result["analysis_k4_to_k5"] = analyze(eps, phi, layers, 4, 5)
    if 2 in eps:
        result["analysis_k2_to_k3"] = analyze(eps, phi, layers, 2, 3)
    result["solves"] = [
        budget_solve(eps, phi, layers, float(f))
        for f in args.budgets.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eps-analysis.json").write_text(json.dumps(result, indent=1))
    a = result["analysis_k3_to_k4"]
    print(f"layers analyzed: {len(layers)}, Ks: {ks}")
    print("mean eps per K:", result["mean_eps_per_k"])
    print(f"K2 abort fires: {a['k2_abort']}")
    for s in result["solves"]:
        print(f"budget {s['budget_frac']}: advantage vs uniform "
              f"{s['advantage_vs_uniform_pct']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
