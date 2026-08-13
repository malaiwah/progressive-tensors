#!/usr/bin/env python3
"""Measure original-space MSE for real SIQ and simulated MSRT weights."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import fq_assemble_lora as lora
from fq_verify import decode_proj

EXL3_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.rank(\d+)\."
    r"(trellis|suh|svh|mcg)$"
)


def parse_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        values = [int(token) for token in value.split(",") if token]
    except ValueError as exc:
        raise lora.CartridgeError(
            f"invalid comma-separated integer list {value!r}") from exc
    if (not values or any(value < 0 for value in values)
            or len(set(values)) != len(values)):
        raise lora.CartridgeError("ID lists must contain unique non-negative integers")
    return values


def select_stratified_experts(
    expert_ids: list[int], stages: list[dict[str, Any]], per_tier: int
) -> dict[str, list[int]]:
    """Group by emitted stage set and sample each tier independently."""
    tiers: dict[str, list[int]] = {}
    for expert in expert_ids:
        labels = [stage["label"] for stage in lora.stages_for_expert(stages, expert)]
        tier = "+".join(labels) if labels else "base"
        tiers.setdefault(tier, []).append(expert)
    if per_tier > 0:
        tiers = {tier: ids[:per_tier] for tier, ids in tiers.items()}
    return tiers


def load_siq_slots(path: Path, layer: int, expert: int, projection: str):
    """Load every rank slice for one actual SIQ expert projection."""
    from safetensors import safe_open

    ranks: dict[int, dict[str, Any]] = {}
    with safe_open(str(path), framework="pt") as source:
        keys = list(source.keys())
        for key in keys:
            match = EXL3_KEY_RE.fullmatch(key)
            if not match:
                continue
            key_layer, key_expert, key_projection, rank, component = (
                int(match.group(1)), int(match.group(2)), match.group(3),
                int(match.group(4)), match.group(5))
            if (key_layer, key_expert, key_projection) != (
                    layer, expert, projection):
                continue
            ranks.setdefault(rank, {})[component] = source.get_tensor(key)
    if not ranks:
        raise lora.CartridgeError(
            f"{path}: no SIQ tensors for layer {layer} expert {expert} {projection}")
    required = {"trellis", "suh", "svh", "mcg"}
    for rank, slot in ranks.items():
        missing = required - set(slot)
        if missing:
            raise lora.CartridgeError(
                f"{path}: rank {rank} is missing SIQ components {sorted(missing)}")
    return [ranks[rank] for rank in sorted(ranks)]


def mse(reference, reconstructed) -> float:
    return (reference.float() - reconstructed.float()).square().mean().item()


def simulate_msrt(weight, recipe, device, enc, expert: int) -> dict[str, float]:
    """Measure every graph node through the production encoder itself.

    The encoder already reports original-space MSE per node, so this tool never
    keeps a second copy of the quantization pipeline that could drift from the
    one that writes checkpoints.
    """
    nodes = lora.encode_matrix_dag(
        weight, recipe["bases"],
        lora.stages_for_expert(recipe["stages"], expert), device, enc)
    return {f"msrt_{label}": node["mse"] for label, node in nodes.items()}


def run_measurement(args) -> dict[str, Any]:
    lora.require_quant_dependencies()
    recipe = lora.load_recipe(args.recipe)
    layers = parse_ids(args.layers) or recipe["moe_layers"]
    requested_experts = parse_ids(args.experts)
    bf16_shards = lora.resolve_layer_shards(args.bf16, layers)
    siq_shards = lora.resolve_layer_shards(args.siq, layers)
    device = lora.torch.device(args.device)
    if device.type == "cuda" and not lora.torch.cuda.is_available():
        raise lora.CartridgeError(f"--device {device}: CUDA is unavailable")

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    samples: dict[str, Any] = {}
    encoder_parent = str(args.encoder_source.expanduser().resolve().parent)
    sys.path.insert(0, encoder_parent)
    try:
        from safetensors import safe_open

        with lora.bootstrap_encoder(args.encoder_source) as enc:
            for layer in layers:
                with safe_open(str(bf16_shards[layer]), framework="pt") as source:
                    keys = list(source.keys())
                    experts = lora.inspect_source_layer(keys, layer)
                    available = sorted(experts)
                    if requested_experts is not None:
                        missing = set(requested_experts) - set(available)
                        if missing:
                            raise lora.CartridgeError(
                                f"layer {layer}: requested experts missing: "
                                f"{sorted(missing)}")
                        tiers = {"explicit": requested_experts}
                    else:
                        tiers = select_stratified_experts(
                            available, recipe["stages"], args.experts_per_tier)
                    selected = [expert for ids in tiers.values() for expert in ids]
                    samples[str(layer)] = {"tiers": tiers, "experts": selected}

                    for expert in selected:
                        for projection in lora.PROJECTIONS:
                            source_weight = source.get_tensor(
                                experts[expert][projection]).float()
                            internal = source_weight.T.contiguous().to(device)
                            for name, value in simulate_msrt(
                                    internal, recipe, device, enc,
                                    expert).items():
                                sums[name] = sums.get(name, 0.0) + value
                                counts[name] = counts.get(name, 0) + 1

                            slots = load_siq_slots(
                                siq_shards[layer], layer, expert, projection)
                            decoded = decode_proj(slots, projection, str(device))
                            if decoded.shape != source_weight.shape:
                                raise lora.CartridgeError(
                                    f"SIQ shape {tuple(decoded.shape)} != BF16 shape "
                                    f"{tuple(source_weight.shape)} for layer {layer} "
                                    f"expert {expert} {projection}")
                            value = mse(source_weight, decoded.cpu())
                            sums["siq_actual"] = sums.get("siq_actual", 0.0) + value
                            counts["siq_actual"] = counts.get("siq_actual", 0) + 1
                            del source_weight, internal, decoded, slots
                if device.type == "cuda":
                    lora.torch.cuda.empty_cache()
    finally:
        if sys.path and sys.path[0] == encoder_parent:
            sys.path.pop(0)

    if not counts:
        raise lora.CartridgeError("measurement selected no projections")
    return {
        "schema": "fq-msrt-mse/1",
        "recipe": str(args.recipe),
        "bf16": str(args.bf16),
        "siq": str(args.siq),
        "layers": layers,
        "sampling": {
            "experts_explicit": requested_experts,
            "experts_per_tier": args.experts_per_tier,
            "selected": samples,
        },
        "metrics": {
            name: {"mse": sums[name] / counts[name], "projections": counts[name]}
            for name in sorted(counts)
        },
    }


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", required=True, type=Path)
    parser.add_argument("--siq", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--encoder-source", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--layers", help="Comma-separated layer IDs; defaults to the recipe")
    parser.add_argument(
        "--experts", help="Explicit comma-separated expert IDs for every layer")
    parser.add_argument(
        "--experts-per-tier", type=int, default=10,
        help="Stratified experts per emitted stage chain; 0 means all")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.experts_per_tier < 0:
        parser.error("--experts-per-tier must be non-negative")
    try:
        results = run_measurement(args)
        write_json_atomic(args.out, results)
    except (lora.CartridgeError, ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(results["metrics"], indent=2))
    print(f"Results saved to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
