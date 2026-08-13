#!/usr/bin/env python3
"""Combine validated MSRT stage shards into one sharded custom adapter."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import fq_assemble_lora as lora
from fq_assemble import AssemblyError, StagedOutput, check_out_dir

STAGE_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.rank(\d+)\."
    r"(trellis|suh|svh|scale)_([A-Za-z0-9_-]{1,32})$"
)
COMPONENTS = {"trellis", "suh", "svh", "scale"}


def load_stage_config(
    path: Path, expected_stage: dict[str, Any], identity: tuple[int, str] | None
) -> tuple[dict[str, Any], tuple[int, str]]:
    """Validate one stage config and its base-checkpoint identity."""
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise lora.CartridgeError(f"{path}: invalid adapter config ({exc})") from exc
    if not isinstance(config, dict) or config.get("schema") != lora.ADAPTER_CONFIG_SCHEMA:
        raise lora.CartridgeError(
            f"{path}: schema must be {lora.ADAPTER_CONFIG_SCHEMA!r}")
    if config.get("format") != "exl3-msrt-full-rank":
        raise lora.CartridgeError(f"{path}: unsupported format {config.get('format')!r}")
    if config.get("standard_lora_compatible") is not False:
        raise lora.CartridgeError(f"{path}: standard_lora_compatible must be false")
    stages = config.get("stages")
    expected = {
        "label": expected_stage["label"],
        "k": expected_stage["k"],
        "experts": expected_stage["experts"],
    }
    if stages != [expected]:
        raise lora.CartridgeError(
            f"{path}: stage metadata {stages!r} != recipe {expected!r}")
    if (not isinstance(config.get("shards"), list)
            or not config["shards"]
            or any(not isinstance(value, str) or Path(value).is_absolute()
                   or ".." in Path(value).parts for value in config["shards"])):
        raise lora.CartridgeError(f"{path}: shards must be safe relative paths")
    current = (config.get("base_k"), config.get("base_manifest_sha256"))
    if (isinstance(current[0], bool) or not isinstance(current[0], int)
            or not isinstance(current[1], str) or len(current[1]) != 64):
        raise lora.CartridgeError(f"{path}: invalid base checkpoint identity")
    if identity is not None and current != identity:
        raise lora.CartridgeError(
            f"{path}: base checkpoint identity {current!r} != {identity!r}")
    return config, current


def validate_stage_tensors(
    tensors: dict[str, Any], stage: dict[str, Any], shard: str
) -> None:
    """Require exact components, K geometry, projection, and expert coverage."""
    label = stage["label"]
    groups: dict[tuple[int, int, str, int], set[str]] = {}
    observed_experts: dict[int, set[int]] = {}
    for key, tensor in tensors.items():
        match = STAGE_KEY_RE.fullmatch(key)
        if not match:
            raise lora.CartridgeError(f"{shard}: unexpected tensor key {key!r}")
        layer, expert, projection, rank, component, key_label = (
            int(match.group(1)), int(match.group(2)), match.group(3),
            int(match.group(4)), match.group(5), match.group(6))
        if key_label != label:
            raise lora.CartridgeError(
                f"{shard}: tensor {key!r} carries label {key_label!r}, expected {label!r}")
        groups.setdefault((layer, expert, projection, rank), set()).add(component)
        observed_experts.setdefault(layer, set()).add(expert)
        if component == "trellis":
            if tensor.ndim != 3 or tensor.shape[-1] != stage["k"] * 16:
                raise lora.CartridgeError(
                    f"{shard}: {key} has shape {tuple(tensor.shape)}, expected last "
                    f"dimension {stage['k'] * 16}")
        elif component in {"suh", "svh"} and tensor.ndim != 1:
            raise lora.CartridgeError(f"{shard}: {key} must be a vector")
        elif component == "scale" and tensor.ndim != 0:
            raise lora.CartridgeError(f"{shard}: {key} must be a scalar")
    if not groups:
        raise lora.CartridgeError(f"{shard}: stage contains no tensors")
    for group, components in groups.items():
        if components != COMPONENTS:
            raise lora.CartridgeError(
                f"{shard}: {group} components {sorted(components)} != "
                f"{sorted(COMPONENTS)}")
    for layer, experts in observed_experts.items():
        expected_experts = (
            experts if stage["experts"] == "all" else set(stage["experts"])
        )
        if experts != expected_experts:
            raise lora.CartridgeError(
                f"{shard}: layer {layer} experts {sorted(experts)} != "
                f"recipe {sorted(expected_experts)}")
        expected_groups = {
            (layer, expert, projection, 0)
            for expert in expected_experts for projection in lora.PROJECTIONS
        }
        actual_groups = {group for group in groups if group[0] == layer}
        if actual_groups != expected_groups:
            raise lora.CartridgeError(
                f"{shard}: incomplete expert/projection/rank coverage")


def combine(args) -> int:
    lora.require_quant_dependencies()
    recipe = lora.load_recipe(args.recipe)
    root = args.cartridges.expanduser().resolve()
    if not root.is_dir():
        raise lora.CartridgeError(f"--cartridges {root} is not a directory")
    out = check_out_dir(args.out, source=root, policy=args.recipe)

    configs: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    identity = None
    basenames = None
    for stage in recipe["stages"]:
        stage_dir = root / stage["label"]
        config, identity = load_stage_config(
            stage_dir / "adapter_config.json", stage, identity)
        names = {Path(value).name for value in config["shards"]}
        if len(names) != len(config["shards"]):
            raise lora.CartridgeError(
                f"{stage_dir}: duplicate shard basenames are ambiguous")
        if basenames is None:
            basenames = names
        elif names != basenames:
            raise lora.CartridgeError(
                f"{stage_dir}: shard coverage {sorted(names)} != {sorted(basenames)}")
        configs.append((stage, config, stage_dir))

    from safetensors import safe_open

    staged = StagedOutput(out, args.force)
    work = staged.begin()
    output_shards: list[str] = []
    tensor_count = 0
    try:
        for basename in sorted(basenames or []):
            combined: dict[str, Any] = {}
            for stage, config, stage_dir in configs:
                relative = next(
                    value for value in config["shards"]
                    if Path(value).name == basename)
                source = stage_dir / relative
                if not source.is_file():
                    # Encoder configs store paths relative to cartridge root.
                    source = root / relative
                if not source.is_file():
                    raise lora.CartridgeError(f"missing stage shard {relative}")
                with safe_open(str(source), framework="pt") as handle:
                    tensors = {
                        key: handle.get_tensor(key) for key in handle
                    }
                validate_stage_tensors(tensors, stage, str(source))
                collisions = set(combined) & set(tensors)
                if collisions:
                    raise lora.CartridgeError(
                        f"{source}: duplicate tensor keys {sorted(collisions)[:3]}")
                combined.update(tensors)
            lora.save_safetensors(combined, work / basename)
            output_shards.append(basename)
            tensor_count += len(combined)
        lora.write_adapter_config(
            work, identity[0], identity[1], recipe["stages"],
            output_shards, tensor_count)
        staged.commit()
    except Exception:
        staged.abort()
        raise
    print(f"Combined {len(output_shards)} shards into {out}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument(
        "--cartridges", required=True, type=Path,
        help="Directory containing one validated stage directory per recipe label")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        return combine(args)
    except (AssemblyError, lora.CartridgeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
