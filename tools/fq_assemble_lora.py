#!/usr/bin/env python3
"""fq_assemble_lora — Encode BF16 weights as MSRT EXL3 cartridges.

Given a BF16 checkpoint and an ``fq-cartridge/1`` recipe, this tool:
  1. emits a complete, loadable EXL3 base checkpoint;
  2. quantizes selected residual stages with MSRT; and
  3. emits sharded custom cartridge adapters plus an explicit runtime contract.

Cartridges are full-rank additive trellis weights, not PEFT/LoRA matrices.
Their execution pattern is LoRA-like (base GEMM plus correction GEMMs), but
standard vLLM/SGLang ``add_lora`` APIs cannot load them without an EXL3 MSRT
runtime implementation. The emitted ``fq-cartridge-adapter/1`` config records
that custom contract instead of claiming standard LoRA compatibility.

MSRT (Multi-Stage Rescaled Trellis) is described in:
  research/fungible-quant/poc/V50-LOW-BITRATE-MSRT.md
  research/fungible-quant/MSRT-CARTRIDGE-FEASIBILITY-AND-PLAN.md

Cartridge recipe format (fq-cartridge/1):

  {
    "schema": "fq-cartridge/1",
    "base_k": 2,
    "stages": [
      {"k": 1, "label": "res1", "experts": "all"},
      {"k": 2, "label": "res2", "experts": [0, 1, 10, 11, ...]}
    ],
    "moe_layers": [3, 4, 5, ...]
  }

Usage:
  python tools/fq_assemble_lora.py encode \\
    --source ./bf16-checkpoint \\
    --recipe recipes/fruit-k2-k3k4-cart.json \\
    --out ./output \\
    --encoder-source /opt/fruit-pip/exllamav3
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # Base installs must still support --help.
    torch = None

sys.path.insert(0, str(Path(__file__).parent))
from fq_assemble import (
    AssemblyError,
    StagedOutput,
    check_out_dir,
    regenerate_manifest,
    regenerate_shard_index,
    sha256_file,
)
from fq_repack import PROJ_ORDER

# ── Constants ──────────────────────────────────────────────────────────────

TOOL_VERSION = "fq_assemble_lora/2"
CARTRIDGE_SCHEMA = "fq-cartridge/1"
ADAPTER_CONFIG_SCHEMA = "fq-cartridge-adapter/1"
HADAMARD_BLOCK = 128
MCG_SENTINEL_SIGNED = 0xCBAC1FED - (1 << 32)
LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
BF16_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
PROJECTIONS = tuple(sorted(PROJ_ORDER, key=PROJ_ORDER.get))


class CartridgeError(RuntimeError):
    """A recipe, source checkpoint, or encoded artifact is invalid."""


def require_quant_dependencies() -> None:
    if torch is None:
        raise CartridgeError(
            "MSRT encoding requires the 'quant' extra: "
            "pip install 'progressive-tensors[quant]'")
    try:
        import safetensors  # noqa: F401
    except ModuleNotFoundError as exc:
        raise CartridgeError(
            "MSRT encoding requires the 'quant' extra: "
            "pip install 'progressive-tensors[quant]'") from exc


# ── EXL3 Encoder Bootstrap ────────────────────────────────────────────────

@contextmanager
def bootstrap_encoder(encoder_source: str):
    """Load trusted EXL3 encoder modules temporarily and restore sys.modules."""
    import importlib.util
    import types

    pkg_root = Path(encoder_source).expanduser().resolve()
    required = (
        pkg_root / "ext.py",
        pkg_root / "util" / "hadamard.py",
        pkg_root / "modules" / "quant" / "exl3_lib" / "quantize.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise CartridgeError(
            f"--encoder-source {pkg_root} is not an exllamav3 checkout; "
            f"missing {missing}")

    names = [
        "exllamav3",
        "exllamav3.util",
        "exllamav3.modules",
        "exllamav3.modules.quant",
        "exllamav3.modules.quant.exl3_lib",
        "exllamav3.util.progress",
        "exllamav3.util.memory",
        "exllamav3.util.tensor",
        "exllamav3.ext",
        "exllamav3.util.hadamard",
        "exllamav3.modules.quant.exl3_lib.quantize",
    ]
    previous = {name: sys.modules.get(name) for name in names}
    try:
        pkg = types.ModuleType("exllamav3")
        pkg.__path__ = [str(pkg_root)]
        sys.modules["exllamav3"] = pkg
        for sub in ["util", "modules", "modules.quant", "modules.quant.exl3_lib"]:
            full = f"exllamav3.{sub}"
            module = types.ModuleType(full)
            module.__path__ = [str(pkg_root / sub.replace(".", "/"))]
            sys.modules[full] = module

        progress = types.ModuleType("exllamav3.util.progress")

        class _DisabledProgress:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def update(self, *args): pass
            def new_task(self, *args, **kwargs): pass

        progress.ProgressBar = _DisabledProgress
        sys.modules["exllamav3.util.progress"] = progress

        memory = types.ModuleType("exllamav3.util.memory")
        memory.free_mem = lambda: None
        memory.list_gpu_tensors = list
        sys.modules["exllamav3.util.memory"] = memory

        util = types.ModuleType("exllamav3.util")
        util.__path__ = [str(pkg_root / "util")]
        util.cuda_sync_active = (
            lambda *args, **kwargs:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        )
        sys.modules["exllamav3.util"] = util

        tensor = types.ModuleType("exllamav3.util.tensor")
        tensor.save_tensor_image = lambda *args, **kwargs: None
        sys.modules["exllamav3.util.tensor"] = tensor

        def load(name: str, path: Path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            if spec is None or spec.loader is None:
                raise CartridgeError(f"cannot load encoder module {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        ext_mod = load("exllamav3.ext", required[0])
        had_mod = load("exllamav3.util.hadamard", required[1])
        quant_mod = load(
            "exllamav3.modules.quant.exl3_lib.quantize", required[2])
        ext = getattr(ext_mod, "exllamav3_ext", None)
        if not callable(getattr(ext, "pack_trellis", None)):
            raise CartridgeError(
                f"{pkg_root}: exllamav3 extension lacks pack_trellis")
        yield (
            ext, had_mod.get_hadamard_dt,
            quant_mod.tensor_core_perm, quant_mod.tensor_core_perm_i,
            quant_mod.quantize_tiles, quant_mod.codebook_scale,
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


# ── Quantization Primitives ────────────────────────────────────────────────

def block_rms(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """RMS along a dimension."""
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()


def validate_quant_shape(w: torch.Tensor, *, who: str = "weight") -> tuple[int, int]:
    """Return a valid EXL3 matrix shape or raise a useful error."""
    if w.ndim != 2:
        raise ValueError(f"{who}: expected a 2-D BF16 weight, got shape {tuple(w.shape)}")
    k, n = w.shape
    if k % HADAMARD_BLOCK or n % HADAMARD_BLOCK:
        raise ValueError(
            f"{who}: shape {(k, n)} must be divisible by Hadamard block "
            f"{HADAMARD_BLOCK} on both axes")
    if k % 16 or n % 16:
        raise ValueError(f"{who}: shape {(k, n)} must be divisible by trellis tile 16")
    return k, n


def _finite_fp16_scale(x: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Round scales once while keeping every divisor finite and non-zero."""
    minimum = torch.finfo(torch.float16).tiny
    safe = sign * x.abs().clamp_min(minimum)
    rounded = safe.to(torch.float16)
    if not torch.isfinite(rounded).all() or (rounded == 0).any():
        raise ValueError("Hadamard scale vector contains zero or non-finite values")
    return rounded


def regularize_with_vectors(
    w: torch.Tensor,
    device: torch.device,
    ghd: Any,
    cbs: float,
    had_k: int = HADAMARD_BLOCK,
    had_n: int = HADAMARD_BLOCK,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Regularize with the exact FP16 vectors that the checkpoint stores."""
    k, n = validate_quant_shape(w)
    if not math.isfinite(float(cbs)) or float(cbs) == 0:
        raise ValueError(f"codebook_scale must be finite and non-zero, got {cbs!r}")

    g = torch.Generator(device="cpu").manual_seed(seed)
    su_sign = (torch.randn(k, generator=g).sign() + 1e-5).sign().to(device)
    sv_sign = (torch.randn(n, generator=g).sign() + 1e-5).sign().to(device)

    out_scales = block_rms(w, dim=0)
    mean = out_scales.mean()
    if torch.isfinite(mean) and mean.item() > 0:
        out_scales = out_scales / mean
    svh = _finite_fp16_scale(out_scales, sv_sign)
    transformed = (w / svh.float().unsqueeze(0)).contiguous()

    had_n_mat = ghd(had_n, device, torch.float, 1.0 / math.sqrt(had_n))
    transformed = (
        transformed.view(k, n // had_n, had_n) @ had_n_mat
    ).view(k, n).contiguous()

    in_scales = block_rms(transformed, dim=1)
    suh_sign = su_sign * (-1.0 if cbs > 0 else 1.0)
    suh = _finite_fp16_scale(in_scales / abs(float(cbs)), suh_sign)
    transformed = (transformed / suh.float().unsqueeze(1)).contiguous()

    had_k_mat = ghd(had_k, device, torch.float, 1.0 / math.sqrt(had_k))
    transformed = (
        had_k_mat @ transformed.view(k // had_k, had_k, n)
    ).view(k, n).contiguous()
    if not torch.isfinite(transformed).all():
        raise ValueError("regularized weight contains non-finite values")
    return transformed, suh.contiguous(), svh.contiguous()


def regularize(
    w: torch.Tensor,
    device: torch.device,
    ghd: Any,
    cbs: float,
    had_k: int = HADAMARD_BLOCK,
    had_n: int = HADAMARD_BLOCK,
    seed: int = 0,
) -> torch.Tensor:
    """Compatibility wrapper returning only the regularized weight."""
    return regularize_with_vectors(
        w, device, ghd, cbs, had_k=had_k, had_n=had_n, seed=seed
    )[0]


def inverse_regularize(
    w_reg: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    device: torch.device,
    ghd: Any,
    had_k: int = HADAMARD_BLOCK,
    had_n: int = HADAMARD_BLOCK,
) -> torch.Tensor:
    """Invert regularization using the exact serialized FP16 scale vectors."""
    k, n = validate_quant_shape(w_reg, who="reconstruction")
    had_k_mat = ghd(had_k, device, torch.float, 1.0 / math.sqrt(had_k))
    restored = (
        had_k_mat.transpose(0, 1)
        @ w_reg.view(k // had_k, had_k, n)
    ).view(k, n)
    restored = restored * suh.float().unsqueeze(1)
    had_n_mat = ghd(had_n, device, torch.float, 1.0 / math.sqrt(had_n))
    restored = (
        restored.view(k, n // had_n, had_n)
        @ had_n_mat.transpose(0, 1)
    ).view(k, n)
    restored = restored * svh.float().unsqueeze(0)
    return restored.contiguous()


def _validate_k(K: int, *, who: str = "K") -> int:
    if isinstance(K, bool) or not isinstance(K, int) or not 1 <= K <= 8:
        raise ValueError(f"{who} must be an integer in 1..8, got {K!r}")
    return K


def quantize_trellis(
    data: torch.Tensor,
    K: int,
    device: torch.device,
    tcp: Any,
    tcpi: Any,
    qtf: Any,
) -> torch.Tensor:
    """Quantize a valid 2-D EXL3 matrix and return its reconstruction."""
    _validate_k(K)
    k, n = validate_quant_shape(data)
    tiles_n = n // 16
    weight_q = torch.zeros_like(data)
    qa = {"K": K, "mcg": True}
    perm = tcp(device)
    perm_i = tcpi(device)

    for bi in range(0, k, 16):
        tiles = (
            data[bi:bi + 16].reshape(16, tiles_n, 16)
            .permute(1, 0, 2).reshape(tiles_n, 256)
        )
        quant_w, _ = qtf(tiles[:, perm].contiguous(), qa)
        quant_w = (
            quant_w[:, perm_i].reshape(tiles_n, 16, 16)
            .permute(1, 0, 2).reshape(16, n)
        )
        weight_q[bi:bi + 16] = quant_w
    return weight_q


def quantize_trellis_packed(
    data: torch.Tensor,
    K: int,
    device: torch.device,
    tcp: Any,
    tcpi: Any,
    qtf: Any,
    ext: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reconstruction plus the runtime-compatible packed EXL3 trellis."""
    _validate_k(K)
    if ext is None or not callable(getattr(ext, "pack_trellis", None)):
        raise RuntimeError(
            "the selected exllamav3 build lacks pack_trellis; refusing to "
            "write raw Viterbi indices as an EXL3 checkpoint")
    k, n = validate_quant_shape(data)
    tiles_n = n // 16
    weight_q = torch.zeros_like(data)
    raw_indices = torch.zeros(
        k // 16, tiles_n, 256, dtype=torch.int16, device=device
    )
    qa = {"K": K, "mcg": True}
    perm = tcp(device)
    perm_i = tcpi(device)

    for bi in range(0, k, 16):
        tk = bi // 16
        tiles = (
            data[bi:bi + 16].reshape(16, tiles_n, 16)
            .permute(1, 0, 2).reshape(tiles_n, 256)
        )
        quant_w, quant_idx = qtf(tiles[:, perm].contiguous(), qa)
        raw_indices[tk] = quant_idx
        quant_w = (
            quant_w[:, perm_i].reshape(tiles_n, 16, 16)
            .permute(1, 0, 2).reshape(16, n)
        )
        weight_q[bi:bi + 16] = quant_w

    packed = torch.zeros(
        (k // 16, tiles_n, K * 16), dtype=torch.int16, device=device
    )
    ext.pack_trellis(packed, raw_indices.contiguous(), K)
    return weight_q, packed


def rescaled_trellis_quantize(
    base_q: torch.Tensor,
    residual: torch.Tensor,
    K_res: int,
    device: torch.device,
    tcp: Any, tcpi: Any, qtf: Any,
    cbs: float,
    ext: Any,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Quantize one rescaled residual into runtime-compatible trellis form."""
    _validate_k(K_res, who="residual K")
    k, n = validate_quant_shape(residual, who="residual")
    if ext is None or not callable(getattr(ext, "pack_trellis", None)):
        raise RuntimeError("exllamav3 pack_trellis is required")
    residual_rms = residual.square().mean().sqrt().item()
    if not math.isfinite(residual_rms):
        raise ValueError("residual RMS is non-finite")
    if residual_rms < 1e-12:
        return base_q, torch.zeros(
            k // 16, n // 16, K_res * 16,
            dtype=torch.int16, device=device), 1.0

    scale = abs(float(cbs)) / residual_rms
    recon_scaled, packed = quantize_trellis_packed(
        residual * scale, K_res, device, tcp, tcpi, qtf, ext
    )
    return base_q + recon_scaled / scale, packed, scale


# ── Hadamard Vectors ──────────────────────────────────────────────────────

def compute_hadamard_vectors(
    w: torch.Tensor,
    device: torch.device,
    ghd: Any,
    cbs: float,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Return the same finite FP16 vectors used by regularization."""
    _, suh, svh = regularize_with_vectors(w, device, ghd, cbs, seed=seed)
    return {"suh": suh, "svh": svh}


# ── Encoding Pipeline ─────────────────────────────────────────────────────

def encode_expert_msrt(
    w_bf16: torch.Tensor,
    base_k: int,
    stages: list[dict[str, Any]],
    device: torch.device,
    ghd: Any, tcp: Any, tcpi: Any, qtf: Any,
    cbs: float,
    ext: Any,
) -> dict[str, Any]:
    """Encode one matrix and measure the reconstruction actually emitted."""
    _validate_k(base_k, who="base_k")
    w_reg, suh, svh = regularize_with_vectors(w_bf16, device, ghd, cbs)
    base_recon, base_packed = quantize_trellis_packed(
        w_reg, base_k, device, tcp, tcpi, qtf, ext
    )
    result = {
        "base": {
            "trellis": base_packed.cpu(),
            "suh": suh.cpu(),
            "svh": svh.cpu(),
        },
        "stages": {},
    }

    current_recon = base_recon
    for stage in stages:
        label = stage["label"]
        residual = w_reg - current_recon
        current_recon, packed, scale = rescaled_trellis_quantize(
            current_recon, residual, stage["k"], device,
            tcp, tcpi, qtf, cbs, ext
        )
        result["stages"][label] = {
            "trellis": packed.cpu(),
            "suh": suh.cpu(),
            "svh": svh.cpu(),
            "scale": scale,
        }

    reconstructed = inverse_regularize(
        current_recon, suh, svh, device, ghd
    )
    result["mse"] = (
        w_bf16.float() - reconstructed.float()
    ).square().mean().item()
    result["regularized_mse"] = (
        w_reg - current_recon
    ).square().mean().item()
    return result


# ── Safetensors Output ────────────────────────────────────────────────────

def save_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Save an independent tensor mapping after validating finite metadata."""
    from safetensors.torch import save_file

    if not tensors:
        raise CartridgeError(f"refusing to write empty safetensors file {path}")
    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise CartridgeError(f"{path}: tensor {name} contains non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def load_recipe(path: Path) -> dict[str, Any]:
    """Load and semantically validate one fq-cartridge/1 recipe."""
    try:
        recipe = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CartridgeError(f"{path}: cannot read cartridge recipe ({exc})") from exc
    if not isinstance(recipe, dict):
        raise CartridgeError(f"{path}: recipe must be a JSON object")
    if recipe.get("schema") != CARTRIDGE_SCHEMA:
        raise CartridgeError(
            f"{path}: schema must be {CARTRIDGE_SCHEMA!r}, "
            f"got {recipe.get('schema')!r}")
    _validate_k(recipe.get("base_k"), who="base_k")

    layers = recipe.get("moe_layers")
    if (not isinstance(layers, list) or not layers
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                   for v in layers)
            or len(set(layers)) != len(layers)):
        raise CartridgeError("moe_layers must be a non-empty list of unique integers")
    recipe["moe_layers"] = sorted(layers)

    stages = recipe.get("stages")
    if not isinstance(stages, list) or not stages:
        raise CartridgeError("stages must be a non-empty list")
    labels: set[str] = set()
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise CartridgeError(f"stage {index}: must be an object")
        _validate_k(stage.get("k"), who=f"stage {index} k")
        label = stage.get("label")
        if not isinstance(label, str) or not LABEL_RE.fullmatch(label):
            raise CartridgeError(
                f"stage {index}: label must match {LABEL_RE.pattern}")
        if label in labels:
            raise CartridgeError(f"stage {index}: duplicate label {label!r}")
        labels.add(label)
        experts = stage.get("experts")
        if experts == "all":
            continue
        if (not isinstance(experts, list)
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                       for v in experts)
                or len(set(experts)) != len(experts)):
            raise CartridgeError(
                f"stage {label!r}: experts must be 'all' or unique non-negative IDs")
        stage["experts"] = sorted(experts)
    return recipe


def effective_bpw(recipe: dict[str, Any], expert_count: int) -> float:
    """Nominal weight bits, excluding suh/svh metadata."""
    if expert_count <= 0:
        raise ValueError("expert_count must be positive")
    total = expert_count * recipe["base_k"]
    for stage in recipe["stages"]:
        selected = expert_count if stage["experts"] == "all" else len(stage["experts"])
        total += selected * stage["k"]
    return total / expert_count


def selected_stages(
    stages: list[dict[str, Any]], expert_id: int
) -> list[dict[str, Any]]:
    """Resolve stage applicability before residual chaining begins."""
    return [
        stage for stage in stages
        if stage["experts"] == "all" or expert_id in stage["experts"]
    ]


def resolve_layer_shards(source: Path, layers: list[int]) -> dict[int, Path]:
    """Require the supported one-safetensors-file-per-layer BF16 layout."""
    if not source.is_dir():
        raise CartridgeError(f"--source {source} is not a directory")
    if not (source / "config.json").is_file():
        raise CartridgeError(f"--source {source} has no config.json")
    resolved: dict[int, Path] = {}
    for layer in layers:
        candidates = [
            source / f"model-layer-{layer:03d}.safetensors",
            source / f"model-layer-{layer:04d}.safetensors",
        ]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            suffix = (
                "standard Hugging Face indexed shards are not supported by "
                "this encoder; convert to per-layer BF16 shards first"
                if not matches else f"ambiguous candidates: {matches}"
            )
            raise CartridgeError(f"layer {layer}: source shard missing; {suffix}")
        resolved[layer] = matches[0]
    return resolved


def inspect_source_layer(keys: list[str], layer: int) -> dict[int, dict[str, str]]:
    """Preflight exact BF16 expert coverage without loading weight payloads."""
    experts: dict[int, dict[str, str]] = {}
    for key in keys:
        match = BF16_EXPERT_RE.fullmatch(key)
        if not match:
            continue
        key_layer, expert, projection = (
            int(match.group(1)), int(match.group(2)), match.group(3))
        if key_layer != layer:
            raise CartridgeError(
                f"layer {layer} shard also contains BF16 expert tensor {key}")
        if projection in experts.setdefault(expert, {}):
            raise CartridgeError(
                f"layer {layer} expert {expert}: duplicate {projection}")
        experts[expert][projection] = key
    if not experts:
        raise CartridgeError(
            f"layer {layer}: no BF16 routed expert .weight tensors found")
    expected = set(PROJECTIONS)
    for expert, projections in experts.items():
        have = set(projections)
        if have != expected:
            raise CartridgeError(
                f"layer {layer} expert {expert}: projections "
                f"{sorted(have)} != {sorted(expected)}")
    return experts


def preserved_source_keys(keys: list[str]) -> list[str]:
    """Source tensors copied byte-for-value into a rewritten MoE shard."""
    return [key for key in keys if not BF16_EXPERT_RE.fullmatch(key)]


def load_preserved_tensors(source_file: Any, keys: list[str]) -> dict[str, Any]:
    """Materialize only non-expert tensors from an open source shard."""
    return {
        key: source_file.get_tensor(key)
        for key in preserved_source_keys(keys)
    }


def copy_source_checkpoint(
    source: Path, base_dir: Path, selected_shards: set[str]
) -> None:
    """Copy all untouched checkpoint content, excluding stale integrity files."""
    base_dir.mkdir(parents=True, exist_ok=True)
    skip = selected_shards | {
        "model.safetensors.index.json",
        "MANIFEST.sha256",
    }
    for entry in source.iterdir():
        if entry.name in skip:
            continue
        destination = base_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination)
        elif entry.is_file():
            shutil.copy2(entry, destination)


def write_base_metadata(
    base_dir: Path,
    base_k: int,
    layers: list[int],
    experts_per_layer: dict[int, int],
) -> None:
    """Synchronize loader-visible EXL3 config, bitmap, index, and manifest."""
    config_path = base_dir / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CartridgeError(f"{config_path}: invalid source config ({exc})") from exc
    if not isinstance(config, dict):
        raise CartridgeError(f"{config_path}: config must be an object")

    counts = set(experts_per_layer.values())
    tail = config.setdefault("hybrid_tr3_tail", {})
    if not isinstance(tail, dict):
        raise CartridgeError("config.json: hybrid_tr3_tail must be an object")
    tail.update({
        "format": "exl3-trellis",
        "bits": float(base_k),
        "codebook": "mcg",
        "moe_layers": [min(layers), max(layers)],
        "tensor_schema": (
            "model.layers.{L}.mlp.experts.{E}.{proj}.rank{rank}.{component}"),
        "tp": 1,
        "mcg_multiplier": 0xCBAC1FED,
    })
    if len(counts) == 1:
        tail["experts_per_layer"] = next(iter(counts))
    else:
        tail.pop("experts_per_layer", None)
    tail.pop("k_values", None)
    tail.pop("bits_per_expert", None)
    config["quantization_config"] = {
        "quant_method": "exl3",
        "bits": float(base_k),
        "codebook": "mcg",
        "version": "rank-sliced",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    bitmap_path = base_dir / "tier_bitmap.json"
    try:
        bitmap = json.loads(bitmap_path.read_text()) if bitmap_path.exists() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise CartridgeError(f"{bitmap_path}: invalid source bitmap ({exc})") from exc
    if not isinstance(bitmap, dict):
        raise CartridgeError(f"{bitmap_path}: bitmap must be an object")
    for layer, count in experts_per_layer.items():
        entry = bitmap.setdefault(str(layer), {})
        if not isinstance(entry, dict):
            entry = {}
            bitmap[str(layer)] = entry
        entry["k"] = [base_k] * count
        entry["bits_per_expert"] = [base_k] * count
    bitmap_path.write_text(json.dumps(bitmap, indent=2) + "\n")

    regenerate_shard_index(base_dir)
    regenerate_manifest(base_dir)


def _stage_tensor_names(
    layer: int, expert: int, projection: str, label: str,
    result: dict[str, Any],
) -> dict[str, torch.Tensor]:
    prefix = (
        f"model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.rank0")
    return {
        f"{prefix}.trellis_{label}": result["trellis"],
        f"{prefix}.suh_{label}": result["suh"],
        f"{prefix}.svh_{label}": result["svh"],
        f"{prefix}.scale_{label}": torch.tensor(
            result["scale"], dtype=torch.float32),
    }


def write_adapter_config(
    directory: Path,
    base_k: int,
    base_manifest_sha256: str,
    stages: list[dict[str, Any]],
    shards: list[str],
    tensor_count: int,
) -> None:
    """Write the explicit custom MSRT runtime contract."""
    config = {
        "schema": ADAPTER_CONFIG_SCHEMA,
        "base_k": base_k,
        "base_manifest_sha256": base_manifest_sha256,
        "format": "exl3-msrt-full-rank",
        "standard_lora_compatible": False,
        "runtime_operation": (
            "base_exl3_gemm + sum(stage_exl3_gemm / stage_scale)"),
        "codebook": "mcg",
        "mcg_multiplier": 0xCBAC1FED,
        "mcg_ownership": "adapter-config",
        "scale_shape": [],
        "stages": [
            {"label": stage["label"], "k": stage["k"],
             "experts": stage["experts"]}
            for stage in stages
        ],
        "shards": shards,
        "num_tensors": tensor_count,
        "tool_version": TOOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n")


def cmd_encode(args) -> int:
    """Encode a per-layer BF16 checkpoint into base EXL3 plus MSRT shards."""
    require_quant_dependencies()
    recipe = load_recipe(args.recipe)
    base_k = recipe["base_k"]
    stages = recipe["stages"]
    layers = recipe["moe_layers"]
    layer_shards = resolve_layer_shards(args.source, layers)
    out_dir = check_out_dir(
        args.out, source=args.source, policy=args.recipe)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise CartridgeError(f"--device {device}: CUDA is unavailable")
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = device.type
    print(f"Device: {device} ({device_name})", flush=True)
    print(
        f"Base K={base_k}, {len(stages)} stages, "
        f"MoE layers {layers[0]}-{layers[-1]} ({len(layers)} layers)",
        flush=True)

    staged = StagedOutput(out_dir, args.force)
    work = staged.begin()
    base_dir = work / "base"
    cartridge_dir = work / "cartridges"
    stage_shards: dict[str, list[str]] = {
        stage["label"]: [] for stage in stages}
    stage_tensor_counts = {stage["label"]: 0 for stage in stages}
    combined_shards: list[str] = []
    combined_tensor_count = 0
    experts_per_layer: dict[int, int] = {}
    total_mse = 0.0
    total_regularized_mse = 0.0
    expert_count = 0
    projection_count = 0

    try:
        copy_source_checkpoint(
            args.source, base_dir,
            {path.name for path in layer_shards.values()})
        from safetensors import safe_open

        with bootstrap_encoder(args.encoder_source) as encoder:
            ext, ghd, tcp, tcpi, qtf, cbs = encoder
            print(f"codebook_scale = {cbs}", flush=True)
            for layer in layers:
                source_shard = layer_shards[layer]
                print(f"\nEncoding layer {layer}...", flush=True)
                with safe_open(str(source_shard), framework="pt") as source_file:
                    keys = list(source_file.keys())
                    experts = inspect_source_layer(keys, layer)
                    experts_per_layer[layer] = len(experts)
                    base_tensors = load_preserved_tensors(source_file, keys)
                    stage_tensors: dict[str, dict[str, torch.Tensor]] = {
                        stage["label"]: {} for stage in stages}
                    layer_mses: list[float] = []

                    for expert in sorted(experts):
                        applicable = selected_stages(stages, expert)
                        for projection in PROJECTIONS:
                            source_name = experts[expert][projection]
                            source_weight = source_file.get_tensor(source_name)
                            if source_weight.ndim != 2:
                                raise CartridgeError(
                                    f"{source_name}: expected 2-D BF16 weight, "
                                    f"got {tuple(source_weight.shape)}")
                            weight = source_weight.float().T.contiguous().to(device)
                            result = encode_expert_msrt(
                                weight, base_k, applicable, device,
                                ghd, tcp, tcpi, qtf, cbs, ext)
                            prefix = (
                                f"model.layers.{layer}.mlp.experts.{expert}."
                                f"{projection}.rank0")
                            base_tensors[f"{prefix}.trellis"] = (
                                result["base"]["trellis"])
                            base_tensors[f"{prefix}.suh"] = result["base"]["suh"]
                            base_tensors[f"{prefix}.svh"] = result["base"]["svh"]
                            base_tensors[f"{prefix}.mcg"] = torch.tensor(
                                MCG_SENTINEL_SIGNED, dtype=torch.int32)
                            for label, stage_result in result["stages"].items():
                                stage_tensors[label].update(_stage_tensor_names(
                                    layer, expert, projection, label, stage_result))
                            total_mse += result["mse"]
                            total_regularized_mse += result["regularized_mse"]
                            layer_mses.append(result["mse"])
                            projection_count += 1
                            del weight, source_weight, result
                        expert_count += 1

                save_safetensors(base_tensors, base_dir / source_shard.name)
                combined: dict[str, torch.Tensor] = {}
                for stage in stages:
                    label = stage["label"]
                    tensors = stage_tensors[label]
                    if not tensors:
                        continue
                    relative = f"{label}/{source_shard.name}"
                    save_safetensors(tensors, cartridge_dir / relative)
                    stage_shards[label].append(relative)
                    stage_tensor_counts[label] += len(tensors)
                    combined.update(tensors)
                if combined:
                    relative = f"combined/{source_shard.name}"
                    save_safetensors(combined, cartridge_dir / relative)
                    combined_shards.append(relative)
                    combined_tensor_count += len(combined)
                average = sum(layer_mses) / len(layer_mses)
                print(
                    f"  Layer {layer}: avg original-space MSE={average:.4e}; "
                    f"{len(experts)} experts, {len(layer_mses)} projections",
                    flush=True)
                del base_tensors, stage_tensors, combined
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if expert_count == 0 or projection_count == 0:
            raise CartridgeError("encoding produced no experts or projections")
        write_base_metadata(base_dir, base_k, layers, experts_per_layer)
        base_manifest_sha256 = sha256_file(base_dir / "MANIFEST.sha256")
        for stage in stages:
            label = stage["label"]
            if not stage_shards[label]:
                raise CartridgeError(
                    f"stage {label!r} selected no experts in the source")
            write_adapter_config(
                cartridge_dir / label, base_k, base_manifest_sha256, [stage],
                stage_shards[label], stage_tensor_counts[label])
        write_adapter_config(
            cartridge_dir / "combined", base_k, base_manifest_sha256, stages,
            combined_shards, combined_tensor_count)

        summary = {
            "tool": TOOL_VERSION,
            "base_k": base_k,
            "effective_bpw_excluding_metadata": effective_bpw(
                recipe, max(experts_per_layer.values())),
            "stages": [
                {"k": stage["k"], "label": stage["label"],
                 "experts": stage["experts"]}
                for stage in stages
            ],
            "moe_layers": layers,
            "overall_mse_original_space": total_mse / projection_count,
            "overall_mse_regularized_space": (
                total_regularized_mse / projection_count),
            "n_experts_encoded": expert_count,
            "n_projections_encoded": projection_count,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (work / "encoding_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        staged.commit()
    except Exception:
        staged.abort()
        raise

    print(f"\nDone! Output: {out_dir}", flush=True)
    print(
        f"  Original-space MSE: "
        f"{total_mse / projection_count:.4e}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode BF16 → base K + cartridge adapters")
    enc.add_argument("--source", required=True, type=Path,
                     help="Source BF16 checkpoint directory")
    enc.add_argument("--recipe", required=True, type=Path,
                     help="Cartridge recipe JSON (fq-cartridge/1)")
    enc.add_argument("--out", required=True, type=Path,
                     help="Output directory (base/ and cartridges/ subdirs)")
    enc.add_argument("--encoder-source", required=True, type=Path,
                     help="Path to exllamav3 Python package")
    enc.add_argument("--device", default="cuda:0")
    enc.add_argument(
        "--force", action="store_true",
        help="Replace a previous fq_assemble_lora output carrying its sentinel")

    args = p.parse_args(argv)
    try:
        if args.command == "encode":
            return cmd_encode(args)
    except (AssemblyError, CartridgeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
