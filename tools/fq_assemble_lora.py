#!/usr/bin/env python3
"""fq_assemble_lora — Encode BF16 weights as MSRT EXL3 cartridges.

Given a BF16 checkpoint and an ``fq-cartridge/2`` recipe, this tool encodes a
directed acyclic graph of quantization tiers in one pass over the weights:

  * every declared **base** tier becomes a complete, loadable EXL3 checkpoint;
  * every declared **stage** is a rescaled trellis residual against the
    reconstruction of its named ``parent`` node, so one shared parent pass
    feeds many descendants instead of re-encoding the chain per product.

Cartridges are full-rank additive trellis weights, not PEFT/LoRA matrices.
Their execution pattern is LoRA-like (base GEMM plus correction GEMMs), but
standard vLLM/SGLang ``add_lora`` APIs cannot load them without an EXL3 MSRT
runtime implementation. The emitted ``fq-cartridge-adapter/2`` config records
that custom contract instead of claiming standard LoRA compatibility.

MSRT (Multi-Stage Rescaled Trellis) is described in:
  research/fungible-quant/poc/V50-LOW-BITRATE-MSRT.md
  research/fungible-quant/MSRT-CARTRIDGE-FEASIBILITY-AND-PLAN.md

Cartridge recipe format (fq-cartridge/2):

  {
    "schema": "fq-cartridge/2",
    "bases": [{"label": "k2", "k": 2}, {"label": "k3", "k": 3}],
    "stages": [
      {"label": "k2r1", "k": 1, "parent": "k2", "experts": "all"},
      {"label": "k2r1r1", "k": 1, "parent": "k2r1", "experts": "all"}
    ],
    "assemblies": [{"label": "k3like", "base": "k2", "chain": ["k2r1"]}],
    "moe_layers": [3, 4, 5, ...]
  }

Work is addressed as (layer, expert block), so an interrupted or preempted run
resumes at block granularity and independent workers can encode disjoint
blocks on separate GPUs with no shared state.

Usage:
  python tools/fq_assemble_lora.py plan     --source S --recipe R --out O
  python tools/fq_assemble_lora.py skeleton --source S --recipe R --out O
  python tools/fq_assemble_lora.py encode   --source S --recipe R --out O \\
      --encoder-source /opt/exllamav3-python/exllamav3 --devices cuda:0,cuda:1
  python tools/fq_assemble_lora.py finalize --source S --recipe R --out O
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

try:
    import torch
except ModuleNotFoundError:  # Base installs must still support --help.
    torch = None

sys.path.insert(0, str(Path(__file__).parent))
from fq_assemble import (
    AssemblyError,
    check_out_dir,
    regenerate_manifest,
    regenerate_shard_index,
    sha256_file,
)
from fq_fetch import Source
from fq_repack import (
    ATTESTATION_SCHEMA,
    PROJ_ORDER,
    Signer,
    load_source_shas,
    read_header,
)

# ── Constants ──────────────────────────────────────────────────────────────

TOOL_VERSION = "fq_assemble_lora/3"
CARTRIDGE_SCHEMA = "fq-cartridge/2"
ADAPTER_CONFIG_SCHEMA = "fq-cartridge-adapter/2"
ASSEMBLY_SCHEMA = "fq-cartridge-assembly/1"
PLAN_SCHEMA = "fq-msrt-plan/1"
BLOCK_SCHEMA = "fq-msrt-block/1"
SENTINEL = ".fq-msrt-encode.json"
SENTINEL_SCHEMA = "fq-msrt-sentinel/1"
HADAMARD_BLOCK = 128
MCG_MULTIPLIER = 0xCBAC1FED
MCG_SENTINEL_SIGNED = MCG_MULTIPLIER - (1 << 32)
RUNTIME_OPERATION = "base_exl3_gemm + sum(stage_exl3_gemm / stage_scale)"
DEFAULT_BLOCK_SIZE = 32
MAX_LAYER = 999
# One quantize_tiles launch runs one block per tile, and its work buffers are
# sized for 256. Measured on an RTX 5090 over the nine-node graph: 128 tiles
# per launch is the optimum at both model scales -- 2.29x faster than 16-row
# launches on 512-wide Fruit experts, and indistinguishable from them on
# 2048-wide GLM-5.2 experts, where 256 costs 10.7% because the K1 dynamic
# programming tables no longer fit the cache.
TILE_BATCH = 128
LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
BF16_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
LAYER_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.")
EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")
SAFETENSORS_DTYPE = {
    "float16": "F16", "bfloat16": "BF16", "float32": "F32", "float64": "F64",
    "int8": "I8", "int16": "I16", "int32": "I32", "int64": "I64",
    "uint8": "U8", "bool": "BOOL",
}
# Residual scale is stored per (expert, projection, stage); zero-RMS residuals
# are floored instead of special-cased so no stage ever ships a fabricated
# all-zero trellis, which would decode to a nonzero codebook value.
MIN_RESIDUAL_RMS = 1e-12
PROJECTIONS = tuple(sorted(PROJ_ORDER, key=PROJ_ORDER.get))


class CartridgeError(RuntimeError):
    """A recipe, source checkpoint, or encoded artifact is invalid."""


class Encoder(NamedTuple):
    """The trusted exllamav3 entry points used by every quantization pass."""

    ext: Any
    ghd: Any
    tcp: Any
    tcpi: Any
    qtf: Any
    cbs: float
    tile_batch: int = TILE_BATCH
    # sha256 of the exact encoder bundle that produced a shard's bytes; every
    # encode-of attestation pins it, so it is resolved where the bundle is
    # loaded rather than re-derived by each caller.
    identity: dict[str, Any] = {}


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


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── EXL3 Encoder Bootstrap ────────────────────────────────────────────────

@contextmanager
def bootstrap_encoder(encoder_source: str, tile_batch: int = TILE_BATCH):
    """Load trusted EXL3 encoder modules temporarily and restore sys.modules."""
    import importlib.util
    import types

    pkg_root = Path(encoder_source).expanduser().resolve()
    required = [
        pkg_root / "ext.py",
        pkg_root / "util" / "hadamard.py",
        pkg_root / "modules" / "quant" / "exl3_lib" / "quantize.py",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise CartridgeError(
            f"--encoder-source {pkg_root} is not an exllamav3 package "
            f"(missing {missing})")

    names = [
        "exllamav3",
        "exllamav3.util",
        "exllamav3.util.progress",
        "exllamav3.util.memory",
        "exllamav3.util.tensor",
        "exllamav3.modules",
        "exllamav3.modules.quant",
        "exllamav3.modules.quant.exl3_lib",
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
        yield Encoder(
            ext=ext,
            ghd=had_mod.get_hadamard_dt,
            tcp=quant_mod.tensor_core_perm,
            tcpi=quant_mod.tensor_core_perm_i,
            qtf=quant_mod.quantize_tiles,
            cbs=quant_mod.codebook_scale,
            tile_batch=tile_batch,
            identity=encoder_identity(pkg_root, required, ext),
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
        raise ValueError(
            f"{who}: expected a 2-D BF16 weight, got shape {tuple(w.shape)}")
    k, n = w.shape
    if k % HADAMARD_BLOCK or n % HADAMARD_BLOCK:
        raise ValueError(
            f"{who}: shape {(k, n)} must be divisible by Hadamard block "
            f"{HADAMARD_BLOCK} on both axes")
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
    if isinstance(K, bool) or not isinstance(K, int) or not 1 <= K <= 6:
        raise ValueError(
            f"{who} must be an integer in 1..6 (the runtime-supported trellis "
            f"bitrates), got {K!r}")
    return K


def quantize_trellis_packed(
    data: torch.Tensor, K: int, enc: Encoder, *, tile_batch: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a whole EXL3 matrix and return recon plus packed trellis.

    Tiles are submitted in groups of whole 16-row blocks, up to ``tile_batch``
    tiles per launch. A narrow matrix therefore stops launching 32-tile grids
    that leave most SMs idle, and a wide one keeps both its staging copy and
    the extension's dynamic-programming working set small. See ``TILE_BATCH``
    for the measured optimum; the reconstruction is bit-identical at every
    width, so this only trades launch count against working-set size.
    """
    _validate_k(K)
    if enc.ext is None or not callable(getattr(enc.ext, "pack_trellis", None)):
        raise RuntimeError(
            "the selected exllamav3 build lacks pack_trellis; refusing to "
            "write raw Viterbi indices as an EXL3 checkpoint")
    k, n = validate_quant_shape(data)
    tk, tn = k // 16, n // 16
    device = data.device
    perm = enc.tcp(device)
    perm_i = enc.tcpi(device)
    options = {"K": K, "mcg": True}
    blocks = max(1, min(tk, (tile_batch or enc.tile_batch) // tn))

    recon = torch.empty_like(data)
    raw = torch.empty((tk, tn, 256), dtype=torch.int16, device=device)
    for start in range(0, tk, blocks):
        stop = min(start + blocks, tk)
        span = stop - start
        tiles = (
            data[start * 16:stop * 16].view(span, 16, tn, 16)
            .permute(0, 2, 1, 3).reshape(span * tn, 256)[:, perm].contiguous()
        )
        quant_w, quant_idx = enc.qtf(tiles, options)
        raw[start:stop] = quant_idx.view(span, tn, 256)
        recon[start * 16:stop * 16] = (
            quant_w[:, perm_i].view(span, tn, 16, 16)
            .permute(0, 2, 1, 3).reshape(span * 16, n)
        )
    packed = torch.zeros((tk, tn, K * 16), dtype=torch.int16, device=device)
    enc.ext.pack_trellis(packed, raw, K)
    return recon, packed


def rescaled_trellis_quantize(
    parent_recon: torch.Tensor,
    residual: torch.Tensor,
    K_res: int,
    enc: Encoder,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Quantize one rescaled residual into runtime-compatible trellis form."""
    _validate_k(K_res, who="residual K")
    validate_quant_shape(residual, who="residual")
    residual_rms = residual.square().mean().sqrt().item()
    if not math.isfinite(residual_rms):
        raise ValueError("residual RMS is non-finite")
    scale = abs(float(enc.cbs)) / max(residual_rms, MIN_RESIDUAL_RMS)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"residual rescale factor is unusable: {scale}")
    recon_scaled, packed = quantize_trellis_packed(residual * scale, K_res, enc)
    return parent_recon + recon_scaled / scale, packed, scale


# ── Recipe ────────────────────────────────────────────────────────────────

def _validate_label(value: Any, *, who: str, seen: set[str]) -> str:
    if not isinstance(value, str) or not LABEL_RE.fullmatch(value):
        raise CartridgeError(
            f"{who}: label must match {LABEL_RE.pattern}, got {value!r}")
    if value in seen:
        raise CartridgeError(f"{who}: duplicate label {value!r}")
    seen.add(value)
    return value


def _validate_experts(value: Any, *, who: str) -> str | list[int]:
    if value == "all":
        return "all"
    if (not isinstance(value, list) or not value
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                   for v in value)
            or len(set(value)) != len(value)):
        raise CartridgeError(
            f"{who}: experts must be 'all' or unique non-negative IDs")
    return sorted(value)


def _covers(parent: str | list[int], child: str | list[int]) -> bool:
    """A residual only exists where the reconstruction it corrects exists."""
    if parent == "all":
        return True
    return child != "all" and set(child) <= set(parent)


def load_recipe(path: Path) -> dict[str, Any]:
    """Load and semantically validate one fq-cartridge/2 recipe."""
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

    layers = recipe.get("moe_layers")
    if (not isinstance(layers, list) or not layers
            or any(isinstance(v, bool) or not isinstance(v, int)
                   or not 0 <= v <= MAX_LAYER for v in layers)
            or len(set(layers)) != len(layers)):
        raise CartridgeError(
            f"moe_layers must be unique integers in 0..{MAX_LAYER}")
    recipe["moe_layers"] = sorted(layers)

    bases = recipe.get("bases")
    if not isinstance(bases, list) or not bases:
        raise CartridgeError("bases must be a non-empty list")
    labels: set[str] = set()
    experts_by_label: dict[str, str | list[int]] = {}
    for index, base in enumerate(bases):
        if not isinstance(base, dict):
            raise CartridgeError(f"base {index}: must be an object")
        label = _validate_label(base.get("label"), who=f"base {index}", seen=labels)
        _validate_k(base.get("k"), who=f"base {label!r} k")
        # A base tier is a complete checkpoint: it must cover every expert.
        if base.get("experts", "all") != "all":
            raise CartridgeError(
                f"base {label!r}: bases cover all experts; move a partial "
                f"tier into stages")
        base["experts"] = "all"
        experts_by_label[label] = "all"

    stages = recipe.get("stages")
    if not isinstance(stages, list):
        raise CartridgeError("stages must be a list")
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise CartridgeError(f"stage {index}: must be an object")
        label = _validate_label(stage.get("label"), who=f"stage {index}", seen=labels)
        _validate_k(stage.get("k"), who=f"stage {label!r} k")
        parent = stage.get("parent")
        if parent not in experts_by_label:
            raise CartridgeError(
                f"stage {label!r}: parent {parent!r} is not a base or an "
                f"earlier stage; stages must be listed parent-first")
        experts = _validate_experts(stage.get("experts"), who=f"stage {label!r}")
        if not _covers(experts_by_label[parent], experts):
            raise CartridgeError(
                f"stage {label!r}: experts must be a subset of parent "
                f"{parent!r}; a residual cannot correct a tier that is absent")
        stage["experts"] = experts
        experts_by_label[label] = experts

    assemblies = recipe.get("assemblies", [])
    if not isinstance(assemblies, list):
        raise CartridgeError("assemblies must be a list")
    stage_by_label = {stage["label"]: stage for stage in stages}
    base_labels = {base["label"] for base in bases}
    assembly_labels: set[str] = set()
    for index, assembly in enumerate(assemblies):
        if not isinstance(assembly, dict):
            raise CartridgeError(f"assembly {index}: must be an object")
        label = _validate_label(
            assembly.get("label"), who=f"assembly {index}", seen=assembly_labels)
        base = assembly.get("base")
        if base not in base_labels:
            raise CartridgeError(f"assembly {label!r}: base {base!r} is not declared")
        chain = assembly.get("chain", [])
        if (not isinstance(chain, list)
                or len(set(chain)) != len(chain)
                or any(name not in stage_by_label for name in chain)):
            raise CartridgeError(
                f"assembly {label!r}: chain must be unique declared stage labels")
        expected = base
        for name in chain:
            if stage_by_label[name]["parent"] != expected:
                raise CartridgeError(
                    f"assembly {label!r}: stage {name!r} corrects "
                    f"{stage_by_label[name]['parent']!r}, not {expected!r}; a "
                    f"chain must be one path through the recipe graph")
            expected = name
        assembly["chain"] = list(chain)
    recipe["assemblies"] = assemblies
    return recipe


def node_index(recipe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """label -> node, for bases and stages alike."""
    return {node["label"]: node
            for node in [*recipe["bases"], *recipe["stages"]]}


def stages_for_expert(
    stages: list[dict[str, Any]], expert_id: int
) -> list[dict[str, Any]]:
    """Resolve stage applicability before residual chaining begins."""
    return [stage for stage in stages
            if stage["experts"] == "all" or expert_id in stage["experts"]]


def selected_count(node: dict[str, Any], expert_count: int) -> int:
    return expert_count if node["experts"] == "all" else len(node["experts"])


def assembly_bpw(
    recipe: dict[str, Any], assembly: dict[str, Any], expert_count: int
) -> float:
    """Nominal weight bits per weight for one product, excluding metadata."""
    if expert_count <= 0:
        raise ValueError("expert_count must be positive")
    nodes = node_index(recipe)
    total = nodes[assembly["base"]]["k"] * expert_count
    for label in assembly["chain"]:
        node = nodes[label]
        total += node["k"] * selected_count(node, expert_count)
    return total / expert_count


def encoded_bits_per_weight(recipe: dict[str, Any], expert_count: int) -> float:
    """Bits emitted per source weight for the whole recipe graph."""
    nodes = node_index(recipe)
    return sum(node["k"] * selected_count(node, expert_count)
               for node in nodes.values()) / expert_count


# ── Encoding Pipeline ─────────────────────────────────────────────────────

def encode_matrix_dag(
    w_bf16: torch.Tensor,
    bases: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    device: torch.device,
    enc: Encoder,
) -> dict[str, dict[str, Any]]:
    """Encode one matrix into every base tier and applicable residual stage.

    ``stages`` must be topologically ordered (parents first) and already
    filtered to the stages that apply to this expert. Each parent
    reconstruction is computed once and released as soon as its last child has
    consumed it, so a wide graph costs one quantization pass per node instead
    of one pass per product chain.
    """
    if not bases:
        raise CartridgeError("encoding requires at least one base tier")
    w_reg, suh, svh = regularize_with_vectors(w_bf16, device, enc.ghd, enc.cbs)
    remaining = Counter(stage["parent"] for stage in stages)
    recon: dict[str, torch.Tensor] = {}
    nodes: dict[str, dict[str, Any]] = {}

    def record(label: str, packed: torch.Tensor, current: torch.Tensor,
               scale: float | None) -> None:
        restored = inverse_regularize(current, suh, svh, device, enc.ghd)
        nodes[label] = {
            "trellis": packed.cpu(),
            "suh": suh.cpu(),
            "svh": svh.cpu(),
            "scale": scale,
            "mse": (w_bf16.float() - restored).square().mean().item(),
            "regularized_mse": (w_reg - current).square().mean().item(),
        }

    for base in bases:
        label = base["label"]
        current, packed = quantize_trellis_packed(w_reg, base["k"], enc)
        record(label, packed, current, None)
        del packed
        if remaining[label]:
            recon[label] = current
        else:
            del current

    for stage in stages:
        parent = stage["parent"]
        if parent not in recon:
            raise CartridgeError(
                f"stage {stage['label']!r}: parent {parent!r} was not encoded")
        parent_recon = recon[parent]
        current, packed, scale = rescaled_trellis_quantize(
            parent_recon, w_reg - parent_recon, stage["k"], enc)
        record(stage["label"], packed, current, scale)
        del packed
        remaining[parent] -= 1
        if remaining[parent] <= 0:
            del recon[parent]
        if remaining[stage["label"]]:
            recon[stage["label"]] = current
        else:
            del current
    return nodes


# ── Source Checkpoint ─────────────────────────────────────────────────────

class SourceCheckpoint:
    """Random-access reader for a BF16 checkpoint in either supported layout.

    ``per-layer``  one ``model-layer-NNN.safetensors`` per layer.
    ``indexed``    standard Hugging Face shards plus
                   ``model.safetensors.index.json``.

    Only the requested tensors are materialized, so a 1.5 TB checkpoint is
    encoded shard-by-shard without ever loading a whole shard, and a small
    handle cache keeps repeated expert reads on one already-open shard.
    """

    PER_LAYER = "per-layer"
    INDEXED = "indexed"

    def __init__(self, root: Path, *, max_open: int = 6):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise CartridgeError(f"--source {self.root} is not a directory")
        if not (self.root / "config.json").is_file():
            raise CartridgeError(f"--source {self.root} has no config.json")
        self.max_open = max_open
        self._open: OrderedDict[str, Any] = OrderedDict()
        self._layer_keys: dict[int, dict[str, str]] = {}
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            try:
                weight_map = json.loads(index.read_text())["weight_map"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise CartridgeError(f"{index}: unusable weight map ({exc})") from exc
            if not isinstance(weight_map, dict) or not weight_map:
                raise CartridgeError(f"{index}: empty weight map")
            for key, shard in weight_map.items():
                # A source index is untrusted input: it names files this tool
                # opens. Anything but a plain relative name could read outside
                # the checkpoint.
                if not isinstance(key, str) or not isinstance(shard, str):
                    raise CartridgeError(f"{index}: non-string weight map entry")
                relative = PurePosixPath(shard)
                if (relative.is_absolute() or ".." in relative.parts
                        or not shard.endswith(".safetensors")):
                    raise CartridgeError(
                        f"{index}: unsafe shard path {shard!r}")
            missing = sorted({
                shard for shard in set(weight_map.values())
                if not (self.root / shard).is_file()
            })
            self.layout = self.INDEXED
            self.weight_map: dict[str, str] = weight_map
            self.absent_shards = missing
        else:
            shards = sorted(self.root.glob("model-layer-*.safetensors"))
            if not shards:
                raise CartridgeError(
                    f"--source {self.root} has neither "
                    f"model.safetensors.index.json nor per-layer shards")
            self.layout = self.PER_LAYER
            self.weight_map = {}
            self.absent_shards = []

    @property
    def topology_id(self) -> str:
        """A fingerprint of the checkpoint's tensor topology.

        Informational only. It is deliberately independent of which shards are
        present on this disk, because the campaign stages source bytes in
        windows and deletes them after each one: anything that changed as
        shards came and went could not recognise the same source across a
        resumed run. Artifact identity is the immutable base revision, not
        this digest.
        """
        cached = getattr(self, "_topology_id", None)
        if cached is None:
            if self.layout == self.INDEXED:
                rows = [f"{key}\t{shard}"
                        for key, shard in sorted(self.weight_map.items())]
            else:
                rows = sorted(
                    path.name for path in
                    self.root.glob("model-layer-*.safetensors"))
            digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
            cached = self._topology_id = digest
        return cached

    # -- handles ---------------------------------------------------------
    def _handle(self, shard: str):
        handle = self._open.get(shard)
        if handle is None:
            from safetensors import safe_open

            path = self.root / shard
            if not path.is_file():
                raise CartridgeError(f"source shard {shard} is missing")
            handle = safe_open(str(path), framework="pt")
            handle.__enter__()
            self._open[shard] = handle
            while len(self._open) > self.max_open:
                _, evicted = self._open.popitem(last=False)
                evicted.__exit__(None, None, None)
        else:
            self._open.move_to_end(shard)
        return handle

    def keys_in_shard(self, shard: str) -> list[str]:
        return list(self._handle(shard).keys())

    def tensor(self, shard: str, key: str):
        return self._handle(shard).get_tensor(key)

    def close(self) -> None:
        for handle in self._open.values():
            handle.__exit__(None, None, None)
        self._open.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    # -- topology --------------------------------------------------------
    def layer_shard(self, layer: int) -> str:
        return resolve_layer_shards(self.root, [layer])[layer].name

    def layer_keys(self, layer: int) -> dict[str, str]:
        """key -> shard for every tensor of one layer."""
        cached = self._layer_keys.get(layer)
        if cached is not None:
            return cached
        if self.layout == self.INDEXED:
            prefix = f"model.layers.{layer}."
            keys = {key: shard for key, shard in self.weight_map.items()
                    if key.startswith(prefix)}
        else:
            shard = self.layer_shard(layer)
            keys = {key: shard for key in self.keys_in_shard(shard)}
            foreign = sorted(
                key for key in keys
                if (match := LAYER_KEY_RE.match(key))
                and int(match.group(1)) != layer)
            if foreign:
                raise CartridgeError(
                    f"layer {layer} shard {shard} also contains tensors for "
                    f"other layers, e.g. {foreign[:3]}")
        if not keys:
            raise CartridgeError(f"layer {layer}: no tensors found in {self.root}")
        self._layer_keys[layer] = keys
        return keys

    @property
    def declared_experts(self) -> int | None:
        """``n_routed_experts`` from the source config, when it states one."""
        cached = getattr(self, "_declared_experts", "unset")
        if cached == "unset":
            try:
                config = json.loads((self.root / "config.json").read_text())
                value = config.get("n_routed_experts")
            except (OSError, json.JSONDecodeError, AttributeError):
                value = None
            cached = value if isinstance(value, int) and value > 0 else None
            self._declared_experts = cached
        return cached

    def experts(self, layer: int) -> dict[int, dict[str, str]]:
        """expert -> projection -> key, validated for exact coverage."""
        return inspect_source_layer(
            list(self.layer_keys(layer)), layer,
            declared=self.declared_experts)

    def shard_bytes(self, shards: set[str]) -> dict[str, int]:
        out = {}
        for shard in sorted(shards):
            path = self.root / shard
            out[shard] = path.stat().st_size if path.is_file() else 0
        return out


def resolve_layer_shards(source: Path, layers: list[int]) -> dict[int, Path]:
    """Resolve the one-safetensors-file-per-layer layout, 3- or 4-digit."""
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise CartridgeError(f"{root} is not a directory")
    resolved: dict[int, Path] = {}
    for layer in layers:
        found = [
            path for path in
            (root / f"model-layer-{layer:0{width}d}.safetensors"
             for width in (3, 4))
            if path.is_file()
        ]
        if len(found) != 1:
            detail = (f"ambiguous candidates {[p.name for p in found]}" if found
                      else "standard Hugging Face indexed shards are supported "
                           "by the encoder, but this caller needs per-layer shards")
            raise CartridgeError(
                f"layer {layer}: shard not resolved in {root}; {detail}")
        resolved[layer] = found[0]
    return resolved


def inspect_source_layer(
    keys: list[str], layer: int, *, declared: int | None = None
) -> dict[int, dict[str, str]]:
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
    # tier_bitmap.json and hybrid_tr3_tail.experts_per_layer describe a layer
    # by expert *count*, and the loader indexes rows positionally, so a sparse
    # or shifted id set would misstate what the checkpoint contains.
    ids = sorted(experts)
    if ids != list(range(len(ids))):
        raise CartridgeError(
            f"layer {layer}: routed expert ids are not dense 0..{len(ids) - 1} "
            f"(got {ids[:4]}...{ids[-1]}); positional loader metadata cannot "
            f"describe that")
    if declared is not None and len(ids) != declared:
        raise CartridgeError(
            f"layer {layer}: found {len(ids)} routed experts but config.json "
            f"declares n_routed_experts={declared}")
    return experts


def expert_blocks(expert_ids: list[int], block_size: int) -> list[list[int]]:
    """Split one layer's experts into fixed-size, order-stable work blocks."""
    if block_size < 1:
        raise CartridgeError("--block-size must be positive")
    ordered = sorted(expert_ids)
    return [ordered[i:i + block_size]
            for i in range(0, len(ordered), block_size)]


# ── Output Layout ─────────────────────────────────────────────────────────

def block_name(layer: int, block: int) -> str:
    """Flat, loader-friendly shard name for one (layer, expert block)."""
    if not 0 <= layer <= MAX_LAYER:
        raise CartridgeError(f"layer {layer} is out of range")
    return f"model-layer-{layer:03d}-b{block:03d}.safetensors"


def node_dir(out: Path, recipe: dict[str, Any], label: str) -> Path:
    kind = "base" if label in {b["label"] for b in recipe["bases"]} else "stages"
    return out / kind / label


def digest_path(directory: Path, name: str) -> Path:
    """The commit marker for one shard, colocated with the shard itself."""
    return directory / "digests" / f"{name}.sha256"


def save_shard_tensors(
    tensors: dict[str, torch.Tensor], directory: Path, name: str,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Validate and write one safetensors shard through a temporary name.

    Non-finite payloads are refused before anything is renamed into place, so
    a readable shard is always a usable shard.
    """
    from safetensors.torch import save_file

    if not tensors:
        raise CartridgeError(f"refusing to write empty safetensors {name}")
    for key, tensor in tensors.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise CartridgeError(f"{name}: tensor {key} is not finite")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    staging = directory / f".{name}.partial"
    save_file(tensors, str(staging), metadata=metadata)
    os.replace(staging, target)
    return target


# ── Provenance ────────────────────────────────────────────────────────────

def attestation_path(directory: Path, name: str) -> Path:
    """Provenance lives *inside* the node directory it describes.

    Publishing a product means uploading its node subtree, so an attestation
    kept in a parallel tree could be silently left behind. Colocating it also
    puts it under the base checkpoint's MANIFEST.sha256, which binds provenance
    to the checkpoint digest a cartridge names.
    """
    return directory / "attestations" / f"{name}.jsonl"


def infer_source_identity(root: Path) -> tuple[str | None, str | None]:
    """Recover (repo, commit) from a Hugging Face snapshot directory layout.

    ``.../models--org--name/snapshots/<40-hex commit>/`` is how ``hf download``
    lays a checkpoint out, and the directory name *is* the immutable revision
    an ``encode-of`` attestation must pin. Anything else returns None and the
    operator must state the identity explicitly.
    """
    revision = root.name if Source.is_immutable_commit(root.name) else None
    repo = None
    for parent in root.parents:
        if parent.name.startswith("models--"):
            repo = parent.name[len("models--"):].replace("--", "/")
            break
    return repo, revision


def encoder_identity(
    root: Path, modules: list[Path], ext: Any
) -> dict[str, Any]:
    """Digest the exact encoder bundle that produced a shard's bytes.

    Both the Python modules this tool executes and the compiled extension that
    runs the trellis search are hashed: a rebuilt ``.so`` changes the bytes it
    emits, so an attestation that pinned only the Python side would claim
    reproducibility it cannot deliver.
    """
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in modules
    }
    binary = getattr(ext, "__file__", None)
    if not binary or not Path(binary).is_file():
        raise CartridgeError(
            "cannot locate the compiled exllamav3 extension; an encode-of "
            "attestation must pin the binary that produced the bytes")
    files[Path(binary).name] = sha256_file(Path(binary))
    return {
        "encoder": "exllamav3",
        "encoder_bundle": str(root),
        "encoder_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "encoder_files": files,
    }


def determinism_scope(device: torch.device, tile_batch: int) -> dict[str, Any]:
    """The boundary inside which re-encoding is expected to be byte-identical."""
    scope = {
        "device_type": device.type,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "tile_batch": tile_batch,
        "note": ("Trellis search is deterministic for a fixed encoder build, "
                 "GPU architecture and library stack. Across stacks the "
                 "honest claim is measured equivalence, not byte identity."),
    }
    if device.type == "cuda":
        scope["gpu"] = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        scope["compute_capability"] = f"{major}.{minor}"
    return scope


def capture_evidence(
    recipe_sha: str, base_revision: str, cbs: float
) -> tuple[str, dict[str, Any]]:
    """Fingerprint the encode inputs of a data-free quantizer.

    ``fq-attestation/1`` requires ``materials.capture_fingerprint`` to be a
    sha256 because every previous producer in this family consumed a
    calibration capture. MSRT trellis encoding consumes none: the inputs are
    the recipe graph, the source topology and the fixed regularization seed.
    The fingerprint therefore covers *that* descriptor, and the descriptor
    ships beside it under ``capture_descriptor`` with an explicit
    ``capture_kind`` so nobody can read it as a corpus that does not exist.
    """
    descriptor = {
        "capture_kind": "data-free-trellis",
        "recipe_sha256": recipe_sha,
        "base_revision": base_revision,
        "regularize_seed": 0,
        "hadamard_block": HADAMARD_BLOCK,
        "codebook": "mcg",
        "codebook_scale": float(cbs),
        "note": ("No calibration corpus or Hessian is used; these are the "
                 "deterministic encode inputs instead."),
    }
    fingerprint = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return fingerprint, descriptor


def load_digest_map(path: Path | None) -> dict[str, str]:
    """``{filename: sha256}`` supplied by the operator, e.g. Hugging Face oids.

    A safetensors file on the hub is LFS-backed, so its ``lfs.oid`` *is* its
    sha256 and can be read from the API without downloading anything:

        curl -s 'https://huggingface.co/api/models/<repo>/tree/<rev>?recursive=1'
    """
    if path is None:
        return {}
    try:
        raw = json.loads(Path(path).expanduser().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CartridgeError(f"{path}: unreadable digest map ({exc})") from exc
    if isinstance(raw, list):  # an HF tree listing, passed through untouched
        raw = {entry["path"]: (entry.get("lfs") or {}).get("oid")
               for entry in raw if isinstance(entry, dict) and entry.get("path")}
    if not isinstance(raw, dict):
        raise CartridgeError(f"{path}: expected an object of file -> sha256")
    digests = {name: value for name, value in raw.items()
               if isinstance(value, str) and len(value) == 64}
    if not digests:
        raise CartridgeError(f"{path}: no sha256 digests found")
    return digests


def source_digest_resolver(
    source: SourceCheckpoint, out: Path, supplied: dict[str, str] | None = None
):
    """Digest one source shard, preferring evidence that already exists.

    ``repack-of`` pins the file its bytes were copied out of, so the digest is
    required. Hashing hundreds of gigabytes of source shards to attest 38 GB of
    skeleton is avoidable work: an operator-supplied map (Hugging Face LFS oids)
    is used first, then a ``MANIFEST.sha256`` the source publishes about itself,
    and anything computed locally is cached in the campaign so a resumed or
    re-windowed run never hashes the same shard twice.
    """
    published = {**load_source_shas(source.root), **(supplied or {})}
    cache_path = out / "source-digests.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        cache = {}

    def resolve(name: str) -> str | None:
        if name in published:
            return published[name]
        if name in cache:
            return cache[name]
        path = source.root / name
        if not path.is_file():
            return None
        cache[name] = sha256_file(path)
        tmp = cache_path.with_name(f".source-digests.{os.getpid()}.json")
        tmp.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, cache_path)
        return cache[name]

    return resolve


def build_provenance(
    *, predicate: str, recipe_sha: str, source: SourceCheckpoint,
    base_model: str, base_revision: str, signer: Any,
    source_digest: Any = None,
    encoder: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    capture: tuple[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Everything a run needs to attest one artifact, computed once."""
    if predicate not in {"encode-of", "repack-of"}:
        raise CartridgeError(f"unsupported attestation predicate {predicate!r}")
    if not Source.is_immutable_commit(base_revision):
        raise CartridgeError(
            f"--base-revision {base_revision!r} is not an immutable commit; "
            f"an attestation that pins a moving reference proves nothing")
    if not base_model:
        raise CartridgeError("--base-model is required to attest an artifact")
    return {
        "predicate": predicate,
        "recipe_sha256": recipe_sha,
        "source": source,
        "base_model": base_model,
        "base_revision": base_revision,
        "signer": signer,
        "source_digest": source_digest,
        "encoder": encoder,
        "scope": scope,
        "capture": capture,
    }


def write_attestation(
    directory: Path, name: str, *, sha: str, size: int, body_offset: int,
    spans: dict[str, tuple[int, int]], digests: dict[str, str],
    group_kind: str, provenance: dict[str, Any],
) -> Path:
    """Sign and write one ``fq-attestation/1`` line for one shard."""
    predicate = provenance["predicate"]
    payload: dict[str, Any] = {
        "schema": ATTESTATION_SCHEMA,
        "predicate": predicate,
        "fragment": {"file": name, "sha256": sha, "size": size,
                     "body_offset": body_offset},
        "created_utc": now_utc(),
        "layout": "rank-sliced",
        "base_model": provenance["base_model"],
        "calibration_corpus_sha256": None,
        "tool": {"name": "fq_assemble_lora", "version": TOOL_VERSION},
        "recipe_sha256": provenance["recipe_sha256"],
        "base_revision": provenance["base_revision"],
    }
    if group_kind == "expert":
        payload["expert_sha256"] = digests
        payload["experts"] = {key: list(span) for key, span in spans.items()}
    else:
        # No routed experts in this fragment. fq-attestation/1 requires
        # expert_sha256, so a non-expert fragment declares its kind and carries
        # a per-tensor map instead -- the same shape fq_prime's shared-profile
        # lines use, and the shape fq_verify normalizes before schema checking.
        payload["kind"] = "skeleton"
        payload["tensor_sha256"] = digests
        payload["tensors"] = {key: list(span) for key, span in spans.items()}
    payload.update(provenance.get("extra") or {})
    if predicate == "encode-of":
        fingerprint, descriptor = provenance["capture"]
        payload["materials"] = {
            "base_model": provenance["base_model"],
            "base_revision": provenance["base_revision"],
            "capture_fingerprint": fingerprint,
            **provenance["encoder"],
        }
        payload["capture_descriptor"] = descriptor
        payload["determinism_scope"] = provenance["scope"]
    else:
        payload["materials"] = {
            "repo": provenance["base_model"],
            "revision": provenance["base_revision"],
            "file": name,
            "file_sha256": provenance["source_digest"](name),
        }
    line = provenance["signer"].sign_line(payload)
    path = attestation_path(directory, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(line + "\n")
    os.replace(tmp, path)
    return path


def read_signed_line(path: Path) -> dict[str, Any]:
    """Load one detached-signature envelope and verify it against its own key.

    This proves the payload was signed by the key it names and has not been
    edited since. It says nothing about that key's *authority*, which comes
    from keys/FINGERPRINTS out of band -- callers that care must compare
    ``_keyid`` against a pinned value.
    """
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey

    if not path.is_file():
        raise CartridgeError(f"no signed line at {path}")
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise CartridgeError(f"{path}: expected exactly one signed line")
    try:
        envelope = json.loads(lines[0])
        raw = base64.b64decode(envelope["payload"], validate=True)
        signature = base64.b64decode(envelope["signature"], validate=True)
        VerifyKey(bytes.fromhex(envelope["keyid"])).verify(raw, signature)
        payload = json.loads(raw)
    except (KeyError, TypeError, ValueError, BadSignatureError,
            json.JSONDecodeError, binascii.Error) as exc:
        raise CartridgeError(f"{path}: unusable attestation ({exc})") from exc
    payload["_keyid"] = envelope["keyid"]
    return payload


def read_attestation(directory: Path, name: str) -> dict[str, Any]:
    """Load and self-verify the attestation line for one shard."""
    return read_signed_line(attestation_path(directory, name))


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Raw little-endian payload of one tensor, dtype-agnostic."""
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).numpy().tobytes()


def write_grouped_shard(
    groups: list[tuple[str, list[tuple[str, torch.Tensor]]]],
    directory: Path, name: str, metadata: dict[str, str],
) -> tuple[Path, str, int, dict[str, tuple[int, int]], dict[str, str]]:
    """Serialize one safetensors shard with a layout we choose, hashing as we go.

    ``safetensors.torch.save_file`` orders the payload by dtype, which
    interleaves the four components of different experts and leaves no expert
    occupying one byte range. A signed ``expert_sha256`` has to be the digest of
    exactly the bytes a consumer would range-read, so this writer emits groups
    in the order given -- expert by expert -- and returns each group's span.

    Hashing happens during the single write pass: no re-read, and the digest
    covers the bytes that actually landed on disk.
    """
    seen: set[str] = set()
    entries: dict[str, dict[str, Any]] = {}
    payload: list[tuple[str, list[bytes]]] = []
    offset = 0
    spans: dict[str, tuple[int, int]] = {}
    for group, tensors in groups:
        if group in spans:
            raise CartridgeError(f"{name}: duplicate group {group!r}")
        if not tensors:
            raise CartridgeError(f"{name}: group {group!r} has no tensors")
        start = offset
        chunks: list[bytes] = []
        for key, tensor in tensors:
            if key in seen:
                raise CartridgeError(f"{name}: duplicate tensor {key!r}")
            seen.add(key)
            if str(tensor.dtype).removeprefix("torch.") not in SAFETENSORS_DTYPE:
                raise CartridgeError(f"{name}: {key} has unsupported dtype "
                                     f"{tensor.dtype}")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise CartridgeError(f"{name}: tensor {key} is not finite")
            raw = tensor_bytes(tensor)
            entries[key] = {
                "dtype": SAFETENSORS_DTYPE[
                    str(tensor.dtype).removeprefix("torch.")],
                "shape": list(tensor.shape),
                "data_offsets": [offset, offset + len(raw)],
            }
            chunks.append(raw)
            offset += len(raw)
        spans[group] = (start, offset)
        payload.append((group, chunks))
    if not entries:
        raise CartridgeError(f"refusing to write empty safetensors {name}")

    header = json.dumps({"__metadata__": dict(metadata), **entries},
                        separators=(",", ":")).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    prefix = struct.pack("<Q", len(header)) + header
    body_offset = len(prefix)

    whole = hashlib.sha256()
    whole.update(prefix)
    digests: dict[str, str] = {}
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    staging = directory / f".{name}.{os.getpid()}.partial"
    with open(staging, "wb") as handle:
        handle.write(prefix)
        for group, chunks in payload:
            span = hashlib.sha256()
            for raw in chunks:
                handle.write(raw)
                whole.update(raw)
                span.update(raw)
            digests[group] = span.hexdigest()
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, target)
    return target, whole.hexdigest(), body_offset, spans, digests


def verify_shard(
    path: Path, *, group_kind: str
) -> tuple[str, int, dict[str, str]]:
    """Re-derive one shard's digests from the bytes on disk.

    ``write_grouped_shard`` hashes while writing, which is what makes the
    encode path cheap, but a digest that was only ever computed in the same
    process that wrote it proves nothing about the file that survived. Finalize
    re-reads every fragment through this function before publishing a manifest
    or an assembly plan.
    """
    header, body = read_header(path)
    header.pop("__metadata__", None)
    ordered = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    runs: list[tuple[str, int, int]] = []
    for name, meta in ordered:
        lo, hi = meta["data_offsets"]
        if group_kind == "expert":
            match = EXPERT_ID_RE.search(name)
            if match is None:
                raise CartridgeError(f"{path}: {name} is not an expert tensor")
            key = match.group(1)
        else:
            key = name
        if runs and runs[-1][0] == key:
            if runs[-1][2] != lo:
                raise CartridgeError(
                    f"{path}: {group_kind} {key} is not one byte range; a "
                    f"ranged read of it could not be verified")
            runs[-1] = (key, runs[-1][1], hi)
        else:
            if any(existing == key for existing, _, _ in runs):
                raise CartridgeError(
                    f"{path}: {group_kind} {key} appears in two byte ranges")
            runs.append((key, lo, hi))

    whole = hashlib.sha256()
    digests: dict[str, str] = {}
    with open(path, "rb") as handle:
        whole.update(handle.read(body))
        position = 0
        for key, lo, hi in runs:
            if lo != position:
                whole.update(handle.read(lo - position))
            span = hashlib.sha256()
            remaining = hi - lo
            while remaining:
                chunk = handle.read(min(remaining, 1 << 22))
                if not chunk:
                    raise CartridgeError(f"{path}: truncated payload")
                whole.update(chunk)
                span.update(chunk)
                remaining -= len(chunk)
            position = hi
            digests[key] = span.hexdigest()
        while True:
            chunk = handle.read(1 << 22)
            if not chunk:
                break
            whole.update(chunk)
    return whole.hexdigest(), body, digests


def write_shard(
    groups: list[tuple[str, list[tuple[str, torch.Tensor]]]],
    directory: Path, name: str, metadata: dict[str, str], *,
    provenance: dict[str, Any] | None = None, group_kind: str = "expert",
) -> str:
    """Write one campaign shard, its digest and its signed attestation.

    The payload lands under a temporary name and is renamed into place; the
    attestation is written next, and the digest sidecar last. A committed
    digest therefore means the shard is complete *and* attested, which is
    exactly what ``resume`` needs and what the final MANIFEST is built from
    without re-reading terabytes.
    """
    # Retract the old commit markers BEFORE the payload changes. A crash
    # between writing new bytes and re-committing would otherwise leave a
    # digest and a signed attestation that agree with each other and not with
    # the shard, which resume would accept as finished work.
    for stale in (digest_path(directory, name), attestation_path(directory, name)):
        stale.unlink(missing_ok=True)
    target, sha, body, spans, digests = write_grouped_shard(
        groups, directory, name, metadata)
    if provenance is not None:
        write_attestation(
            directory, name, sha=sha, size=target.stat().st_size,
            body_offset=body, spans=spans, digests=digests,
            group_kind=group_kind, provenance=provenance)
    sidecar = digest_path(directory, name)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    tmp.write_text(f"{sha}  {name}\n")
    os.replace(tmp, sidecar)
    return sha


def read_digest(directory: Path, name: str) -> str | None:
    sidecar = digest_path(directory, name)
    if not sidecar.is_file():
        return None
    line = sidecar.read_text().strip()
    sha = line.split(maxsplit=1)[0] if line else ""
    return sha if len(sha) == 64 else None


def shard_is_complete(
    directory: Path, name: str, expect: dict[str, str],
) -> bool:
    """A shard counts as done when its digest is committed and it matches.

    The recorded metadata binds the shard to the recipe, the encoder version
    and the exact expert list, so a resumed run can never silently mix work
    from a different graph or a different block layout.
    """
    target = directory / name
    if not target.is_file() or read_digest(directory, name) is None:
        return False
    try:
        header, _ = read_header(target)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    meta = header.get("__metadata__") or {}
    for key, value in expect.items():
        if meta.get(key) != value:
            raise CartridgeError(
                f"{target}: {key} is {meta.get(key)!r}, expected {value!r}. "
                f"This output directory was produced by a different recipe or "
                f"block layout; use a fresh --out.")
    return True


def prepare_out_dir(
    out: Path, *, recipe_path: Path, recipe_sha: str, source: SourceCheckpoint,
    base_model: str, base_revision: str, block_size: int, force: bool,
) -> Path:
    """Create or re-attach to a campaign output directory.

    The sentinel binds the recipe, the block layout and the *immutable source
    revision*. It deliberately does not bind which source shards are on this
    disk: the campaign stages bytes in windows and deletes them, so a
    presence-sensitive identity would reject its own next window.

    Nothing is ever deleted: a campaign directory can hold a terabyte of
    finished work, so a mismatch is an error rather than a purge.
    """
    resolved = check_out_dir(out, source=source.root, policy=recipe_path)
    resolved.mkdir(parents=True, exist_ok=True)
    marker = resolved / SENTINEL
    expected = {
        "schema": SENTINEL_SCHEMA,
        "tool": TOOL_VERSION,
        "recipe_sha256": recipe_sha,
        "base_model": base_model,
        "base_revision": base_revision,
        "block_size": block_size,
    }
    if marker.is_file():
        try:
            found = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CartridgeError(f"{marker}: unreadable ({exc})") from exc
        drift = {key: (found.get(key), value) for key, value in expected.items()
                 if found.get(key) != value}
        if drift:
            # --force re-encodes blocks of *this* campaign; it never converts a
            # directory to a different recipe, source or block layout. Changing
            # the block size would leave the previous layout's shards behind,
            # and both would end up in the regenerated index.
            raise CartridgeError(
                f"{marker} does not match this run: {drift}. This campaign "
                f"directory belongs to a different recipe, source revision or "
                f"block layout: encode into a fresh --out.")
        if not drift:
            # Nothing to record: leave the marker alone so parallel workers
            # never race each other rewriting the same path.
            return resolved
    payload = dict(expected)
    payload.update({"source": str(source.root), "layout": source.layout,
                    "topology_id": source.topology_id,
                    "updated_utc": now_utc()})
    tmp = marker.with_name(f".{SENTINEL}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, marker)
    return resolved


# ── Tensor Naming ─────────────────────────────────────────────────────────

def base_tensor_names(
    layer: int, expert: int, projection: str, node: dict[str, Any]
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank0"
    return {
        f"{prefix}.trellis": node["trellis"],
        f"{prefix}.suh": node["suh"],
        f"{prefix}.svh": node["svh"],
        f"{prefix}.mcg": torch.tensor(MCG_SENTINEL_SIGNED, dtype=torch.int32),
    }


def stage_tensor_names(
    layer: int, expert: int, projection: str, label: str, node: dict[str, Any]
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank0"
    return {
        f"{prefix}.trellis_{label}": node["trellis"],
        f"{prefix}.suh_{label}": node["suh"],
        f"{prefix}.svh_{label}": node["svh"],
        f"{prefix}.scale_{label}": torch.tensor(
            node["scale"], dtype=torch.float32),
    }


# ── Skeleton (everything that is not a routed expert weight) ──────────────

def skeleton_keys(source: SourceCheckpoint, layers: list[int]) -> dict[str, list[str]]:
    """shard -> keys that must be copied verbatim into every base checkpoint."""
    quantized = set()
    for layer in layers:
        for projections in source.experts(layer).values():
            quantized.update(projections.values())
    grouped: dict[str, list[str]] = {}
    if source.layout == SourceCheckpoint.INDEXED:
        items = source.weight_map.items()
    else:
        items = [
            (key, shard.name)
            for shard in sorted(source.root.glob("model-*.safetensors"))
            for key in source.keys_in_shard(shard.name)
        ]
    for key, shard in items:
        if key in quantized:
            continue
        grouped.setdefault(shard, []).append(key)
    return {shard: sorted(keys) for shard, keys in sorted(grouped.items()) if keys}


def copy_source_aux_files(source: Path, dest: Path) -> list[str]:
    """Copy tokenizer/config/license files; never stale integrity metadata.

    Hidden entries are skipped: a Hugging Face snapshot directory carries
    ``.cache`` and lock files that are not part of the checkpoint.
    """
    skip = {"model.safetensors.index.json", "MANIFEST.sha256", SENTINEL}
    copied = []
    dest.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if (entry.name in skip or entry.name.startswith(".")
                or entry.name.endswith(".safetensors")):
            continue
        target = dest / entry.name
        if entry.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(entry, target)
        elif entry.is_file():
            shutil.copy2(entry, target)
        copied.append(entry.name)
    return copied


def load_signer(path: Path, *, create: bool, outside: tuple[Path, ...] = ()) -> Signer:
    """Open the campaign signing key, creating it only where that is safe.

    Two hazards are refused here. ``Signer`` generates a key when the file is
    absent, which races a parallel encode: each worker would mint its own
    identity and sign part of the campaign with it, so only single-process
    entry points may create. And a key placed inside the campaign or the source
    tree would be uploaded with the artifacts, publishing the private seed that
    the whole trust root rests on.
    """
    path = Path(path).expanduser()
    resolved = path.resolve()
    for tree in outside:
        root = Path(tree).expanduser().resolve()
        if resolved == root or root in resolved.parents:
            raise CartridgeError(
                f"--sign-key {resolved} is inside {root}: refusing. That "
                f"directory is published; a private ed25519 seed must live "
                f"outside every tree the campaign uploads.")
    if not resolved.exists() and not create:
        raise CartridgeError(
            f"signing key {resolved} does not exist. Parallel workers must not "
            f"create it: run `skeleton` first, or start encode with --devices, "
            f"either of which creates it once.")
    return Signer(path)


def resolve_identity(args, source: SourceCheckpoint) -> tuple[str, str]:
    """The (base_model, base_revision) every attestation pins."""
    repo, revision = infer_source_identity(source.root)
    base_model = args.base_model or repo
    base_revision = args.base_revision or revision
    if not base_model or not base_revision:
        raise CartridgeError(
            f"cannot attest {source.root}: pass --base-model and "
            f"--base-revision (a 40-hex commit). They are inferred "
            f"automatically only from a Hugging Face snapshot directory.")
    return base_model, base_revision


def cmd_skeleton(args) -> int:
    """Extract every non-expert tensor from the shards that are present."""
    require_quant_dependencies()
    recipe = load_recipe(args.recipe)
    recipe_sha = sha256_file(args.recipe)
    with SourceCheckpoint(args.source) as source:
        base_model, base_revision = resolve_identity(args, source)
        out = prepare_out_dir(
            args.out, recipe_path=args.recipe, recipe_sha=recipe_sha,
            source=source, base_model=base_model,
            base_revision=base_revision,
            block_size=args.block_size, force=args.force)
        signer = load_signer(args.sign_key, create=True,
                             outside=(out, source.root))
        # The skeleton is copied verbatim out of the source shards, so its
        # honest predicate is repack-of, pinned to the source file's digest.
        provenance = build_provenance(
            predicate="repack-of", recipe_sha=recipe_sha, source=source,
            base_model=base_model, base_revision=base_revision, signer=signer,
            source_digest=source_digest_resolver(
                source, out, load_digest_map(args.source_digests)))
        skeleton = out / "skeleton"
        grouped = skeleton_keys(source, recipe["moe_layers"])
        aux = copy_source_aux_files(source.root, skeleton)
        print(f"skeleton: {len(grouped)} shards, {len(aux)} auxiliary files, "
              f"signer {signer.pub_hex[:16]}", flush=True)
        written = skipped = absent = 0
        for shard, keys in grouped.items():
            expect = {"schema": BLOCK_SCHEMA, "recipe_sha256": recipe_sha,
                      "base_revision": base_revision,
                      "kind": "skeleton", "tensor_count": str(len(keys))}
            if not args.force and shard_is_complete(skeleton, shard, expect):
                skipped += 1
                continue
            if not (source.root / shard).is_file():
                absent += 1
                continue
            groups = [(key, [(key, source.tensor(shard, key))])
                      for key in keys]
            write_shard(groups, skeleton, shard,
                        {**expect, "tool": TOOL_VERSION},
                        provenance=provenance, group_kind="tensor")
            written += 1
            del groups
            print(f"  {shard}: {len(keys)} tensors", flush=True)
    print(f"skeleton done: {written} written, {skipped} already complete, "
          f"{absent} shards not present locally", flush=True)
    return 0


# ── Encode ────────────────────────────────────────────────────────────────

def selected_layers(recipe: dict[str, Any], spec: str | None) -> list[int]:
    if not spec:
        return recipe["moe_layers"]
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            wanted.update(range(int(lo), int(hi) + 1))
        else:
            wanted.add(int(part))
    layers = [layer for layer in recipe["moe_layers"] if layer in wanted]
    unknown = sorted(wanted - set(recipe["moe_layers"]))
    if unknown:
        raise CartridgeError(f"--layers requests non-recipe layers {unknown}")
    if not layers:
        raise CartridgeError("--layers selected nothing")
    return layers


def build_work_list(
    source: SourceCheckpoint, recipe: dict[str, Any], layers: list[int],
    block_size: int,
) -> list[dict[str, Any]]:
    work = []
    for layer in layers:
        blocks = expert_blocks(list(source.experts(layer)), block_size)
        for index, experts in enumerate(blocks):
            work.append({"layer": layer, "block": index, "experts": experts})
    return work


def block_outputs(
    recipe: dict[str, Any], experts: list[int]
) -> dict[str, list[int]]:
    """label -> experts of this block that the node covers (non-empty only)."""
    out: dict[str, list[int]] = {}
    for node in [*recipe["bases"], *recipe["stages"]]:
        covered = (experts if node["experts"] == "all"
                   else [e for e in experts if e in set(node["experts"])])
        if covered:
            out[node["label"]] = covered
    return out


@contextmanager
def block_claim(out: Path, layer: int, block: int):
    """Advisory single-writer claim for one (layer, block).

    Two launchers pointed at one campaign would otherwise spend rented GPU
    hours encoding the same block twice and interleave its commit markers. The
    claim is created with ``O_EXCL``, so exactly one worker wins; the loser
    skips the block. A crashed worker leaves its claim behind on purpose --
    ``--devices`` clears them once, before it forks, when no worker can be
    running.
    """
    locks = out / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    path = locks / f"layer-{layer:03d}-b{block:03d}.lock"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        yield False
        return
    try:
        os.write(handle, f"{os.getpid()}\n".encode())
        os.close(handle)
        yield True
    finally:
        path.unlink(missing_ok=True)


def encode_block(
    source: SourceCheckpoint, recipe: dict[str, Any], out: Path,
    item: dict[str, Any], device: torch.device, enc: Encoder,
    recipe_sha: str, force: bool, provenance: dict[str, Any],
) -> dict[str, Any]:
    """Encode one (layer, expert block) into one shard per graph node."""
    layer, block, experts = item["layer"], item["block"], item["experts"]
    name = block_name(layer, block)
    wanted = block_outputs(recipe, experts)
    expect_common = {
        "schema": BLOCK_SCHEMA, "recipe_sha256": recipe_sha,
        "base_revision": provenance["base_revision"],
        "layer": str(layer), "block": str(block),
        "experts": ",".join(str(e) for e in experts),
    }
    # Not part of the resume identity: a rebuilt encoder must be free to
    # re-encode a block, and it rewrites the whole block when it does. It is
    # recorded so finalize can refuse to publish a family that silently spans
    # two encoder builds.
    encoder_sha = provenance["encoder"]["encoder_sha256"]
    # A block is one consistency unit. Residual stages are defined against the
    # reconstruction of the base shard that ships beside them, so a run that
    # rewrote only the missing stages would pin them to a base it recomputed in
    # memory -- which is the committed base only as long as nothing about the
    # stack changed. Either every node of this block is already committed, or
    # all of them are re-encoded together.
    plan = {}
    for label, covered in wanted.items():
        directory = node_dir(out, recipe, label)
        expect = {**expect_common, "label": label,
                  "covered_experts": ",".join(str(e) for e in covered)}
        plan[label] = (directory, expect, covered)
    complete = all(shard_is_complete(directory, name, expect)
                   for directory, expect, _covered in plan.values())
    if complete and not force:
        return {"layer": layer, "block": block, "skipped": True}
    pending = plan

    base_labels = {base["label"] for base in recipe["bases"]}
    source_experts = source.experts(layer)
    # label -> expert id -> [(tensor name, tensor)], written expert by expert so
    # every expert occupies one byte range that a signed digest can cover.
    tensors: dict[str, dict[int, list[tuple[str, torch.Tensor]]]] = {
        label: {} for label in pending}
    mses: dict[str, list[float]] = {label: [] for label in pending}
    started = time.perf_counter()
    for expert in experts:
        stages = stages_for_expert(recipe["stages"], expert)
        for projection in PROJECTIONS:
            key = source_experts[expert][projection]
            shard = source.layer_keys(layer)[key]
            raw = source.tensor(shard, key)
            if raw.ndim != 2 or raw.dtype is not torch.bfloat16:
                raise CartridgeError(
                    f"{key}: expected a 2-D BF16 weight, got "
                    f"{tuple(raw.shape)} {raw.dtype}")
            # Transpose on the device: half the bytes cross PCIe as BF16 and
            # the 12.6M-element permutation runs as a GPU kernel instead of a
            # CPU copy on the critical path of every matrix.
            weight = raw.to(device, non_blocking=True).float().T.contiguous()
            del raw
            nodes = encode_matrix_dag(
                weight, recipe["bases"], stages, device, enc)
            for label, node in nodes.items():
                if label not in pending:
                    continue
                named = (base_tensor_names(layer, expert, projection, node)
                         if label in base_labels
                         else stage_tensor_names(
                             layer, expert, projection, label, node))
                tensors[label].setdefault(expert, []).extend(named.items())
                mses[label].append(node["mse"])
            del weight, nodes
    quantize_s = time.perf_counter() - started

    nodes_by_label = node_index(recipe)
    written = {}
    # Parent-first: a stage's attestation names the exact bytes of the parent
    # shard it corrects, so "this residual belongs to that reconstruction" is a
    # signed, checkable fact rather than a naming convention.
    order = [base["label"] for base in recipe["bases"]]
    order += [stage["label"] for stage in recipe["stages"]]
    for label in [name for name in order if name in pending]:
        directory, expect, _covered = pending[label]
        node = nodes_by_label[label]
        parent = node.get("parent")
        parent_fragment = None
        if parent is not None:
            if parent not in written:
                raise CartridgeError(
                    f"layer {layer} block {block}: stage {label!r} has no "
                    f"parent shard for {parent!r} in this block")
            parent_fragment = {
                "label": parent,
                "file": name,
                "path": node_dir(out, recipe, parent).name + "/" + name,
                "sha256": written[parent],
            }
        groups = [(str(expert), tensors[label][expert])
                  for expert in sorted(tensors[label])]
        written[label] = write_shard(
            groups, directory, name,
            # Deliberately no timestamp: the shard's bytes are a function of
            # the recipe, the source and the encoder build, so re-encoding
            # inside the declared determinism scope reproduces this file
            # exactly. The wall clock lives in the attestation instead.
            {**expect, "tool": TOOL_VERSION, "encoder_sha256": encoder_sha},
            provenance={
                **provenance,
                "extra": {
                    "k": node["k"],
                    "quant_args": {
                        "K": node["k"],
                        "codebook": "mcg",
                        "mcg_multiplier": MCG_MULTIPLIER,
                        "codebook_scale": float(enc.cbs),
                        "label": label,
                        "parent": node.get("parent"),
                        "role": "base" if label in base_labels else "stage",
                        "regularize_seed": 0,
                        "hadamard_block": HADAMARD_BLOCK,
                        "rescale": (None if label in base_labels
                                    else "codebook_scale / residual_rms"),
                        "tile_batch": enc.tile_batch,
                    },
                    "cartridge": {
                        "label": label, "parent": parent,
                        "layer": layer, "block": block,
                        "covered_experts": _covered,
                        "mean_mse_original_space": (
                            sum(mses[label]) / len(mses[label])),
                    },
                    **({"parents": [parent_fragment]} if parent_fragment
                       else {}),
                },
            })
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Two rates, because only one of them is comparable to a campaign estimate:
    # the matrix loop is GPU work, and commit_s adds the grouped writes, span
    # hashing, signatures and digest sidecars that a block is not done without.
    commit_s = time.perf_counter() - started
    return {
        "layer": layer, "block": block, "skipped": False,
        "experts": len(experts), "encode_s": commit_s,
        "quantize_s": quantize_s, "commit_s": commit_s,
        "labels": {label: {"sha256": written[label],
                           "mse": sum(mses[label]) / len(mses[label])}
                   for label in sorted(written)},
    }


def cmd_encode(args) -> int:
    if args.devices:
        return _spawn_device_workers(args)
    require_quant_dependencies()
    recipe = load_recipe(args.recipe)
    recipe_sha = sha256_file(args.recipe)
    layers = selected_layers(recipe, args.layers)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise CartridgeError("--shard-index must be within --shard-count")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise CartridgeError(f"--device {device}: CUDA is unavailable")
        # Bind the process to its GPU before anything allocates. The encoder's
        # synchronization helper calls torch.cuda.synchronize() with no
        # argument, so a worker that left the current device at 0 would sync --
        # and create a context on -- the wrong GPU.
        torch.cuda.set_device(device)
        name = torch.cuda.get_device_name(device)
    else:
        name = device.type

    with SourceCheckpoint(args.source) as source:
        base_model, base_revision = resolve_identity(args, source)
        out = prepare_out_dir(
            args.out, recipe_path=args.recipe, recipe_sha=recipe_sha,
            source=source, base_model=base_model,
            base_revision=base_revision,
            block_size=args.block_size, force=args.force)
        signer = load_signer(args.sign_key,
                             create=args.shard_count == 1,
                             outside=(out, source.root))
        work = build_work_list(source, recipe, layers, args.block_size)
        mine = work[args.shard_index::args.shard_count]
        print(f"worker {args.shard_index}/{args.shard_count} on {device} "
              f"({name}): {len(mine)} of {len(work)} blocks, "
              f"{len(recipe['bases'])} bases + {len(recipe['stages'])} stages, "
              f"signer {signer.pub_hex[:16]}", flush=True)
        done = skipped = contended = 0
        elapsed = quantized = 0.0
        matrices = 0
        with bootstrap_encoder(args.encoder_source, args.tile_batch) as enc:
            print(f"codebook_scale = {enc.cbs}, tile_batch = {enc.tile_batch}",
                  flush=True)
            provenance = build_provenance(
                predicate="encode-of", recipe_sha=recipe_sha, source=source,
                base_model=base_model, base_revision=base_revision,
                signer=signer,
                encoder=enc.identity,
                scope=determinism_scope(device, enc.tile_batch),
                capture=capture_evidence(
                    recipe_sha, base_revision, enc.cbs))
            for item in mine:
                with block_claim(out, item["layer"], item["block"]) as owned:
                    if not owned:
                        contended += 1
                        continue
                    result = encode_block(
                        source, recipe, out, item, device, enc, recipe_sha,
                        args.force, provenance)
                if result["skipped"]:
                    skipped += 1
                    continue
                done += 1
                elapsed += result["commit_s"]
                quantized += result["quantize_s"]
                count = result["experts"] * len(PROJECTIONS)
                matrices += count
                print(f"  layer {result['layer']} block {result['block']}: "
                      f"{result['experts']} experts, "
                      f"{result['commit_s']:.1f}s committed "
                      f"({result['commit_s'] / count:.3f}s/matrix, of which "
                      f"{result['quantize_s'] / count:.3f}s quantizing)",
                      flush=True)
    if matrices:
        print(f"worker {args.shard_index}: {done} blocks encoded, {skipped} "
              f"already complete, {elapsed / matrices:.4f}s per matrix "
              f"committed ({quantized / matrices:.4f}s quantizing), "
              f"{elapsed / 3600:.2f} GPU-h. Extrapolate a campaign from the "
              f"committed rate.", flush=True)
    else:
        print(f"worker {args.shard_index}: nothing to do "
              f"({skipped} blocks already complete)", flush=True)
    if contended:
        print(f"worker {args.shard_index}: {contended} blocks were claimed by "
              f"another worker and left to it", flush=True)
    return 0


def _spawn_device_workers(args) -> int:
    """Run one single-device worker process per GPU over disjoint blocks."""
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        raise CartridgeError("--devices listed no devices")
    require_quant_dependencies()
    # Run the full output/identity preflight in the parent, before creating a
    # single directory: `--out` pointing at the source or the checkout must fail
    # without having written anything anywhere.
    recipe = load_recipe(args.recipe)
    recipe_sha = sha256_file(args.recipe)
    selected_layers(recipe, args.layers)
    with SourceCheckpoint(args.source) as source:
        base_model, base_revision = resolve_identity(args, source)
        out = prepare_out_dir(
            args.out, recipe_path=args.recipe, recipe_sha=recipe_sha,
            source=source, base_model=base_model,
            base_revision=base_revision,
            block_size=args.block_size, force=args.force)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # No worker can be running yet, so any claim left here is from a crash.
    stale = sorted((out / "locks").glob("*.lock"))
    for path in stale:
        path.unlink()
    if stale:
        print(f"cleared {len(stale)} stale block claims", flush=True)
    # Create the signing key once here: eight workers racing to generate it
    # would each write a different seed and sign with a different identity.
    signer = load_signer(args.sign_key, create=True,
                         outside=(out, Path(args.source)))
    print(f"signing key {signer.pub_hex[:16]} at {args.sign_key}", flush=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = [sys.executable, str(Path(__file__).resolve()), "encode",
            "--source", str(args.source), "--recipe", str(args.recipe),
            "--out", str(args.out), "--encoder-source", str(args.encoder_source),
            "--sign-key", str(args.sign_key),
            "--block-size", str(args.block_size),
            "--tile-batch", str(args.tile_batch),
            "--shard-count", str(len(devices))]
    for flag, value in (("--base-model", args.base_model),
                        ("--base-revision", args.base_revision),
                        ("--layers", args.layers)):
        if value:
            base += [flag, value]
    if args.force:
        base += ["--force"]
    children = []
    for index, device in enumerate(devices):
        log = logs / f"encode-{stamp}-w{index}.log"
        handle = log.open("w")
        cmd = base + ["--device", device, "--shard-index", str(index)]
        children.append((
            index, device, log, handle,
            subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)))
        print(f"worker {index} -> {device}, log {log}", flush=True)
    failures = []
    for index, device, log, handle, process in children:
        code = process.wait()
        handle.close()
        tail = log.read_text().strip().splitlines()[-1:] or ["(no output)"]
        print(f"worker {index} ({device}) exit {code}: {tail[-1]}", flush=True)
        if code != 0:
            failures.append(index)
    if failures:
        raise CartridgeError(
            f"workers {failures} failed; see {logs}. Blocks already written "
            f"are complete: rerun the same command to resume.")
    return 0


# ── Finalize ──────────────────────────────────────────────────────────────

def write_base_metadata(
    base_dir: Path,
    base_k: int,
    layers: list[int],
    experts_per_layer: dict[int, int],
    known: dict[str, str],
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
        "mcg_multiplier": MCG_MULTIPLIER,
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
    regenerate_manifest(base_dir, known)


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink when the filesystem allows it; the payload is identical."""
    if dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_adapter_config(
    directory: Path,
    recipe: dict[str, Any],
    assembly: dict[str, Any],
    base_manifest_sha256: str,
    shards: list[str],
    tensor_count: int,
    *,
    schema: str = ADAPTER_CONFIG_SCHEMA,
    filename: str = "adapter_config.json",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the explicit custom MSRT runtime contract for one product."""
    nodes = node_index(recipe)
    base = nodes[assembly["base"]]
    config = {
        "schema": schema,
        "assembly": assembly["label"],
        "base": {"label": base["label"], "k": base["k"],
                 "manifest_sha256": base_manifest_sha256},
        "chain": [
            {"label": label, "k": nodes[label]["k"],
             "parent": nodes[label]["parent"],
             "experts": nodes[label]["experts"]}
            for label in assembly["chain"]
        ],
        "format": "exl3-msrt-full-rank",
        "standard_lora_compatible": False,
        "runtime_operation": RUNTIME_OPERATION,
        "codebook": "mcg",
        "mcg_multiplier": MCG_MULTIPLIER,
        "mcg_ownership": "adapter-config",
        "scale_shape": [],
        "shards": shards,
        "num_tensors": tensor_count,
        "tool_version": TOOL_VERSION,
        "created_utc": now_utc(),
    }
    if extra:
        config.update(extra)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(json.dumps(config, indent=2) + "\n")
    return config


def cmd_finalize(args) -> int:
    """Complete every base checkpoint and publish the assembly contracts."""
    recipe = load_recipe(args.recipe)
    recipe_sha = sha256_file(args.recipe)
    layers = recipe["moe_layers"]
    with SourceCheckpoint(args.source) as source:
        base_model, base_revision = resolve_identity(args, source)
        out = prepare_out_dir(
            args.out, recipe_path=args.recipe, recipe_sha=recipe_sha,
            source=source, base_model=base_model,
            base_revision=base_revision,
            block_size=args.block_size, force=False)
        experts_per_layer: dict[int, int] = {}
        expected: dict[str, list[tuple[Path, str, dict[str, str], str]]] = {}
        for layer in layers:
            ids = list(source.experts(layer))
            experts_per_layer[layer] = len(ids)
            blocks = expert_blocks(ids, args.block_size)
            for index, experts in enumerate(blocks):
                common = {
                    "schema": BLOCK_SCHEMA, "recipe_sha256": recipe_sha,
                    "base_revision": base_revision,
                    "layer": str(layer), "block": str(index),
                    "experts": ",".join(str(e) for e in experts),
                }
                for label, covered in block_outputs(recipe, experts).items():
                    expected.setdefault(label, []).append((
                        node_dir(out, recipe, label), block_name(layer, index),
                        {**common, "label": label,
                         "covered_experts": ",".join(str(e) for e in covered)},
                        "expert",
                    ))
        skeleton = out / "skeleton"
        skeleton_expected = [
            (skeleton, shard,
             {"schema": BLOCK_SCHEMA, "recipe_sha256": recipe_sha,
              "base_revision": base_revision, "kind": "skeleton",
              "tensor_count": str(len(keys))},
             "tensor")
            for shard, keys in sorted(skeleton_keys(source, layers).items())
        ]

    missing: list[str] = []
    broken: list[str] = []
    digests: dict[str, dict[str, str]] = {}
    signers: set[str] = set()
    encoders: set[str] = set()
    parents: dict[tuple[str, str], dict[str, Any] | None] = {}
    attested = 0

    def check(bucket: str, directory: Path, name: str,
              expect: dict[str, str], group_kind: str) -> None:
        """Publish nothing that is not on disk, hashed, and signed for.

        Every fragment is re-hashed from the bytes that survived, the recorded
        block identity is re-checked so a copied shard cannot masquerade as a
        different block, and its signed line must name the same digest and the
        same per-span digests.
        """
        nonlocal attested
        target = directory / name
        recorded = read_digest(directory, name)
        if not target.is_file() or recorded is None:
            missing.append(f"{bucket}/{name}")
            return
        try:
            if not shard_is_complete(directory, name, expect):
                missing.append(f"{bucket}/{name}")
                return
            sha, body, spans = verify_shard(target, group_kind=group_kind)
            payload = read_attestation(directory, name)
        except CartridgeError as exc:
            broken.append(str(exc))
            return
        if sha != recorded:
            broken.append(
                f"{bucket}/{name}: on-disk sha256 {sha} != recorded {recorded}")
            return
        fragment = payload.get("fragment") or {}
        claimed = payload.get(
            "expert_sha256" if group_kind == "expert" else "tensor_sha256")
        if (fragment.get("sha256") != sha or fragment.get("file") != name
                or fragment.get("size") != target.stat().st_size
                or fragment.get("body_offset") != body):
            broken.append(f"{bucket}/{name}: attestation names other bytes")
            return
        if claimed != spans:
            broken.append(f"{bucket}/{name}: attested span digests do not hold")
            return
        if (payload.get("recipe_sha256") != recipe_sha
                or payload.get("base_revision") != base_revision):
            broken.append(f"{bucket}/{name}: attests a different source or recipe")
            return
        expected_predicate = "encode-of" if group_kind == "expert" else "repack-of"
        if payload.get("predicate") != expected_predicate:
            broken.append(
                f"{bucket}/{name}: predicate {payload.get('predicate')!r} != "
                f"{expected_predicate!r}")
            return
        if payload.get("materials", {}).get("encoder_sha256") is not None:
            encoders.add(payload["materials"]["encoder_sha256"])
        parents[(bucket, name)] = (payload.get("parents") or [None])[0]
        digests.setdefault(bucket, {})[name] = sha
        signers.add(payload["_keyid"])
        attested += 1

    # A finalized base holds its own expert blocks *and* one hardlink per
    # skeleton shard, because that is what makes it loadable. Finalize must
    # therefore accept those names when it re-runs: a crash or preemption in the
    # middle of a multi-terabyte finalize is a resume, not a lost campaign.
    base_labels = {base["label"] for base in recipe["bases"]}
    published_skeleton = {name for _d, name, _e, _g in skeleton_expected}
    for label, entries in expected.items():
        for directory, name, expect, group_kind in entries:
            check(label, directory, name, expect, group_kind)
        directory = node_dir(out, recipe, label)
        allowed = {name for _d, name, _e, _g in entries}
        if label in base_labels:
            allowed |= published_skeleton
        stale = sorted(path.name for path in directory.glob("model-*.safetensors")
                       if path.name not in allowed)
        if stale:
            raise CartridgeError(
                f"{directory} holds {len(stale)} shards this recipe does not "
                f"describe, e.g. {stale[:3]}. They would land in the "
                f"regenerated index next to the real ones: remove them or "
                f"encode into a fresh --out.")
    for directory, name, expect, group_kind in skeleton_expected:
        check("skeleton", directory, name, expect, group_kind)
    expected_skeleton = {name for _d, name, _e, _g in skeleton_expected}
    stale = sorted(path.name for path in skeleton.glob("model-*.safetensors")
                   if path.name not in expected_skeleton)
    if stale:
        raise CartridgeError(
            f"{skeleton} holds {len(stale)} shards this recipe does not "
            f"describe, e.g. {stale[:3]}; they would be published into every "
            f"base checkpoint.")
    if missing:
        raise CartridgeError(
            f"{len(missing)} outputs are missing, e.g. {sorted(missing)[:5]}. "
            f"Run encode/skeleton to completion before finalize.")
    if broken:
        raise CartridgeError(
            f"{len(broken)} fragments failed verification, e.g. "
            f"{sorted(broken)[:3]}. Every published fragment must hash to its "
            f"recorded digest and carry a signed line naming those bytes.")
    if len(signers) != 1:
        raise CartridgeError(
            f"fragments are signed by {len(signers)} different keys "
            f"({sorted(k[:16] for k in signers)}); a campaign must publish one "
            f"signer identity")
    if len(encoders) > 1:
        raise CartridgeError(
            f"fragments were produced by {len(encoders)} different encoder "
            f"builds ({sorted(e[:16] for e in encoders)}). A residual is only "
            f"valid against the reconstruction its own encoder produced: "
            f"re-encode the affected blocks with one build before publishing.")
    # Every stage shard must name the parent shard bytes it corrects, and that
    # digest must be the one this campaign actually published for the parent.
    node_parents = {stage["label"]: stage["parent"] for stage in recipe["stages"]}
    for (bucket, name), claim in parents.items():
        expected_parent = node_parents.get(bucket)
        if expected_parent is None:
            if claim is not None:
                broken.append(f"{bucket}/{name}: a base tier claims a parent")
            continue
        published = digests.get(expected_parent, {}).get(name)
        if (not isinstance(claim, dict) or claim.get("label") != expected_parent
                or claim.get("file") != name
                or claim.get("sha256") != published):
            broken.append(
                f"{bucket}/{name}: attests parent {claim!r}, but this campaign "
                f"published {expected_parent}/{name} as {published}")
    if broken:
        raise CartridgeError(
            f"{len(broken)} fragments are not bound to the reconstruction they "
            f"correct, e.g. {sorted(broken)[:3]}")

    base_manifests: dict[str, str] = {}
    for base in recipe["bases"]:
        label = base["label"]
        directory = node_dir(out, recipe, label)
        directory.mkdir(parents=True, exist_ok=True)
        known = dict(digests[label])
        for entry in sorted(skeleton.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                # Merge, never replace: this base's own digests/ and
                # attestations/ live here and deleting them would retract the
                # commit markers for every block already encoded.
                shutil.copytree(entry, directory / entry.name,
                                dirs_exist_ok=True)
                continue
            if entry.suffix == ".safetensors":
                # Immutable payload: one inode shared by every base tier.
                link_or_copy(entry, directory / entry.name)
            else:
                # config.json and tier_bitmap.json are rewritten per base, so
                # they must never share an inode with another base or the
                # skeleton they came from.
                shutil.copy2(entry, directory / entry.name)
            if entry.name in digests.get("skeleton", {}):
                known[entry.name] = digests["skeleton"][entry.name]
        write_base_metadata(
            directory, base["k"], layers, experts_per_layer, known)
        base_manifests[label] = sha256_file(directory / "MANIFEST.sha256")
        print(f"base {label}: K{base['k']} checkpoint finalized in {directory}",
              flush=True)

    products = []
    signer = load_signer(args.sign_key, create=False,
                         outside=(out, Path(args.source)))
    if signer.pub_hex != next(iter(signers)):
        raise CartridgeError(
            f"--sign-key is {signer.pub_hex[:16]} but every fragment was "
            f"signed by {next(iter(signers))[:16]}. A consumer pins one key "
            f"for the plan and the fragments, so two identities would make "
            f"this campaign unusable.")
    for assembly in recipe["assemblies"]:
        shards, tensors = [], 0
        for label in assembly["chain"]:
            directory = node_dir(out, recipe, label)
            for name in sorted(digests[label]):
                relative = (directory / name).relative_to(out).as_posix()
                header, _ = read_header(directory / name)
                meta = header.pop("__metadata__", None) or {}
                shards.append({
                    "label": label,
                    "path": relative,
                    "sha256": digests[label][name],
                    # Coverage is published *in the signed plan* so a consumer
                    # who wants 96 experts can decide what to download before
                    # touching a single payload byte.
                    "layer": int(meta["layer"]),
                    "block": int(meta["block"]),
                    "experts": [int(value) for value
                                in meta["covered_experts"].split(",") if value],
                    "parent_label": node_parents[label],
                    "parent_sha256": digests[node_parents[label]][name],
                    # Provenance travels with the shard list so a consumer that
                    # fetches only this product still gets its signed evidence.
                    "attestation": attestation_path(
                        directory, name).relative_to(out).as_posix(),
                })
                tensors += len(header)
        expert_counts = set(experts_per_layer.values())
        config = write_adapter_config(
            out / "assemblies" / assembly["label"], recipe, assembly,
            base_manifests[assembly["base"]],
            [entry["path"] for entry in shards], tensors,
            schema=ASSEMBLY_SCHEMA, filename="assembly.json",
            extra={
                "paths_relative_to": "campaign root",
                "campaign": {
                    "recipe_sha256": recipe_sha,
                    "base_model": base_model,
                    "base_revision": base_revision,
                    "encoder_sha256": (next(iter(encoders)) if encoders
                                       else None),
                    "signer_pubkey": next(iter(signers)),
                    "block_size": args.block_size,
                    "moe_layers": layers,
                },
                "stage_shards": shards,
                "bits_per_weight": (
                    assembly_bpw(recipe, assembly, max(expert_counts))),
            })
        plan_dir = out / "assemblies" / assembly["label"]
        (plan_dir / "assembly.jsonl").write_text(
            signer.sign_line(config) + "\n")
        products.append({"label": assembly["label"],
                         "bits_per_weight": config["bits_per_weight"],
                         "shards": len(shards),
                         "signed_plan": "assembly.jsonl"})
        print(f"assembly {assembly['label']}: "
              f"{config['bits_per_weight']:.3f} bpw, {len(shards)} stage shards",
              flush=True)

    summary = {
        "schema": "fq-msrt-campaign/1",
        "tool": TOOL_VERSION,
        "recipe_sha256": recipe_sha,
        "block_size": args.block_size,
        "moe_layers": layers,
        "experts_per_layer": {str(k): v for k, v in experts_per_layer.items()},
        "bases": {base["label"]: {"k": base["k"],
                                  "manifest_sha256": base_manifests[base["label"]]}
                  for base in recipe["bases"]},
        "stages": {stage["label"]: {"k": stage["k"], "parent": stage["parent"],
                                    "shards": len(digests[stage["label"]])}
                   for stage in recipe["stages"]},
        "assemblies": products,
        "provenance": {
            "attested_fragments": attested,
            "signer_pubkey": next(iter(signers)),
            "attestation_schema": ATTESTATION_SCHEMA,
            "predicates": {"expert_shards": "encode-of",
                           "skeleton_shards": "repack-of"},
            "note": ("Every fragment carries one signed fq-attestation/1 line "
                     "under attestations/, naming its own sha256 and the "
                     "sha256 of each expert's contiguous byte span. Authority "
                     "over the signing key comes from keys/FINGERPRINTS, not "
                     "from these files."),
        },
        "encoded_bits_per_weight": encoded_bits_per_weight(
            recipe, max(set(experts_per_layer.values()))),
        "created_utc": now_utc(),
    }
    (out / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"\nFinalized {out}", flush=True)
    return 0


# ── Plan ──────────────────────────────────────────────────────────────────

def cmd_plan(args) -> int:
    """Report the work list, staged shard reads, and product bitrates."""
    recipe = load_recipe(args.recipe)
    layers = selected_layers(recipe, args.layers)
    with SourceCheckpoint(args.source) as source:
        per_layer = []
        matrices = 0
        expert_counts = set()
        for layer in layers:
            ids = list(source.experts(layer))
            expert_counts.add(len(ids))
            blocks = expert_blocks(ids, args.block_size)
            shards = sorted({shard for shard in source.layer_keys(layer).values()})
            matrices += len(ids) * len(PROJECTIONS)
            per_layer.append({
                "layer": layer,
                "experts": len(ids),
                "blocks": len(blocks),
                "shards": source.shard_bytes(set(shards)),
            })
        expert_count = max(expert_counts)
        nodes = node_index(recipe)
        plan = {
            "schema": PLAN_SCHEMA,
            "tool": TOOL_VERSION,
            "source": str(source.root),
            "layout": source.layout,
            "absent_source_shards": source.absent_shards[:20],
            "block_size": args.block_size,
            "nodes": {label: {"k": node["k"],
                              "parent": node.get("parent"),
                              "experts": selected_count(node, expert_count)}
                      for label, node in nodes.items()},
            "quantization_passes_per_matrix": len(nodes),
            "matrices": matrices,
            "blocks": sum(entry["blocks"] for entry in per_layer),
            "encoded_bits_per_weight": encoded_bits_per_weight(
                recipe, expert_count),
            "assemblies": [
                {"label": assembly["label"],
                 "base": assembly["base"],
                 "chain": assembly["chain"],
                 "bits_per_weight": assembly_bpw(recipe, assembly, expert_count)}
                for assembly in recipe["assemblies"]
            ],
            "layers": per_layer,
        }
    print(json.dumps(plan, indent=2))
    if args.out_plan:
        args.out_plan.write_text(json.dumps(plan, indent=2) + "\n")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────

def _add_common(parser: argparse.ArgumentParser, *, out_required: bool = True) -> None:
    parser.add_argument("--source", required=True, type=Path,
                        help="Source BF16 checkpoint directory")
    parser.add_argument("--recipe", required=True, type=Path,
                        help="Cartridge recipe JSON (fq-cartridge/2)")
    parser.add_argument("--out", required=out_required, type=Path,
                        help="Campaign output directory")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                        help="Experts per resumable work block")
    # Artifact identity: what every attestation pins and what the campaign
    # sentinel is bound to. Inferred from a Hugging Face snapshot path.
    parser.add_argument("--base-model",
                        help="Source model id, e.g. zai-org/GLM-5.2")
    parser.add_argument("--base-revision",
                        help="Immutable 40-hex source commit")
    parser.add_argument("--source-digests", type=Path,
                        help="JSON map of source file -> sha256 (or a Hugging "
                             "Face tree listing); avoids re-hashing shards")


def _add_signing(parser: argparse.ArgumentParser) -> None:
    """Every emitted fragment is signed; there is no opt-out, because an
    unattested shard can be neither published nor verified."""
    parser.add_argument("--sign-key", required=True, type=Path,
                        help="ed25519 seed file; created if absent")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Report work list and product bitrates")
    _add_common(plan, out_required=False)
    plan.add_argument("--layers", help="Layer subset, e.g. 3-10,40")
    plan.add_argument("--out-plan", type=Path, help="Also write the plan JSON")

    skel = sub.add_parser(
        "skeleton", help="Copy every non-expert tensor into the campaign")
    _add_common(skel)
    _add_signing(skel)
    skel.add_argument("--force", action="store_true",
                      help="Rewrite skeleton shards that are already complete")

    enc = sub.add_parser("encode", help="Encode base tiers and residual stages")
    _add_common(enc)
    _add_signing(enc)
    enc.add_argument("--encoder-source", required=True, type=Path,
                     help="Path to exllamav3 Python package")
    enc.add_argument("--device", default="cuda:0")
    enc.add_argument("--devices",
                     help="Comma-separated devices; runs one worker each")
    enc.add_argument("--layers", help="Layer subset, e.g. 3-10,40")
    enc.add_argument("--shard-index", type=int, default=0)
    enc.add_argument("--shard-count", type=int, default=1)
    enc.add_argument("--force", action="store_true",
                     help="Re-encode blocks that are already complete")
    enc.add_argument("--tile-batch", type=int, default=TILE_BATCH,
                     help=f"Tiles per trellis launch (default {TILE_BATCH}; "
                          f"measured optimum on Blackwell)")

    fin = sub.add_parser("finalize", help="Complete bases and write contracts")
    _add_common(fin)
    _add_signing(fin)

    args = p.parse_args(argv)
    try:
        if args.command == "plan":
            return cmd_plan(args)
        if args.command == "skeleton":
            return cmd_skeleton(args)
        if args.command == "encode":
            return cmd_encode(args)
        if args.command == "finalize":
            return cmd_finalize(args)
    except (AssemblyError, CartridgeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
