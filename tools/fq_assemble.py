#!/usr/bin/env python3
"""fq_assemble — Progressive Tensors segments + policy → bootable checkpoint.

The M0 gate tool (04-milestones.md): given per-layer per-K segment files
(fq_repack output or downloaded), a bits_per_expert policy JSON, and the
source checkpoint (for dense/attention/shared tensors and header order),
emit per-layer shards in the original GG rank-sliced layout.

Byte-identity property: assembling the all-K3 policy from K3 segments
reproduces the source shards byte-for-byte (same header, same order) —
tested in test_fq_assemble.py. Mixed policies swap in K4 expert bytes from
K4 segments as they become available.

Mixed-size support: trellis tensors are [in/16, out/16, 16*K] i16, so K4
rows are 4/3 the bytes of K3 and the source header cannot be reused as a
byte template. When any expert tensor's segment size differs from its
source slot, assemble_layer switches to a reindex path automatically: the
source header's TENSOR ORDER is kept (non-expert tensors stay byte-exact
from the source shard) but the header is rebuilt with dtype/shape taken
from the SEGMENT header for expert tensors and fresh data_offsets.

Reflink mode (--reflink): expert-tensor regions are written with
os.copy_file_range from the segment file's fd instead of read()+write().
On reflink-capable filesystems (XFS with reflink=1, btrfs) the kernel can
then share extents between the segment files and the assembled shards
instead of storing the same bytes twice.  Honest caveats: (1) extent
sharing only actually happens when the kernel/filesystem chooses to
reflink — copy_file_range may silently perform a plain (server-side)
copy; (2) real savings depend on 4K-block alignment of identical regions
at BOTH the source and destination offsets — MB-scale fragments usually
share most interior blocks, but alignment is not guaranteed; (3) output
byte-identity is ALWAYS preserved regardless — the mode changes how bytes
move, never what they are: every region falls back automatically to the
ordinary copy path when copy_file_range is unavailable or fails (EXDEV,
EOPNOTSUPP, ENOSYS, cross-filesystem, partial copy, ...); (4) this is a
local-disk-space optimization only — it has no effect on HF/remote
transfer or storage.  Implementation choice: only expert tensors (whose
bytes come from segment files) use copy_file_range; the safetensors
header, non-expert tensors, and dense pass-through shards keep the
ordinary mmap/read+write path since they come from the source shard, not
the segment files.

Mixed metadata (GG loader contract, exl3.py _configure_rank_sliced /
_load_rank_sliced_bitrates): config.json hybrid_tr3_tail must carry
bits="mixed", k_values=[3,4,...], and bits_per_expert as a
"file.json:field" reference; the referenced JSON maps str(layer) -> dict
whose field holds one int bitrate per expert.  We emit the map into
tier_bitmap.json under the field "bits_per_expert" and regenerate
model.safetensors.index.json + MANIFEST.sha256 for the new shard sizes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fq_repack import EXPERT_RE, read_header  # noqa: E402


def load_segment_index(seg_dir: Path, k: int) -> dict:
    p = seg_dir / f"index-k{k}.json"
    return json.loads(p.read_text()) if p.exists() else {}


class SegmentReader:
    """Random access to expert tensors inside one segment file."""

    def __init__(self, path: Path):
        self.hdr, self.body = read_header(path)
        self.hdr.pop("__metadata__", None)
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def tensor_bytes(self, name: str) -> bytes:
        a, b = self.hdr[name]["data_offsets"]
        return self.mm[self.body + a: self.body + b]

    def tensor_range(self, name: str) -> tuple[int, int, int]:
        """(fd, absolute_file_offset, length) of one tensor's bytes — the
        fd-level view of tensor_bytes, for os.copy_file_range."""
        a, b = self.hdr[name]["data_offsets"]
        return self.f.fileno(), self.body + a, b - a

    def close(self):
        self.mm.close()
        self.f.close()


def _copy_file_range_region(out, src_fd: int, src_off: int, length: int) -> bool:
    """Write length bytes from src_fd[src_off:] at out's current position
    via os.copy_file_range (server-side copy; XFS/btrfs may share extents).

    Returns True when the whole region was copied and out's position was
    advanced past it.  Returns False — with out's position restored to the
    start of the region — when the syscall is unavailable, refused (EXDEV,
    EOPNOTSUPP, ENOSYS, EINVAL, ...), or stalls, including after partial
    progress; the caller then rewrites the SAME region through the ordinary
    read()+write() path, so the output bytes are identical either way.
    """
    cfr = getattr(os, "copy_file_range", None)
    if cfr is None:
        return False
    out.flush()
    dst_fd = out.fileno()
    dst_off = out.tell()
    done = 0
    while done < length:
        try:
            n = cfr(src_fd, dst_fd, length - done,
                    src_off + done, dst_off + done)
        except OSError:
            n = 0
        if n <= 0:  # error or unexpected EOF: caller rewrites the region
            out.seek(dst_off)
            return False
        done += n
    out.seek(dst_off + length)
    return True


def _needs_reindex(hdr: dict, bits_for_expert, readers: dict[int, SegmentReader]) -> bool:
    """True when any expert tensor's segment size differs from its source slot."""
    for name, t in hdr.items():
        m = EXPERT_RE.match(name)
        if not m:
            continue
        k = bits_for_expert(int(m.group(2)))
        seg = readers[k].hdr.get(name)
        if seg is None:
            raise KeyError(f"segment k{k} is missing tensor {name}")
        if (seg["data_offsets"][1] - seg["data_offsets"][0]
                != t["data_offsets"][1] - t["data_offsets"][0]):
            return True
    return False


def assemble_layer(
    src_shard: Path,
    out_shard: Path,
    layer: int,
    bits_for_expert,           # callable expert_id -> K
    readers: dict[int, SegmentReader],
    reflink: bool = False,
) -> dict:
    """Write one output shard.

    Uniform-K (all segment sizes match the source slots): the source header
    is reused verbatim as a byte template — output is byte-identical to the
    source when the policy matches the source K.  Mixed K: reindex path —
    same tensor ORDER as the source, but the header is rebuilt with
    dtype/shape from the segment headers for expert tensors and fresh
    contiguous data_offsets.

    reflink=True: expert-tensor regions go through os.copy_file_range with
    per-region fallback to the ordinary path (see module docstring); the
    returned counts dict then carries "reflinked"/"copied" region tallies
    alongside the per-K tensor counts.  Output bytes are identical either
    way.
    """
    hdr, body_off = read_header(src_shard)
    meta = hdr.pop("__metadata__", None)
    counts = {}
    reindex = _needs_reindex(hdr, bits_for_expert, readers)
    with open(src_shard, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        ordered = sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0])
        if reindex:
            rebuilt, off = {}, 0
            for name, t in ordered:
                m = EXPERT_RE.match(name)
                if m:
                    seg = readers[bits_for_expert(int(m.group(2)))].hdr[name]
                    dtype, shape = seg["dtype"], seg["shape"]
                    size = seg["data_offsets"][1] - seg["data_offsets"][0]
                else:
                    dtype, shape = t["dtype"], t["shape"]
                    size = t["data_offsets"][1] - t["data_offsets"][0]
                rebuilt[name] = {"dtype": dtype, "shape": shape,
                                 "data_offsets": [off, off + size]}
                off += size
            ordered = [(n, rebuilt[n]) for n, _ in ordered]
            header = rebuilt if meta is None else {"__metadata__": meta, **rebuilt}
        else:
            header = dict(hdr) if meta is None else {"__metadata__": meta, **hdr}
        hj = json.dumps(header, separators=(",", ":")).encode()
        hj += b" " * ((8 - len(hj) % 8) % 8)
        tmp = out_shard.with_suffix(".part")
        with open(tmp, "wb") as out:
            out.write(struct.pack("<Q", len(hj)))
            out.write(hj)
            for name, t in ordered:
                want = t["data_offsets"][1] - t["data_offsets"][0]
                m = EXPERT_RE.match(name)
                if m:
                    expert = int(m.group(2))
                    k = bits_for_expert(expert)
                    counts[k] = counts.get(k, 0) + 1
                    if reflink:
                        src_fd, src_off, size = readers[k].tensor_range(name)
                        if size != want:
                            raise AssertionError(
                                f"{name}: got {size} bytes, header slot {want}")
                        if _copy_file_range_region(out, src_fd, src_off, size):
                            counts["reflinked"] = counts.get("reflinked", 0) + 1
                            continue
                        counts["copied"] = counts.get("copied", 0) + 1
                    data = readers[k].tensor_bytes(name)
                else:
                    src_t = hdr[name]
                    a, b = src_t["data_offsets"]
                    data = mm[body_off + a: body_off + b]
                if len(data) != want:
                    raise AssertionError(
                        f"{name}: got {len(data)} bytes, header slot {want}")
                out.write(data)
        mm.close()
    os.replace(tmp, out_shard)
    return counts


BITS_PER_EXPERT_REF = "tier_bitmap.json:bits_per_expert"


def emit_mixed_metadata(out_dir: Path, bpe: dict, source: Path) -> None:
    """Rewrite config.json/tier_bitmap.json for the policy's K set.

    Uniform policy: set hybrid_tr3_tail.bits to the single K (drop any mixed
    fields).  Mixed policy: bits="mixed", k_values, and a bits_per_expert
    "file.json:field" reference per the GG loader contract; the per-layer
    expert bitrate lists are written into tier_bitmap.json.
    """
    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    tail = cfg.get("hybrid_tr3_tail")
    if not isinstance(tail, dict):
        return
    ks = sorted({int(k) for bits in bpe.values() for k in bits})
    if len(ks) == 1:
        changed = tail.get("bits") != float(ks[0])
        tail["bits"] = float(ks[0])
        for key in ("k_values", "bits_per_expert"):
            changed |= tail.pop(key, None) is not None
        if changed:
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        return
    tail["bits"] = "mixed"
    tail["k_values"] = ks
    tail["bits_per_expert"] = BITS_PER_EXPERT_REF
    # vLLM resolves the quant method from hf_config.quantization_config
    # (weight_utils.get_quant_config) BEFORE Exl3Config.maybe_update_config
    # reads hybrid_tr3_tail; without this stub it demands a
    # quantization_config.json file the rank-sliced format does not ship.
    cfg.setdefault("quantization_config", {
        "quant_method": "exl3",
        "bits": "mixed",
        "codebook": tail.get("codebook", "mcg"),
        "version": tail.get("exllamav3_version", "rank-sliced"),
    })
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    bitmap_name, field = BITS_PER_EXPERT_REF.rsplit(":", 1)
    src_bitmap = source / bitmap_name
    bitmap = json.loads(src_bitmap.read_text()) if src_bitmap.exists() else {}
    for layer_s, bits in bpe.items():
        entry = bitmap.setdefault(layer_s, {})
        entry[field] = [int(b) for b in bits]
    (out_dir / bitmap_name).write_text(json.dumps(bitmap, indent=1) + "\n")


def regenerate_shard_index(out_dir: Path) -> None:
    """Rebuild model.safetensors.index.json from the output shards."""
    weight_map, total = {}, 0
    for shard in sorted(out_dir.glob("model-*.safetensors")):
        hdr, _ = read_header(shard)
        hdr.pop("__metadata__", None)
        for name, t in hdr.items():
            weight_map[name] = shard.name
            total += t["data_offsets"][1] - t["data_offsets"][0]
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=1, sort_keys=True) + "\n")


def regenerate_manifest(out_dir: Path) -> None:
    """Rewrite MANIFEST.sha256 (sha256sum format) for the output files."""
    lines = []
    for f in sorted(out_dir.iterdir()):
        if not f.is_file() or f.name == "MANIFEST.sha256":
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        lines.append(f"{h.hexdigest()}  {f.name}")
    (out_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segments", required=True, type=Path, help="segment dir (repack output)")
    p.add_argument("--source", required=True, type=Path, help="source checkpoint snapshot")
    p.add_argument("--policy", required=True, type=Path, help="fq-policy JSON (bits_per_expert)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--layers", default=None)
    p.add_argument(
        "--reflink", action="store_true",
        help="write expert-tensor bytes from segment files with "
             "os.copy_file_range so reflink-capable filesystems (XFS with "
             "reflink=1, btrfs) can share extents with the segments instead "
             "of storing the bytes twice. Caveats: the kernel/filesystem "
             "decides whether extents are actually shared — copy_file_range "
             "may silently do a plain copy; savings depend on 4K-block "
             "alignment of identical regions at both source and destination "
             "offsets (MB-scale fragments usually share most interior "
             "blocks, but alignment is not guaranteed); output bytes are "
             "ALWAYS identical to the normal path (automatic per-region "
             "fallback to read()+write() on EXDEV/EOPNOTSUPP/etc.); saves "
             "local disk space only — no effect on HF/remote.")
    args = p.parse_args(argv)

    policy = json.loads(args.policy.read_text())
    bpe = policy["bits_per_expert"]
    ks = sorted({k for v in bpe.values() for k in v})
    args.out.mkdir(parents=True, exist_ok=True)

    wanted = None
    if args.layers:
        wanted = set()
        for part in args.layers.split(","):
            if "-" in part:
                a, b = part.split("-")
                wanted.update(range(int(a), int(b) + 1))
            else:
                wanted.add(int(part))

    total = {}
    resized = False
    for src_shard in sorted(args.source.glob("model-layer-*.safetensors")):
        layer = int(src_shard.stem.split("-")[-1])
        if wanted is not None and layer not in wanted:
            continue
        lb = bpe.get(str(layer))
        if lb is None:
            # dense layer (or out-of-policy): copy through
            out = args.out / src_shard.name
            if not out.exists():
                out.write_bytes(src_shard.read_bytes())
            continue
        readers = {}
        for k in sorted(set(lb)):
            seg = args.segments / f"layer-{layer:03d}.k{k}.safetensors"
            if not seg.exists():
                raise FileNotFoundError(f"missing segment {seg} for layer {layer}")
            readers[k] = SegmentReader(seg)
        out_shard = args.out / src_shard.name
        counts = assemble_layer(
            src_shard, out_shard, layer,
            lambda e: lb[e], readers, reflink=args.reflink)
        for r in readers.values():
            r.close()
        resized |= out_shard.stat().st_size != src_shard.stat().st_size
        total[layer] = counts
        print(f"layer {layer:3d}: {counts}", flush=True)

    for extra in args.source.iterdir():
        if extra.is_file() and not extra.name.startswith("model-layer-"):
            dst = args.out / extra.name
            if not dst.exists():
                dst.write_bytes(extra.read_bytes())
    emit_mixed_metadata(args.out, bpe, args.source)
    if resized:
        if (args.source / "model.safetensors.index.json").exists():
            regenerate_shard_index(args.out)
        if (args.source / "MANIFEST.sha256").exists():
            regenerate_manifest(args.out)
    print(f"assembled {len(total)} MoE layers -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
