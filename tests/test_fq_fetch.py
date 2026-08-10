"""Tests for fq_fetch: the consumer range-fetch path.

A synthetic segment repo (built with fq_repack from synthetic shards) is
served over monkeypatched HTTP so every byte of remote IO is exercised
locally: ranged GETs, coalescing, signature checks, per-expert digest checks,
multi-source selection, resume, and — the end-to-end property that matters —
a fetched subset tree assembles to bytes identical to a tree that was
downloaded whole.
"""
import hashlib
import json
import shutil
import struct
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_assemble  # noqa: E402
import fq_fetch  # noqa: E402
import fq_release  # noqa: E402
import fq_repack  # noqa: E402
import fq_trust  # noqa: E402
from test_fq_repack import E, LAYERS, write_shard  # noqa: E402

REV = "0123456789abcdef0123456789abcdef01234567"
KS = (3, 4)


# ----------------------------------------------------------------- fixtures

def build_source(tmp_path: Path, name: str, *, ks=KS, key: Path = None,
                 salt: str = "") -> tuple[Path, Path, str]:
    """A segment repo with one index/segment/attestation set per K.

    Returns (repo_dir, snapshot_dir_for_k3, signer_pubkey).  `salt` perturbs
    the shard bytes so two "publishers" can be made to disagree about the
    same expert, which is what content-hash selection has to resolve.
    """
    key = key or (tmp_path / f"{name}.key")
    repo = tmp_path / name
    repo.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    manifest = None
    for k in ks:
        snap = tmp_path / f"{name}-snap-k{k}"
        snap.mkdir(exist_ok=True)
        for i, layer in enumerate(LAYERS):
            write_shard(snap / f"model-layer-{layer:03d}.safetensors", layer,
                        scramble=bool(i), k=k)
        if salt:  # differ from the other publisher in one expert's bytes
            _perturb(snap / f"model-layer-{LAYERS[0]:03d}.safetensors", salt)
        lines = [
            f"{hashlib.sha256((snap / f'model-layer-{l:03d}.safetensors').read_bytes()).hexdigest()}"
            f"  model-layer-{l:03d}.safetensors" for l in LAYERS]
        (snap / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
        snapshots[k] = snap
        out = tmp_path / f"{name}-seg-k{k}"
        assert fq_repack.main([
            "--snapshot", str(snap), "--source-repo", f"test/{name}",
            "--revision", REV, "--base-model", "test/base",
            "--out", str(out), "--k", str(k), "--sign-key", str(key)]) == 0
        for f in out.glob("*.safetensors"):
            shutil.copy2(f, repo / f.name)
        shutil.copy2(out / f"index-k{k}.json", repo / f"index-k{k}.json")
        (repo / "attestations").mkdir(exist_ok=True)
        for f in (out / "attestations").glob("*.jsonl"):
            shutil.copy2(f, repo / "attestations" / f.name)
        manifest = json.loads((out / "fq-manifest.json").read_text())
    manifest["k_variants"] = list(ks)
    manifest["tensor_indexes"] = {str(k): f"index-k{k}.json" for k in ks}
    manifest.pop("tensor_index", None)
    (repo / "fq-manifest.json").write_text(json.dumps(manifest, indent=1))
    return repo, snapshots[ks[0]], manifest["signer_pubkey"]


def _perturb(shard: Path, salt: str) -> None:
    """Rewrite expert 0's first tensor payload so this publisher's fragment
    differs from the other's while staying a valid shard."""
    raw = bytearray(shard.read_bytes())
    hlen = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + hlen])
    name = next(n for n in hdr if n != "__metadata__" and ".experts.0." in n)
    a, b = hdr[name]["data_offsets"]
    body = 8 + hlen
    blob = hashlib.sha256(salt.encode()).digest()
    raw[body + a: body + b] = (blob * 8)[: b - a]
    shard.write_bytes(bytes(raw))


@pytest.fixture()
def served(monkeypatch):
    """Serve one or more repo dirs at https://huggingface.co/<repo>/resolve/..."""
    repos: dict[str, Path] = {}
    calls = {"ranges": [], "full": []}

    def resolve(url: str) -> Path:
        for repo_id, root in repos.items():
            marker = f"/{repo_id}/resolve/"
            if marker in url:
                name = url.split(marker, 1)[1].split("/", 1)[1]
                return root / name
        raise AssertionError(f"unmapped url {url}")

    def get_range(url, start, end, timeout=600.0):
        p = resolve(url)
        if not p.exists():
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        data = p.read_bytes()[start:end]
        calls["ranges"].append((p.name, start, end))
        if len(data) != end - start:
            raise IOError("short read")
        return data, {"commit": REV, "total": p.stat().st_size}

    def get_full(url, timeout=600.0):
        p = resolve(url)
        if not p.exists():
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        calls["full"].append(p.name)
        return p.read_bytes(), {"commit": REV}

    monkeypatch.setattr(fq_fetch, "http_get_range", get_range)
    monkeypatch.setattr(fq_fetch, "http_get_full", get_full)
    calls["mount"] = lambda repo_id, root: repos.__setitem__(repo_id, root)
    return calls


def trust_root(tmp_path: Path, pub: str) -> Path:
    """A keys/FINGERPRINTS-shaped trust root holding the test signer, so the
    tests resolve fingerprints the way a consumer with a git clone does."""
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    (root / "FINGERPRINTS").write_text(
        f"# test trust root\n{pub}  test-signer  active  2026-08-10  segments\n")
    (root / "test-signer.ed25519.pub").write_text(pub + "\n")
    return root


def write_policy(path: Path, mapping: dict) -> Path:
    path.write_text(json.dumps({
        "schema": "fq-policy/2",
        "bits_per_expert": {str(l): list(ks) for l, ks in mapping.items()}}))
    return path


def run(argv, expect=0):
    """Invoke fq_fetch, keeping its local signing key inside tmp_path.

    (Without --sign-key the tool would create ~/.fq_keys/fq_fetch.key, which
    is right for a user and wrong for a test.)"""
    argv = [str(a) for a in argv]
    if "--sign-key" not in argv and "--no-attest" not in argv:
        out = Path(argv[argv.index("--out") + 1])
        argv += ["--sign-key", str(out.parent / "local.key")]
    rc = fq_fetch.main(argv)
    assert rc == expect, f"fq_fetch returned {rc}, expected {expect}"
    return rc


# --------------------------------------------------------------------- tests

def test_dry_run_reports_savings_and_writes_nothing(tmp_path, served, capsys):
    repo, _, pub = build_source(tmp_path, "pub")
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3, 4, 3, 4]})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub),
         "--dry-run", "--json", tmp_path / "plan.json"])
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["dry_run"] and plan["experts"] == E
    # ranged < the two segment files it touches < the whole repo (4 files)
    assert plan["ranged_bytes"] < plan["whole_segment_files_bytes"]
    assert plan["whole_segment_files_bytes"] < plan["whole_repo_bytes"]
    assert not list(out.glob("*.safetensors"))
    assert "what `hf download <repo>` costs" in capsys.readouterr().out


def _fetch_all(tmp_path, served, extra=()):
    repo, snap, pub = build_source(tmp_path, "pub")
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json",
                          {LAYERS[0]: [3, 4, 3, 4], LAYERS[1]: [4, 4, 3, 3]})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub),
         "--sign-key", tmp_path / "local.key", *extra])
    return repo, snap, out, policy, pub


def test_fetched_subset_assembles_byte_identically(tmp_path, served):
    """The property the whole tool exists for: assembling from a range-fetched
    subset gives exactly the checkpoint you would have got from the full
    download."""
    repo, snap, out, policy, pub = _fetch_all(tmp_path, served)
    full_out = tmp_path / "asm-full"
    sub_out = tmp_path / "asm-subset"
    # fq_assemble verifies every segment against a PINNED signer and fails
    # closed without one.  The published family is signed by the publisher;
    # the fetched subset consists of new files, so fq_fetch re-attests them
    # (derived-from, parents pinned by digest) under the local key whose
    # fingerprint it printed.  Both halves are verified — neither uses
    # --insecure — and both must produce the same checkpoint.
    local = json.loads((out / "fq-manifest.json").read_text())["signer_pubkey"]
    assert local and local != pub, "the subset must be signed by the local key"
    for segments, dest, pin in ((repo, full_out, pub), (out, sub_out, local)):
        assert fq_assemble.main([
            "--segments", str(segments), "--source", str(snap),
            "--policy", str(policy), "--out", str(dest),
            "--trust-signer", pin]) == 0
    for shard in sorted(full_out.glob("model-layer-*.safetensors")):
        assert shard.read_bytes() == (sub_out / shard.name).read_bytes(), shard.name


def test_only_needed_bytes_are_fetched(tmp_path, served):
    repo, snap, out, policy, _ = _fetch_all(tmp_path, served)
    report = json.loads((out / "fq-fetch-report.json").read_text())
    # Payload bytes are what scale: expert spans fetched vs the published
    # files they live in.  (Total transport also carries indexes, attestations
    # and segment headers — negligible at GLM scale, comparable in a toy repo
    # whose "experts" are a few hundred bytes each.)
    assert report["bytes"]["ranged_bytes"] < report["bytes"]["whole_segment_files_bytes"]
    assert report["bytes"]["whole_segment_files_bytes"] <= report["bytes"]["whole_repo_bytes"]
    # every fetched segment file is smaller than the published one it came from
    for k in KS:
        for layer_s, entry in json.loads((out / f"index-k{k}.json").read_text()).items():
            published = json.loads((repo / f"index-k{k}.json").read_text())[layer_s]
            assert entry["size"] < published["size"]
            assert set(entry["experts"]) <= set(published["experts"])


def test_local_tree_is_self_describing(tmp_path, served):
    repo, snap, out, policy, pub = _fetch_all(tmp_path, served)
    manifest = json.loads((out / "fq-manifest.json").read_text())
    assert manifest["schema"] == "fq-manifest/1"
    assert manifest["kind"] == "fetched-subset"
    # signer_pubkey names whoever signed THIS tree — the local key, because
    # the subset files are new files — and the upstream key that was checked
    # while fetching is recorded separately rather than conflated with it.
    assert manifest["signer_pubkey"] != pub
    assert manifest["upstream_signer"] == pub
    assert manifest["upstream_trust_rung"] == fq_trust.RUNG_PINNED
    assert sorted(manifest["k_variants"]) == list(KS)
    report = json.loads((out / "fq-fetch-report.json").read_text())
    assert report["trust"]["rung"] == fq_trust.RUNG_PINNED
    assert report["trust"]["signatures_verified"] > 0
    assert report["experts"][str(LAYERS[0])]["k3"]["0"]["source"] == f"test/pub@{REV}"
    # the attestations that justified the bytes are kept for offline re-checks
    assert list((out / "attestations").rglob("*.jsonl"))


def test_expert_digests_are_checked_against_the_attestation(tmp_path, served, capsys):
    repo, snap, pub = build_source(tmp_path, "pub")
    served["mount"]("test/pub", repo)
    seg = repo / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    raw = bytearray(seg.read_bytes())
    raw[-1] ^= 0xFF  # flip a payload bit after the attestation was signed
    seg.write_bytes(bytes(raw))
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)], expect=1)
    err = capsys.readouterr().err
    assert "TRUST FAILURE" in err and "signed attestation" in err
    assert not list(out.glob("*.safetensors"))  # nothing finalized


def test_wrong_pinned_signer_refuses_everything(tmp_path, served, capsys):
    repo, snap, pub = build_source(tmp_path, "pub")
    served["mount"]("test/pub", repo)
    other = "de" * 32
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    run(["--policy", policy, "--out", tmp_path / "f", "--source", f"test/pub@{REV}",
         "--trust-signer", other, "--trust-root", trust_root(tmp_path, pub)], expect=1)
    err = capsys.readouterr().err
    assert "no source carries it" in err or "trusted signer" in err


def test_placeholder_signature_is_not_a_pass(tmp_path, served, capsys):
    """Regression for the review's headline: 'AA==' must never verify."""
    repo, snap, pub = build_source(tmp_path, "pub")
    att = repo / "attestations" / f"layer-{LAYERS[0]:03d}.k3.jsonl"
    line = json.loads(att.read_text())
    line["signature"] = "AA=="
    att.write_text(json.dumps(line) + "\n")
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    run(["--policy", policy, "--out", tmp_path / "f", "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)], expect=1)
    assert "64 bytes" in capsys.readouterr().err


def test_content_hash_selection_picks_the_named_fragment(tmp_path, served):
    """Two publishers, same expert id, different bytes: --prefer-sha decides."""
    key = tmp_path / "shared.key"
    repo_a, _, pub = build_source(tmp_path, "a", ks=(3,), key=key)
    repo_b, _, _ = build_source(tmp_path, "b", ks=(3,), key=key, salt="b")
    served["mount"]("test/a", repo_a)
    served["mount"]("test/b", repo_b)
    want = json.loads(
        (repo_b / "attestations" / f"layer-{LAYERS[0]:03d}.k3.jsonl").read_text())
    import base64
    payload = json.loads(base64.b64decode(want["payload"]))
    sha_b0 = payload["expert_sha256"]["0"]

    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    # source order prefers a, but the content hash names b's fragment
    run(["--policy", policy, "--out", out, "--source", f"test/a@{REV}",
         "--source", f"test/b@{REV}", "--prefer-sha", sha_b0,
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)])
    report = json.loads((out / "fq-fetch-report.json").read_text())
    experts = report["experts"][str(LAYERS[0])]["k3"]
    assert experts["0"]["source"] == f"test/b@{REV}"
    assert experts["0"]["sha256"] == sha_b0
    assert experts["1"]["source"] == f"test/a@{REV}"  # unaffected: order wins
    entry = json.loads((out / "index-k3.json").read_text())[str(LAYERS[0])]
    assert entry["sources"]["0"] == f"test/b@{REV}"


def test_select_map_chooses_provider_per_expert(tmp_path, served):
    key = tmp_path / "shared.key"
    repo_a, _, pub = build_source(tmp_path, "a", ks=(3,), key=key)
    repo_b, _, _ = build_source(tmp_path, "b", ks=(3,), key=key, salt="b")
    served["mount"]("test/a", repo_a)
    served["mount"]("test/b", repo_b)
    sel = tmp_path / "select.json"
    sel.write_text(json.dumps({"schema": "fq-select/1",
                               "experts": {str(LAYERS[0]): {"2": "test/b"}}}))
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/a@{REV}",
         "--source", f"test/b@{REV}", "--select", sel,
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)])
    experts = json.loads((out / "fq-fetch-report.json").read_text())["experts"]
    per_expert = experts[str(LAYERS[0])]["k3"]
    assert per_expert["2"]["source"] == f"test/b@{REV}"
    assert per_expert["0"]["source"] == f"test/a@{REV}"


def test_resume_after_interruption_refetches_only_the_rest(tmp_path, served,
                                                           monkeypatch):
    repo, snap, pub = build_source(tmp_path, "pub")
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json",
                          {LAYERS[0]: [3] * E, LAYERS[1]: [3] * E})
    out = tmp_path / "fetched"
    real = fq_fetch.http_get_range
    state = {"n": 0}

    def flaky(url, start, end, timeout=600.0):
        if start > 8:  # a payload read, not a header read
            state["n"] += 1
            if state["n"] > 2:
                raise KeyboardInterrupt
        return real(url, start, end, timeout)

    monkeypatch.setattr(fq_fetch, "http_get_range", flaky)
    argv = ["--policy", str(policy), "--out", str(out), "--source",
            f"test/pub@{REV}", "--trust-signer", pub,
            "--trust-root", str(trust_root(tmp_path, pub)),
            "--sign-key", str(tmp_path / "local.key"),
            "--max-gap-mb", "0", "--chunk-mb", "0.001"]
    assert fq_fetch.main(argv) == 130
    partial = json.loads((out / "state.json").read_text())
    assert partial["files"], "interrupted run recorded no progress"
    assert any(f["status"] != "done" or f["done"] for f in partial["files"].values())

    monkeypatch.setattr(fq_fetch, "http_get_range", real)
    served["ranges"].clear()
    assert fq_fetch.main(argv) == 0
    done = json.loads((out / "state.json").read_text())
    assert all(f["status"] == "done" for f in done["files"].values())
    # a third run is a no-op: nothing left to fetch
    served["ranges"].clear()
    assert fq_fetch.main(argv) == 0
    assert not [r for r in served["ranges"] if r[0].endswith(".safetensors")]


def test_resumed_bytes_are_rehashed_before_being_trusted(tmp_path, served,
                                                         monkeypatch, capsys):
    repo, snap, pub = build_source(tmp_path, "pub")
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    real = fq_fetch.http_get_range
    state = {"n": 0}

    def flaky(url, start, end, timeout=600.0):
        if start > 8:  # a payload read, not a header read
            state["n"] += 1
            if state["n"] > 1:
                raise KeyboardInterrupt
        return real(url, start, end, timeout)

    argv = ["--policy", str(policy), "--out", str(out), "--source",
            f"test/pub@{REV}", "--trust-signer", pub,
            "--trust-root", str(trust_root(tmp_path, pub)),
            "--sign-key", str(tmp_path / "local.key"),
            "--max-gap-mb", "0", "--chunk-mb", "0.001"]
    monkeypatch.setattr(fq_fetch, "http_get_range", flaky)
    assert fq_fetch.main(argv) == 130
    part = next(out.glob("*.part"))
    raw = bytearray(part.read_bytes())
    body = 8 + struct.unpack("<Q", raw[:8])[0]
    raw[body] ^= 0xFF  # corrupt bytes an earlier run already fetched
    part.write_bytes(bytes(raw))
    monkeypatch.setattr(fq_fetch, "http_get_range", real)
    assert fq_fetch.main(argv) == 0
    seg = out / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    entry = json.loads((out / "index-k3.json").read_text())[str(LAYERS[0])]
    assert hashlib.sha256(seg.read_bytes()).hexdigest() == entry["sha256"]


def test_changed_recipe_discards_the_stale_partial(tmp_path, served):
    repo, snap, pub = build_source(tmp_path, "pub")
    served["mount"]("test/pub", repo)
    out = tmp_path / "fetched"
    base = ["--out", str(out), "--source", f"test/pub@{REV}",
            "--trust-signer", pub, "--trust-root", str(trust_root(tmp_path, pub)),
            "--sign-key", str(tmp_path / "local.key")]
    p1 = write_policy(tmp_path / "r1.json", {LAYERS[0]: [3, 3, 3, 3]})
    assert fq_fetch.main(["--policy", str(p1), *base]) == 0
    first = json.loads((out / "index-k3.json").read_text())[str(LAYERS[0])]
    p2 = write_policy(tmp_path / "r2.json", {LAYERS[0]: [3, 3, 4, 4]})
    assert fq_fetch.main(["--policy", str(p2), *base]) == 0
    second = json.loads((out / "index-k3.json").read_text())[str(LAYERS[0])]
    assert set(second["experts"]) == {"0", "1"}
    assert second["sha256"] != first["sha256"]
    assert set(json.loads((out / "index-k4.json").read_text())
               [str(LAYERS[0])]["experts"]) == {"2", "3"}


def test_release_manifest_is_verified_and_binds_the_indexes(tmp_path, served,
                                                            capsys):
    repo, snap, pub = build_source(tmp_path, "pub")
    key = tmp_path / "pub.key"
    assert fq_release.main([
        "build", "--dir", str(repo), "--release", "test 0.1.0",
        "--repo", "test/pub", "--revision", REV, "--sign-key", str(key)]) == 0
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)])
    assert "fq-release/1 verified" in capsys.readouterr().out

    # swapping an index after the release was signed is caught by the digest
    idx = repo / "index-k3.json"
    doc = json.loads(idx.read_text())
    doc[str(LAYERS[0])]["size"] += 1
    idx.write_text(json.dumps(doc))
    shutil.rmtree(out)
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)], expect=1)
    assert "does not match the signed release" in capsys.readouterr().err


def test_missing_expert_is_reported_not_silently_skipped(tmp_path, served, capsys):
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,))
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3, 5, 3, 3]})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)])
    assert "no source carries it" in capsys.readouterr().err
    assert set(json.loads((out / "index-k3.json").read_text())
               [str(LAYERS[0])]["experts"]) == {"0", "2", "3"}


def test_coalescing_merges_adjacent_experts_into_one_request(tmp_path, served):
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,))
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    served["ranges"].clear()
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)])
    payload_reads = [r for r in served["ranges"]
                     if r[0].endswith(".safetensors") and r[1] > 8]
    header_reads = [r for r in served["ranges"] if r[1] in (0, 8)]
    assert len(payload_reads) == 1, payload_reads  # E adjacent experts, one GET
    assert header_reads  # the segment header itself is a ranged read


def test_coalescing_never_merges_across_a_wide_gap():
    class FakeSource:
        slug = "s"

    def piece(start, end):
        return fq_fetch.Piece(0, FakeSource(), start, end, 0, "0" * 64, [])

    pieces = [piece(0, 100), piece(100, 200), piece(10_000, 10_100)]
    chunks = fq_fetch.coalesce(pieces, max_chunk=1 << 20, max_gap=1024)
    assert [(c[0], c[1]) for c in chunks] == [(0, 200), (10_000, 10_100)]
    tight = fq_fetch.coalesce(pieces, max_chunk=150, max_gap=1024)
    assert len(tight) == 3


def test_subset_is_attested_as_derived_from_its_parents(tmp_path, served):
    """A subset file is a new file, so no publisher signature can cover it.
    fq_fetch signs what it produced, naming the parents by digest."""
    import base64

    repo, snap, out, policy, pub = _fetch_all(tmp_path, served)
    local = json.loads((out / "fq-manifest.json").read_text())["signer_pubkey"]
    att = out / "attestations" / f"layer-{LAYERS[0]:03d}.k3.jsonl"
    envelope = json.loads(att.read_text())
    payload = fq_trust.verify_signature(envelope, local, where=att.name)
    assert payload["predicate"] == "derived-from"
    assert payload["derivation"]["rule"] == "range_subset_v1"
    seg = out / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    assert payload["fragment"]["sha256"] == hashlib.sha256(seg.read_bytes()).hexdigest()
    assert payload["fragment"]["size"] == seg.stat().st_size
    parent = payload["parents"][0]
    assert parent["repo"] == "test/pub" and parent["revision"] == REV
    published = json.loads(
        (repo / "attestations" / f"layer-{LAYERS[0]:03d}.k3.jsonl").read_text())
    published_payload = json.loads(base64.b64decode(published["payload"]))
    assert parent["sha256"] == published_payload["fragment"]["sha256"]
    # the per-expert digests are the publisher's, unchanged: expert bytes are
    # verbatim even though the containing file is not
    for eid, digest in payload["expert_sha256"].items():
        assert digest == published_payload["expert_sha256"][eid]
    # and the publisher's own line is kept for offline re-checking
    kept = out / "attestations" / f"test__pub@{REV[:12]}" / f"layer-{LAYERS[0]:03d}.k3.jsonl"
    assert json.loads(kept.read_text())["keyid"] == pub


def test_no_attest_leaves_an_unsigned_tree_and_says_so(tmp_path, served, capsys):
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,))
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub),
         "--no-attest"])
    assert not (out / "attestations" / f"layer-{LAYERS[0]:03d}.k3.jsonl").exists()
    assert "--insecure" in capsys.readouterr().out
