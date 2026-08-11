"""Tests for fq_fetch: the consumer range-fetch path.

A synthetic segment repo (built with fq_repack from synthetic shards) is
served over monkeypatched HTTP so every byte of remote IO is exercised
locally: ranged GETs, coalescing, signature checks, per-expert digest checks,
multi-source selection, resume, and — the end-to-end property that matters —
a fetched subset tree assembles to bytes identical to a tree that was
downloaded whole.
"""
import base64
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


def drop_header_digests(repo: Path, signer_key: Path) -> None:
    """Re-sign a repo's attestations WITHOUT fragment.header_sha256.

    fq_repack publishes that digest now, so a publisher lacking it has to be
    constructed on purpose — otherwise the "no digest" tests silently stop
    testing anything.
    """
    import base64, json as _json, sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    import fq_repack
    signer = fq_repack.Signer(signer_key)
    for att in (repo / "attestations").glob("*.jsonl"):
        out = []
        for line in att.read_text().splitlines():
            payload = _json.loads(base64.b64decode(_json.loads(line)["payload"]))
            payload["fragment"].pop("header_sha256", None)
            out.append(signer.sign_line(payload))
        att.write_text("\n".join(out) + "\n")

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


def write_policy(path: Path, mapping: dict, **extra) -> Path:
    policy = {
        "schema": "fq-policy/2",
        "bits_per_expert": {str(l): list(ks) for l, ks in mapping.items()},
        **extra,
    }
    path.write_text(json.dumps(policy))
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


def _attested_digests(repo: Path, layer: int, k: int) -> dict[str, str]:
    line = json.loads(
        (repo / "attestations" / f"layer-{layer:03d}.k{k}.jsonl").read_text())
    return json.loads(base64.b64decode(line["payload"]))["expert_sha256"]


def _copy_k_family(repo: Path, family: Path, subdir: str, k: int) -> Path:
    """Publish family documents below a root that has another K family."""
    nested = repo / subdir
    nested.mkdir(parents=True)
    shutil.copy2(family / "fq-manifest.json", nested / "fq-manifest.json")
    shutil.copy2(family / f"index-k{k}.json", nested / f"index-k{k}.json")
    (nested / "attestations").mkdir()
    for layer in LAYERS:
        name = f"layer-{layer:03d}.k{k}.safetensors"
        shutil.copy2(family / name, nested / name)
        shutil.copy2(family / "attestations" / f"{name[:-12]}.jsonl",
                     nested / "attestations" / f"{name[:-12]}.jsonl")
    return nested


def _bound_policy(path: Path, *, source: str, nested: Path,
                  family: Path) -> Path:
    layer = LAYERS[0]
    digests = _attested_digests(family, layer, 4)
    return write_policy(
        path, {layer: [4] * E},
        fetch_binding={
            "schema": "fq-fetch-binding/1",
            "providers": {
                "schema": "fq-select/1",
                "default": source,
            },
            "content": {
                "manifests": {
                    source: hashlib.sha256(
                        (nested / "fq-manifest.json").read_bytes()).hexdigest(),
                },
                "experts": {str(layer): digests},
            },
        },
    )


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


def test_fetch_binding_selects_nested_family_not_root_k4(tmp_path, served):
    """A bound primed K4 recipe cannot silently take same-K root bytes."""
    key = tmp_path / "shared.key"
    repo, _, pub = build_source(tmp_path, "pub", ks=(3, 4), key=key, salt="root")
    family, _, _ = build_source(
        tmp_path, "willfalco", ks=(4,), key=key, salt="willfalco")
    subdir = "sources/willfalco-3.42bpw/expanded"
    nested = _copy_k_family(repo, family, subdir, 4)
    nested_source = f"test/pub@{REV}:{subdir}"
    policy = _bound_policy(
        tmp_path / "recipe.json", source=nested_source, nested=nested, family=family)
    served["mount"]("test/pub", repo)

    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--source", nested_source, "--trust-signer", pub,
         "--trust-root", trust_root(tmp_path, pub)])

    expected = _attested_digests(family, LAYERS[0], 4)
    root = _attested_digests(repo, LAYERS[0], 4)
    assert root["0"] != expected["0"]  # the root K4 is deliberately unrelated
    fetched = json.loads((out / "fq-fetch-report.json").read_text())["experts"]
    selected = fetched[str(LAYERS[0])]["k4"]
    assert {record["source"] for record in selected.values()} == {nested_source}
    assert {expert: record["sha256"] for expert, record in selected.items()} == expected


def test_fetch_binding_missing_source_or_digest_fails_before_payload(
        tmp_path, served, capsys):
    key = tmp_path / "shared.key"
    repo, _, pub = build_source(tmp_path, "pub", ks=(3, 4), key=key, salt="root")
    family, _, _ = build_source(
        tmp_path, "willfalco", ks=(4,), key=key, salt="willfalco")
    subdir = "sources/willfalco-3.42bpw/expanded"
    nested = _copy_k_family(repo, family, subdir, 4)
    nested_source = f"test/pub@{REV}:{subdir}"
    policy = _bound_policy(
        tmp_path / "recipe.json", source=nested_source, nested=nested, family=family)
    served["mount"]("test/pub", repo)

    run(["--policy", policy, "--out", tmp_path / "missing-source",
         "--source", f"test/pub@{REV}", "--trust-signer", pub,
         "--trust-root", trust_root(tmp_path, pub)], expect=1)
    assert "requires source" in capsys.readouterr().err
    assert not [r for r in served["ranges"] if r[0].endswith(".safetensors")]

    broken = json.loads(policy.read_text())
    broken["fetch_binding"]["content"]["experts"][str(LAYERS[0])].pop("0")
    policy.write_text(json.dumps(broken))
    served["ranges"].clear()
    run(["--policy", policy, "--out", tmp_path / "missing-digest",
         "--source", nested_source, "--trust-signer", pub,
         "--trust-root", trust_root(tmp_path, pub)], expect=1)
    assert "no valid digest" in capsys.readouterr().err
    assert not [r for r in served["ranges"] if r[0].endswith(".safetensors")]


def test_fetch_binding_content_mismatch_aborts_other_experts_before_payload(
        tmp_path, served, capsys):
    """One bad bound digest makes the whole multi-expert recipe fail closed."""
    key = tmp_path / "shared.key"
    repo, _, pub = build_source(tmp_path, "pub", ks=(3, 4), key=key, salt="root")
    family, _, _ = build_source(
        tmp_path, "willfalco", ks=(4,), key=key, salt="willfalco")
    subdir = "sources/willfalco-3.42bpw/expanded"
    nested = _copy_k_family(repo, family, subdir, 4)
    nested_source = f"test/pub@{REV}:{subdir}"
    policy = _bound_policy(
        tmp_path / "recipe.json", source=nested_source, nested=nested, family=family)
    binding = json.loads(policy.read_text())
    binding["fetch_binding"]["content"]["experts"][str(LAYERS[0])]["0"] = (
        _attested_digests(repo, LAYERS[0], 4)["0"])
    policy.write_text(json.dumps(binding))
    served["mount"]("test/pub", repo)

    run(["--policy", policy, "--out", tmp_path / "mismatch",
         "--source", f"test/pub@{REV}", "--source", nested_source,
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)],
        expect=1)
    assert "bound recipe has unsatisfied" in capsys.readouterr().err
    assert not [r for r in served["ranges"]
                if r[0].endswith(".safetensors") and r[1] > 8]


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


# ------------------------------- plan authentication (finding P1-4c)

def _att_path(repo: Path, layer: int, k: int) -> Path:
    return repo / "attestations" / f"layer-{layer:03d}.k{k}.jsonl"


def _payload(att: Path) -> dict:
    import base64
    return json.loads(base64.b64decode(
        json.loads(att.read_text().splitlines()[0])["payload"]))


def _resign(att: Path, key: Path, mutate) -> None:
    payload = _payload(att)
    mutate(payload)
    att.write_text(fq_repack.Signer(key).sign_line(payload) + "\n")


def _attest_header_digests(repo: Path, key: Path, ks=KS) -> None:
    """Publish fragment.header_sha256 the way a publisher should, so the
    consumer can authenticate the header without reading the payload."""
    for k in ks:
        for layer in LAYERS:
            att = _att_path(repo, layer, k)
            if not att.exists():
                continue
            seg = repo / f"layer-{layer:03d}.k{k}.safetensors"
            raw = seg.read_bytes()
            body = 8 + struct.unpack("<Q", raw[:8])[0]
            _resign(att, key, lambda p, raw=raw, body=body: p["fragment"].update(
                {"header_sha256": hashlib.sha256(raw[:body]).hexdigest(),
                 "body_offset": body}))


def _retag_header(seg: Path, old: bytes, new: bytes) -> None:
    """Rewrite the segment's safetensors header in place, keeping its length
    (and every payload byte) identical — the substitution a ranged header
    read cannot detect on its own."""
    assert len(old) == len(new)
    raw = bytearray(seg.read_bytes())
    hlen = struct.unpack("<Q", raw[:8])[0]
    head = bytes(raw[8:8 + hlen])
    assert old in head, old
    raw[8:8 + hlen] = head.replace(old, new, 1)
    seg.write_bytes(bytes(raw))


def test_plan_records_which_authenticated_inputs_it_came_from(tmp_path, served):
    """The positive path: the header is proven, and the tree says how."""
    repo, snap, out, policy, pub = _fetch_all(tmp_path, served)
    report = json.loads((out / "fq-fetch-report.json").read_text())
    assert report["trust"]["plan_authenticated"] is True
    per_file = report["header_authentication"][f"layer-{LAYERS[0]:03d}.k3.safetensors"]
    prov = next(iter(per_file.values()))
    # fq_repack publishes fragment.header_sha256, so the cheap attested
    # path applies; the full-fragment fallback is covered by the
    # no-digest test above.
    assert prov["method"] == fq_fetch.AUTH_HEADER_DIGEST
    assert prov["authenticated"] is True
    local = json.loads((out / "fq-manifest.json").read_text())["signer_pubkey"]
    att = out / "attestations" / f"layer-{LAYERS[0]:03d}.k3.jsonl"
    payload = fq_trust.verify_signature(json.loads(att.read_text()), local,
                                        where=att.name)
    parent = payload["parents"][0]
    assert parent["header_authentication"] == fq_fetch.AUTH_HEADER_DIGEST
    assert parent["header_authenticated"] is True
    assert payload["verification"]["plan_inputs"]["plan_authenticated"] is True
    # the subset we produced publishes its own header digest, so a consumer
    # fetching from this tree gets the cheap authenticated path
    seg = out / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    raw = seg.read_bytes()
    body = 8 + struct.unpack("<Q", raw[:8])[0]
    assert payload["fragment"]["header_sha256"] == hashlib.sha256(
        raw[:body]).hexdigest()


def test_tampered_segment_header_is_refused(tmp_path, served, capsys):
    """The finding: names/dtypes/shapes/offsets came from an unauthenticated
    ranged read.  Relabel a dtype without touching one payload byte — every
    per-expert digest still matches — and the fetch must still refuse."""
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,))
    served["mount"]("test/pub", repo)
    _retag_header(repo / f"layer-{LAYERS[0]:03d}.k3.safetensors", b'"I16"', b'"F16"')
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)],
        expect=1)
    err = capsys.readouterr().err
    assert "TRUST FAILURE" in err and "signed attestation" in err
    assert not list(out.glob("*.safetensors"))


def test_attested_header_digest_authenticates_without_the_full_download(
        tmp_path, served, capsys):
    """When the publisher signs fragment.header_sha256, the plan is proven
    from a header-sized read — no whole fragment, no re-download."""
    key = tmp_path / "pub.key"
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,), key=key)
    _attest_header_digests(repo, key, ks=(3,))
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    seg = repo / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    served["ranges"].clear()
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)])
    report = json.loads((out / "fq-fetch-report.json").read_text())
    prov = next(iter(report["header_authentication"][seg.name].values()))
    assert prov["method"] == fq_fetch.AUTH_HEADER_DIGEST
    assert prov["authenticated"] is True
    # nothing read the whole fragment
    biggest = max(r[2] - r[1] for r in served["ranges"] if r[0] == seg.name)
    assert biggest < seg.stat().st_size
    assert "attested-header-digest" in capsys.readouterr().out


def test_attested_header_digest_catches_a_tampered_header(tmp_path, served,
                                                          capsys):
    key = tmp_path / "pub.key"
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,), key=key)
    _attest_header_digests(repo, key, ks=(3,))
    served["mount"]("test/pub", repo)
    _retag_header(repo / f"layer-{LAYERS[0]:03d}.k3.safetensors", b'"I16"', b'"F16"')
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)],
        expect=1)
    err = capsys.readouterr().err
    assert "not the header the publisher signed" in err
    assert not list(out.glob("*.safetensors"))


def test_full_header_fallback_reauthenticates_cache_and_reuses_pass_bytes(
        tmp_path, served):
    """A cached full-file claim is never a substitute for re-hashing source."""
    key = tmp_path / "pub.key"
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,), key=key)
    drop_header_digests(repo, key)
    assert fq_release.main([
        "build", "--dir", str(repo), "--release", "test 0.1.0",
        "--repo", "test/pub", "--revision", REV, "--sign-key", str(key)]) == 0
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3, 4, 4, 4]})
    out = tmp_path / "fetched"
    argv = ["--policy", str(policy), "--out", str(out), "--source",
            f"test/pub@{REV}", "--trust-signer", pub,
            "--trust-root", str(trust_root(tmp_path, pub))]
    seg = repo / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    run(argv)
    raw = seg.read_bytes()
    body = 8 + struct.unpack("<Q", raw[:8])[0]
    segment_reads = [r for r in served["ranges"] if r[0] == seg.name]
    # Full authentication needs the source once, plus the tiny draft header:
    # selected bytes from that verified pass are spooled to disk instead of
    # being fetched again.
    assert sum(end - start for _, start, end in segment_reads) <= len(raw) + body

    cache = next((out / ".fq-fetch-cache").rglob(f"hdr-{seg.name}.json"))
    cached = json.loads(cache.read_text())
    assert cached["authentication"]["method"] == fq_fetch.AUTH_FULL_FRAGMENT
    assert cached["authentication"]["release_manifest"] is True

    # A full-file digest cannot authenticate a separately persisted prefix:
    # it is self-reported and forgeable.  Restart pricing therefore includes
    # a fresh full rehash, and a completed-output skip leaves no spool files.
    plan_path = tmp_path / "restart-plan.json"
    run([*argv, "--dry-run", "--json", plan_path])
    plan = json.loads(plan_path.read_text())
    assert plan["header_authentication"]["extra_bytes"] == len(raw)
    assert plan["header_authentication"]["methods"][fq_fetch.AUTH_FULL_FRAGMENT] == 1
    run(argv)
    assert not list((out / ".fq-fetch-cache" / "test__pub@0123456789ab" /
                     ".verified-pieces").glob("*"))

    # Forge every field in a persisted full-proof record that is locally
    # editable.  The next process still hashes the pinned fragment, detects
    # the substituted header during re-planning, and cleans its error spool.
    forged = bytearray(base64.b64decode(cached["header_bytes"]))
    assert b'"I16"' in forged
    forged = bytes(forged).replace(b'"I16"', b'"F16"', 1)
    cached["header_bytes"] = base64.b64encode(forged).decode()
    cached["header"] = json.loads(forged[8:])
    cached["authentication"]["header_sha256"] = hashlib.sha256(forged).hexdigest()
    cache.write_text(json.dumps(cached))
    served["ranges"].clear()
    run(argv, expect=1)
    reread = [r for r in served["ranges"] if r[0] == seg.name]
    assert sum(end - start for _, start, end in reread) >= len(raw)
    assert not list((out / ".fq-fetch-cache" / "test__pub@0123456789ab" /
                     ".verified-pieces").glob("*"))


def test_signed_header_cache_reduces_restart_authentication_cost(tmp_path, served):
    key = tmp_path / "pub.key"
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,), key=key)
    assert fq_release.main([
        "build", "--dir", str(repo), "--release", "test 0.1.0",
        "--repo", "test/pub", "--revision", REV, "--sign-key", str(key)]) == 0
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3, 4, 4, 4]})
    out = tmp_path / "fetched"
    argv = ["--policy", str(policy), "--out", str(out), "--source",
            f"test/pub@{REV}", "--trust-signer", pub,
            "--trust-root", str(trust_root(tmp_path, pub))]
    run(argv)
    plan_path = tmp_path / "restart-plan.json"
    run([*argv, "--dry-run", "--json", plan_path])
    auth = json.loads(plan_path.read_text())["header_authentication"]
    assert auth["extra_bytes"] == 0
    assert auth["methods"]["cached"] == 1


def test_done_output_is_rehashed_before_it_is_resigned(tmp_path, served):
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,))
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    argv = ["--policy", str(policy), "--out", str(out), "--source",
            f"test/pub@{REV}", "--trust-signer", pub,
            "--trust-root", str(trust_root(tmp_path, pub))]
    run(argv)
    seg = out / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    _retag_header(seg, b'"I16"', b'"F16"')
    raw = bytearray(seg.read_bytes())
    body = 8 + struct.unpack("<Q", raw[:8])[0]
    # State is attacker controlled too: even a matching forged digest must
    # not let a modified completed file bypass the per-expert/header checks.
    state_path = out / "state.json"
    state = json.loads(state_path.read_text())
    entry = state["files"][seg.name]["entry"]
    entry["sha256"] = hashlib.sha256(raw).hexdigest()
    entry["header_sha256"] = hashlib.sha256(raw[:body]).hexdigest()
    state_path.write_text(json.dumps(state))

    served["ranges"].clear()
    run(argv)
    fixed = seg.read_bytes()
    assert fixed != bytes(raw)
    entry = json.loads((out / "index-k3.json").read_text())[str(LAYERS[0])]
    assert hashlib.sha256(fixed).hexdigest() == entry["sha256"]
    assert any(name == seg.name for name, _, _ in served["ranges"])


def test_done_output_header_length_is_bounded_by_authenticated_plan(tmp_path,
                                                                    served):
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,))
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    argv = ["--policy", str(policy), "--out", str(out), "--source",
            f"test/pub@{REV}", "--trust-signer", pub,
            "--trust-root", str(trust_root(tmp_path, pub))]
    run(argv)
    seg = out / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    raw = bytearray(seg.read_bytes())
    raw[:8] = struct.pack("<Q", 1 << 62)
    seg.write_bytes(raw)
    served["ranges"].clear()
    run(argv)
    fixed = seg.read_bytes()
    assert struct.unpack("<Q", fixed[:8])[0] < len(fixed)
    assert any(name == seg.name for name, _, _ in served["ranges"])


def test_header_trust_attested_refuses_when_no_digest_is_published(tmp_path,
                                                                   served,
                                                                   capsys):
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,))
    drop_header_digests(repo, tmp_path / "pub.key")
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub),
         "--header-trust", "attested"], expect=1)
    err = capsys.readouterr().err
    assert "no signed fragment.header_sha256" in err
    assert not list(out.glob("*.safetensors"))


def test_header_trust_unsafe_is_loud_and_recorded(tmp_path, served, capsys):
    """The escape hatch stays available, but it is never silent."""
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,))
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub),
         "--header-trust", "unsafe"])
    assert "UNAUTHENTICATED ranged read" in capsys.readouterr().err
    report = json.loads((out / "fq-fetch-report.json").read_text())
    assert report["trust"]["plan_authenticated"] is False
    local = json.loads((out / "fq-manifest.json").read_text())["signer_pubkey"]
    att = out / "attestations" / f"layer-{LAYERS[0]:03d}.k3.jsonl"
    payload = fq_trust.verify_signature(json.loads(att.read_text()), local,
                                        where=att.name)
    assert payload["parents"][0]["header_authentication"] == fq_fetch.AUTH_NONE
    assert payload["parents"][0]["header_authenticated"] is False
    assert payload["verification"]["plan_inputs"]["plan_authenticated"] is False


def test_release_manifest_and_attestation_must_agree(tmp_path, served, capsys):
    """Two signatures by the same publisher over the same fragment: if they
    disagree, one of them is a rollback and neither is preferred."""
    key = tmp_path / "pub.key"
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,), key=key)
    assert fq_release.main([
        "build", "--dir", str(repo), "--release", "test 0.1.0",
        "--repo", "test/pub", "--revision", REV, "--sign-key", str(key)]) == 0
    # re-sign the attestation so it claims a different fragment digest, and
    # re-sign the release over the new attestation file: both signatures are
    # valid and current, and they contradict each other about the fragment
    att = _att_path(repo, LAYERS[0], 3)
    _resign(att, key, lambda p: p["fragment"].update({"sha256": "b" * 64}))
    import base64
    rel = json.loads((repo / "fq-release.json").read_text())
    payload = json.loads(base64.b64decode(rel["payload"]))
    payload["files"][f"attestations/{att.name}"] = {
        "sha256": hashlib.sha256(att.read_bytes()).hexdigest(),
        "size": att.stat().st_size}
    (repo / "fq-release.json").write_text(
        fq_repack.Signer(key).sign_line(payload) + "\n")
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)],
        expect=1)
    err = capsys.readouterr().err
    assert "signatures disagree" in err or "not listed in the signed release" in err
    assert not list(out.glob("*.safetensors"))


def test_release_manifest_covers_the_fetched_fragments(tmp_path, served, capsys):
    """A fragment absent from the signed release is not fetched from a repo
    that publishes one."""
    key = tmp_path / "pub.key"
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,), key=key)
    assert fq_release.main([
        "build", "--dir", str(repo), "--release", "test 0.1.0",
        "--repo", "test/pub", "--revision", REV, "--sign-key", str(key)]) == 0
    rel = json.loads((repo / "fq-release.json").read_text())
    import base64
    payload = json.loads(base64.b64decode(rel["payload"]))
    payload["files"].pop(f"layer-{LAYERS[0]:03d}.k3.safetensors")
    (repo / "fq-release.json").write_text(
        fq_repack.Signer(key).sign_line(payload) + "\n")
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)],
        expect=1)
    assert "not listed in the signed release" in capsys.readouterr().err


def test_every_attestation_line_is_considered(tmp_path, served):
    """JSON Lines: a stranger's countersignature on line 1 must not hide the
    publisher's line on line 2."""
    key = tmp_path / "pub.key"
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,), key=key)
    stranger = fq_repack.Signer(tmp_path / "stranger.key")
    for layer in LAYERS:
        att = _att_path(repo, layer, 3)
        good = att.read_text().splitlines()[0]
        payload = _payload(att)
        att.write_text(stranger.sign_line(payload) + "\n" + good + "\n")
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)])
    assert (out / f"layer-{LAYERS[0]:03d}.k3.safetensors").exists()


def test_only_untrusted_lines_is_a_refusal(tmp_path, served, capsys):
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,))
    stranger = fq_repack.Signer(tmp_path / "stranger.key")
    for layer in LAYERS:
        att = _att_path(repo, layer, 3)
        att.write_text(stranger.sign_line(_payload(att)) + "\n")
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)],
        expect=1)
    err = capsys.readouterr().err
    assert "no trusted attestation line" in err or "no source carries it" in err
    assert not list(out.glob("*.safetensors"))


def test_absent_attestation_file_is_a_refusal(tmp_path, served, capsys):
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,))
    for att in (repo / "attestations").glob("*.jsonl"):
        att.unlink()
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)],
        expect=1)
    err = capsys.readouterr().err
    assert "cannot verify fetched bytes" in err or "no source carries it" in err
    assert not list(out.glob("*.safetensors"))


def test_missing_expert_digest_is_never_fetched(tmp_path, served, capsys):
    """A signed line that omits an expert: those bytes are unverifiable, so
    they are refused rather than fetched and signed."""
    key = tmp_path / "pub.key"
    repo, snap, pub = build_source(tmp_path, "pub", ks=(3,), key=key)
    _resign(_att_path(repo, LAYERS[0], 3), key,
            lambda p: p["expert_sha256"].pop("1"))
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3] * E})
    out = tmp_path / "fetched"
    run(["--policy", policy, "--out", out, "--source", f"test/pub@{REV}",
         "--trust-signer", pub, "--trust-root", trust_root(tmp_path, pub)])
    assert "no attested digest" in capsys.readouterr().err
    index = json.loads((out / "index-k3.json").read_text())
    assert "1" not in index[str(LAYERS[0])]["experts"]


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
