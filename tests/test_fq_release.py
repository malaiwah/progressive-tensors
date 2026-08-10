"""Tests for fq_release: one signature over a whole release.

What the release manifest has to buy, and therefore what is tested here:
a single verification covers the complete file set, catches substitution of
any covered file (including the attestation files whose digests everything
else leans on), tolerates a deliberately partial tree, refuses when the
signer is not the pinned one, and — under --complete — refuses a tree that
has grown an extra file the signature does not cover.

The `publish` half is tested against a fake hub: what matters there is not
HTTP, it is that the whole coherent set goes up as ONE commit pinned to the
parent we read, that a concurrent writer causes a rejection and a rebuild
rather than an interleave, and that files only the remote has are not left
sitting outside the signature by accident.
"""
import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_release  # noqa: E402
import fq_repack  # noqa: E402
import fq_trust  # noqa: E402
from test_fq_repack import LAYERS, write_shard  # noqa: E402


@pytest.fixture()
def tree(tmp_path):
    """A small signed segment tree plus its trust root."""
    snap = tmp_path / "snap"
    snap.mkdir()
    for i, layer in enumerate(LAYERS):
        write_shard(snap / f"model-layer-{layer:03d}.safetensors", layer, scramble=bool(i))
    out = tmp_path / "segments"
    key = tmp_path / "sign.key"
    assert fq_repack.main([
        "--snapshot", str(snap), "--source-repo", "test/src", "--revision", "cafe",
        "--base-model", "test/base", "--out", str(out), "--sign-key", str(key)]) == 0
    pub = json.loads((out / "fq-manifest.json").read_text())["signer_pubkey"]
    root = tmp_path / "FINGERPRINTS"
    root.write_text(f"{pub}  test-signer  active  2026-08-10  segments\n")
    return out, key, pub, root


def build(tree, **kw):
    out, key, pub, root = tree
    argv = ["build", "--dir", str(out), "--release", "test 0.1.0",
            "--sign-key", str(key)]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    assert fq_release.main(argv) == 0
    return out / "fq-release.json"


def verify(tree, target=None, expect=0, **flags):
    out, key, pub, root = tree
    argv = ["verify", "--dir", str(target or out), "--trust-signer", pub,
            "--trust-root", str(root)]
    if (target or out) != out:
        argv += ["--release", str(out / "fq-release.json")]
    for k, v in flags.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    assert fq_release.main(argv) == expect
    return expect


def test_build_covers_every_published_file(tree):
    out = tree[0]
    rel = json.loads(build(tree).read_text())
    payload = json.loads(base64.b64decode(rel["payload"]))
    assert payload["schema"] == "fq-release/1"
    names = set(payload["files"])
    for layer in LAYERS:
        assert f"layer-{layer:03d}.k3.safetensors" in names
        assert f"attestations/layer-{layer:03d}.k3.jsonl" in names
    assert {"index-k3.json", "fq-manifest.json"} <= names
    assert "state.json" not in names  # local bookkeeping is not a release
    assert payload["manifest_sha256"] == hashlib.sha256(
        (out / "fq-manifest.json").read_bytes()).hexdigest()
    for name, meta in payload["files"].items():
        assert (out / name).stat().st_size == meta["size"]


def test_verify_round_trip(tree, capsys):
    build(tree)
    verify(tree)
    seen = capsys.readouterr().out
    assert "pinned signer" in seen and "0 MISMATCHED" in seen


def test_one_signature_catches_a_swapped_fragment(tree, capsys):
    out = tree[0]
    build(tree)
    seg = out / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    raw = bytearray(seg.read_bytes())
    raw[-1] ^= 0xFF
    seg.write_bytes(bytes(raw))
    verify(tree, expect=1)
    assert "MISMATCH" in capsys.readouterr().err


def test_one_signature_also_covers_the_attestation_files(tree, capsys):
    """The chain that makes per-expert digests trustworthy without checking
    every attestation signature separately."""
    out = tree[0]
    build(tree)
    att = out / "attestations" / f"layer-{LAYERS[0]:03d}.k3.jsonl"
    line = json.loads(att.read_text())
    payload = json.loads(base64.b64decode(line["payload"]))
    first = sorted(payload["expert_sha256"])[0]
    payload["expert_sha256"][first] = "0" * 64      # lie about an expert digest
    line["payload"] = base64.b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).decode()
    att.write_text(json.dumps(line) + "\n")
    verify(tree, expect=1)
    assert "MISMATCH" in capsys.readouterr().err


def test_added_file_is_reported_as_uncovered(tree, capsys):
    out = tree[0]
    build(tree)
    (out / "layer-099.k3.safetensors").write_bytes(b"surprise")
    verify(tree)  # not fatal: it is simply not part of the release
    assert "unlisted" in capsys.readouterr().err


def test_partial_tree_is_fine_unless_completeness_is_demanded(tree, tmp_path, capsys):
    out = tree[0]
    build(tree)
    partial = tmp_path / "partial"
    (partial / "attestations").mkdir(parents=True)
    for name in ("fq-manifest.json", "index-k3.json",
                 f"layer-{LAYERS[0]:03d}.k3.safetensors"):
        (partial / name).write_bytes((out / name).read_bytes())
    verify(tree, target=partial)
    assert "absent" in capsys.readouterr().out
    verify(tree, target=partial, expect=1, complete=True)


def test_wrong_signer_is_refused(tree, tmp_path, capsys):
    out, key, pub, root = tree
    build(tree)
    other = "ab" * 32
    assert fq_release.main([
        "verify", "--dir", str(out), "--trust-signer", other,
        "--trust-root", str(root)]) == 1
    assert "TRUST FAILURE" in capsys.readouterr().err


def test_resigned_release_by_another_key_does_not_pass_pinning(tree, tmp_path, capsys):
    """Repo compromise: attacker rewrites the bytes AND re-signs the release
    with their own key.  Everything is internally consistent; pinning is the
    only thing that says no."""
    out, key, pub, root = tree
    build(tree)
    seg = out / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    seg.write_bytes(seg.read_bytes() + b"evil")
    evil_key = tmp_path / "evil.key"
    assert fq_release.main([
        "build", "--dir", str(out), "--release", "test 0.1.0",
        "--sign-key", str(evil_key)]) == 0
    # self-consistent under the attacker's key
    evil_pub = json.loads(fq_repack.Signer(evil_key).pub_hex)  \
        if False else fq_repack.Signer(evil_key).pub_hex
    assert fq_release.main([
        "verify", "--dir", str(out), "--trust-signer", evil_pub,
        "--trust-root", str(root)]) == 0
    # and refused against the fingerprint published in git
    assert fq_release.main([
        "verify", "--dir", str(out), "--trust-signer", pub,
        "--trust-root", str(root)]) == 1
    assert "trusted signer" in capsys.readouterr().err


def test_json_report(tree, tmp_path):
    build(tree)
    out = tree[0]
    report = tmp_path / "r.json"
    verify(tree, json=report)
    data = json.loads(report.read_text())
    assert data["ok"] and data["rung"] == fq_trust.RUNG_PINNED
    assert not data["bad"] and data["present"]


# ------------------------------------------------- --complete is enforceable

def test_complete_fails_on_an_unlisted_file(tree, capsys):
    """The "no silent additions" guarantee has to be a non-zero exit, not a
    line of stderr a script will never read."""
    out = tree[0]
    build(tree)
    (out / "layer-099.k3.safetensors").write_bytes(b"surprise")
    verify(tree, expect=1, complete=True)
    err = capsys.readouterr().err
    assert "UNLISTED" in err and "layer-099.k3.safetensors" in err


def test_complete_passes_on_an_exact_tree(tree):
    build(tree)
    verify(tree, complete=True)


def test_check_tree_reports_extras_separately_from_missing(tree, tmp_path):
    out = tree[0]
    rel = json.loads(build(tree).read_text())
    payload = json.loads(base64.b64decode(rel["payload"]))
    (out / "unexpected.safetensors").write_bytes(b"x")
    report = fq_release.check_tree(payload, out, complete=True)
    assert report["extra"] == ["unexpected.safetensors"]
    assert not report["missing"] and not report["bad"]
    assert report["ok"] is False
    assert fq_release.check_tree(payload, out, complete=False)["ok"] is True


# --------------------------------------------------------- the digest cache

def test_digest_cache_avoids_rereading_unchanged_files(tree, tmp_path):
    out = tree[0]
    cache_path = tmp_path / "digests.json"
    cache = fq_release.DigestCache(cache_path)
    first = fq_release.build_payload(out, release="r", repo=None, revision=None,
                                     cache=cache)
    cache.save()
    assert cache.misses == first["counts"]["files"] and cache.hits == 0

    warm = fq_release.DigestCache(cache_path)
    second = fq_release.build_payload(out, release="r", repo=None, revision=None,
                                      cache=warm)
    assert warm.hits == first["counts"]["files"] and warm.misses == 0
    assert second["files"] == first["files"]


def test_digest_cache_notices_a_rewritten_file(tree, tmp_path):
    out = tree[0]
    cache_path = tmp_path / "digests.json"
    cache = fq_release.DigestCache(cache_path)
    before = fq_release.build_payload(out, release="r", repo=None, revision=None,
                                      cache=cache)
    cache.save()
    victim = f"layer-{LAYERS[0]:03d}.k3.safetensors"
    (out / victim).write_bytes(b"different bytes entirely")
    warm = fq_release.DigestCache(cache_path)
    after = fq_release.build_payload(out, release="r", repo=None, revision=None,
                                     cache=warm)
    assert after["files"][victim] != before["files"][victim]


def test_a_corrupt_cache_is_just_a_cold_cache(tmp_path):
    p = tmp_path / "digests.json"
    p.write_text("{not json")
    assert fq_release.DigestCache(p).entries == {}


# ------------------------------------------------------------ publish: fake hub

class FakeCommitAdd:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class FakeCommitDelete:
    def __init__(self, path_in_repo):
        self.path_in_repo = path_in_repo


class FakeHTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"http {status}")
        self.response = type("R", (), {"status_code": status})()


class FakeApi:
    """A hub that can be made to move under the publisher's feet."""

    def __init__(self, heads, remote_files, reject_first=0):
        self.heads = list(heads)          # sha seen by each successive read
        self.remote_files = remote_files  # {path: {size, lfs_sha256, blob_id}}
        self.reject_first = reject_first
        self.commits = []
        self.reads = 0

    def __call__(self, *a, **kw):         # HfApi(token=...) -> self
        return self

    def repo_info(self, repo_id, repo_type=None, revision=None, files_metadata=False):
        sha = self.heads[min(self.reads, len(self.heads) - 1)]
        self.reads += 1
        siblings = [
            type("S", (), {"rfilename": p, "size": m.get("size"),
                           "lfs": ({"sha256": m["lfs_sha256"]}
                                   if m.get("lfs_sha256") else None),
                           "blob_id": m.get("blob_id")})()
            for p, m in self.remote_files.items()]
        return type("I", (), {"sha": sha, "siblings": siblings})()

    def create_commit(self, *, repo_id, repo_type, revision, operations,
                      commit_message, commit_description=None, parent_commit=None):
        if len(self.commits) < self.reject_first:
            self.commits.append(("rejected", parent_commit))
            raise FakeHTTPError(412)
        self.commits.append(("ok", parent_commit, operations))
        return type("C", (), {"oid": "newsha" + str(len(self.commits))})()


@pytest.fixture()
def fake_hub(monkeypatch):
    """Install a fake huggingface_hub for the duration of a test."""
    def install(api):
        mod = type(sys)("huggingface_hub")
        mod.HfApi = api
        mod.CommitOperationAdd = FakeCommitAdd
        mod.CommitOperationDelete = FakeCommitDelete
        errors = type(sys)("huggingface_hub.errors")
        errors.HfHubHTTPError = FakeHTTPError
        mod.errors = errors
        monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
        return api
    return install


def publish(tree, api, fake_hub, expect=0, **flags):
    out, key, pub, root = tree
    fake_hub(api)
    argv = ["publish", "--dir", str(out), "--repo", "t/repo",
            "--release", "test 0.1.0", "--sign-key", str(key)]
    for k, v in flags.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    assert fq_release.main(argv) == expect


def test_publish_pushes_one_commit_pinned_to_the_parent_it_read(tree, fake_hub):
    out = tree[0]
    api = FakeApi(heads=["parent-sha"], remote_files={})
    publish(tree, api, fake_hub)
    assert len(api.commits) == 1
    status, parent, ops = api.commits[0]
    assert status == "ok" and parent == "parent-sha"
    names = {o.path_in_repo for o in ops}
    # the whole coherent set, and the signed manifest, in the same commit
    assert fq_release.RELEASE_FILE in names
    for layer in LAYERS:
        assert f"layer-{layer:03d}.k3.safetensors" in names
        assert f"attestations/layer-{layer:03d}.k3.jsonl" in names
    assert {"fq-manifest.json", "index-k3.json"} <= names


def test_published_release_records_the_parent_it_was_built_against(tree, fake_hub):
    api = FakeApi(heads=["parent-sha"], remote_files={})
    publish(tree, api, fake_hub)
    ops = api.commits[0][2]
    blob = next(o for o in ops if o.path_in_repo == fq_release.RELEASE_FILE)
    envelope = json.loads(blob.path_or_fileobj)
    payload = json.loads(base64.b64decode(envelope["payload"]))
    assert payload["parent_revision"] == "parent-sha"
    assert payload["repo"] == "t/repo"


def test_a_concurrent_writer_causes_a_rebuild_not_an_interleave(tree, fake_hub):
    """The remote moves between our read and our push: the push must be
    rejected, and the next attempt must be built against the NEW head."""
    api = FakeApi(heads=["head-1", "head-2"], remote_files={}, reject_first=1)
    publish(tree, api, fake_hub)
    assert [c[1] for c in api.commits] == ["head-1", "head-2"]
    assert api.commits[0][0] == "rejected" and api.commits[1][0] == "ok"
    payload = json.loads(base64.b64decode(json.loads(
        next(o for o in api.commits[1][2]
             if o.path_in_repo == fq_release.RELEASE_FILE).path_or_fileobj
    )["payload"]))
    assert payload["parent_revision"] == "head-2"


def test_publish_gives_up_after_the_retry_budget(tree, fake_hub, capsys):
    api = FakeApi(heads=["h1", "h2", "h3"], remote_files={}, reject_first=99)
    publish(tree, api, fake_hub, expect=1, max_attempts=2)
    assert len(api.commits) == 2
    assert "giving up after 2 attempts" in capsys.readouterr().err


def test_publish_refuses_when_the_remote_holds_files_the_release_omits(
        tree, fake_hub, capsys):
    api = FakeApi(heads=["h1"],
                  remote_files={"layer-900.k9.safetensors":
                                {"size": 4, "lfs_sha256": "0" * 64, "blob_id": None}})
    publish(tree, api, fake_hub, expect=1)
    err = capsys.readouterr().err
    assert "REFUSING TO PUBLISH" in err and "layer-900.k9.safetensors" in err
    assert not api.commits


def test_prune_removes_the_remote_only_files_in_the_same_commit(tree, fake_hub):
    api = FakeApi(heads=["h1"],
                  remote_files={"layer-900.k9.safetensors":
                                {"size": 4, "lfs_sha256": "0" * 64, "blob_id": None}})
    publish(tree, api, fake_hub, prune=True)
    ops = api.commits[0][2]
    deletes = [o.path_in_repo for o in ops if isinstance(o, FakeCommitDelete)]
    assert deletes == ["layer-900.k9.safetensors"]


def test_publish_skips_uploading_bytes_the_remote_already_has(tree, fake_hub):
    """Re-publishing an unchanged tree must cost one small file, not the repo."""
    out = tree[0]
    api0 = FakeApi(heads=["h1"], remote_files={})
    publish(tree, api0, fake_hub)
    listed = {}
    for o in api0.commits[0][2]:
        if o.path_in_repo == fq_release.RELEASE_FILE:
            continue
        p = Path(o.path_or_fileobj)
        data = p.read_bytes()
        listed[o.path_in_repo] = {
            "size": len(data),
            "lfs_sha256": hashlib.sha256(data).hexdigest()
            if p.suffix == ".safetensors" else None,
            "blob_id": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
        }
    api1 = FakeApi(heads=["h2"], remote_files=listed)
    publish(tree, api1, fake_hub)
    ops = api1.commits[0][2]
    assert [o.path_in_repo for o in ops] == [fq_release.RELEASE_FILE]


def test_dry_run_pushes_nothing(tree, fake_hub, capsys):
    api = FakeApi(heads=["h1"], remote_files={})
    publish(tree, api, fake_hub, dry_run=True)
    assert not api.commits
    assert "nothing pushed" in capsys.readouterr().out
