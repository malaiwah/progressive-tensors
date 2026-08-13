#!/usr/bin/env python3
"""Promote a fully uploaded campaign from a staging branch to main, atomically.

A campaign is thousands of files across two bases, several stage trees and the
signed plans. Uploading them to `main` directly means many commits, and a
consumer who fetches between two of them gets a family that does not exist: a
base whose skeleton is missing, or a stage whose parent is not published yet.

So the campaign is uploaded to a staging branch and promoted here in **one
commit**, built from server-side copy operations. Nothing is re-uploaded: each
operation names a path and the revision to copy it from, so promotion costs one
API call regardless of the terabyte behind it.

    fq-promote-campaign --repo malaiwah/GLM-5.2-MSRT --campaign /data/glm52-msrt

The check is the point. It compares the *whole* finalized campaign directory on
disk -- every payload shard, digest sidecar and attestation -- against the file
list and sizes on the staging branch, and refuses to promote unless they match
exactly. A promotion that publishes 14 metadata files and no fragments is atomic
and useless.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import CommitOperationCopy, HfApi

# What a campaign publishes. An allow-list, not a blacklist: a campaign directory
# also accumulates operational files -- logs, locks, the sentinel, plan JSON, the
# source digest cache, campaign.env, the driver's window state -- and a new one
# appearing must not silently become a required upload or a spurious mismatch.
# `skeleton/` is absent on purpose: finalize hardlinks its shards into each base,
# which is what makes a base loadable, so publishing it separately would upload
# the same bytes twice.
PUBLISH_DIRS = ("base", "stages", "assemblies")
PUBLISH_FILES = ("campaign_summary.json",)


def campaign_files(campaign: Path) -> dict[str, int]:
    """Every path a finalized campaign publishes, relative to the repo root."""
    out: dict[str, int] = {}
    for name in PUBLISH_FILES:
        path = campaign / name
        if path.is_file():
            out[name] = path.stat().st_size
    for directory in PUBLISH_DIRS:
        for path in (campaign / directory).rglob("*"):
            if path.is_file():
                out[path.relative_to(campaign).as_posix()] = path.stat().st_size
    return out


def staged(api: HfApi, repo: str, revision: str) -> dict[str, int]:
    """path -> size on the staging revision, for every non-hidden blob."""
    out: dict[str, int] = {}
    for entry in api.list_repo_tree(repo_id=repo, revision=revision,
                                    recursive=True, expand=True):
        size = getattr(entry, "size", None)
        if size is None:                      # a directory
            continue
        if entry.path.startswith(".") or entry.path.startswith("_"):
            continue
        lfs = getattr(entry, "lfs", None)
        out[entry.path] = getattr(lfs, "size", None) or size
    return out


def compare(local: dict[str, int], remote: dict[str, int]) -> list[str]:
    problems = []
    missing = sorted(set(local) - set(remote))
    if missing:
        problems.append(
            f"{len(missing)} campaign files are not on the branch, e.g. "
            f"{missing[:5]}")
    extra = sorted(set(remote) - set(local))
    if extra:
        problems.append(
            f"{len(extra)} files on the branch are not part of this campaign, "
            f"e.g. {extra[:5]}; promote from a branch used by one campaign only")
    truncated = sorted(path for path in set(local) & set(remote)
                       if local[path] != remote[path])
    if truncated:
        problems.append(
            f"{len(truncated)} files differ in size from the local campaign, "
            f"e.g. {[(p, local[p], remote[p]) for p in truncated[:3]]}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="destination repo id")
    parser.add_argument("--campaign", required=True, type=Path,
                        help="the finalized campaign directory this publishes")
    parser.add_argument("--from", dest="source", default="staging",
                        help="staging branch holding the uploaded campaign")
    parser.add_argument("--to", dest="target", default="main")
    parser.add_argument("--check", action="store_true",
                        help="verify the branch only; never touch --to")
    args = parser.parse_args(argv)

    if not (args.campaign / "campaign_summary.json").is_file():
        print(f"error: {args.campaign} has no campaign_summary.json; finalize "
              f"it before publishing", file=sys.stderr)
        return 2

    api = HfApi()
    # Resolve first: comparing the branch name and then copying from a SHA
    # resolved afterwards would leave exactly the window a concurrent upload
    # needs to get unverified bytes onto main.
    refs = api.list_repo_refs(repo_id=args.repo)
    pinned = next((branch.target_commit for branch in refs.branches
                   if branch.name == args.source), None)
    if pinned is None:
        print(f"error: {args.repo} has no branch {args.source}", file=sys.stderr)
        return 2
    local = campaign_files(args.campaign)
    remote = staged(api, args.repo, pinned)
    problems = compare(local, remote)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2
    print(f"{len(local)} files at {args.repo}@{pinned[:12]} ({args.source}) "
          f"match the campaign in name and size")
    if args.check:
        return 0

    operations = [
        CommitOperationCopy(src_path_in_repo=path, path_in_repo=path,
                            src_revision=pinned)
        for path in sorted(local)
    ]
    info = api.create_commit(
        repo_id=args.repo,
        operations=operations,
        revision=args.target,
        commit_message=f"publish MSRT campaign from {args.source}",
        commit_description=(
            f"One commit promoting {len(operations)} files from {pinned[:12]}. "
            f"Server-side copies: no payload is re-uploaded, and no consumer can "
            f"observe a partially published family."),
    )
    print(f"promoted {len(operations)} files from {pinned[:12]} to "
          f"{args.target}: {info.commit_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
