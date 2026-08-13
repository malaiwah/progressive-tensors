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

# Written by the campaign, not uploaded: local bookkeeping and the resume state.
LOCAL_ONLY = {".fq-msrt-encode.json", "source-digests.json"}
LOCAL_ONLY_DIRS = {"logs", "locks", ".driver", "skeleton"}


def campaign_files(campaign: Path) -> dict[str, int]:
    """Every path a finalized campaign publishes, relative to the repo root.

    `skeleton/` is excluded because `finalize` hardlinks its shards into each
    base, which is what makes a base loadable; publishing it twice would double
    the upload for nothing.
    """
    out: dict[str, int] = {}
    for path in campaign.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(campaign)
        if relative.parts[0] in LOCAL_ONLY_DIRS or relative.name in LOCAL_ONLY:
            continue
        # Every dotted path in a campaign is local bookkeeping: the sentinel and
        # its guard, the campaign and worker locks, the driver's window state.
        if any(part.startswith(".") for part in relative.parts):
            continue
        out[relative.as_posix()] = path.stat().st_size
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
    local = campaign_files(args.campaign)
    remote = staged(api, args.repo, args.source)
    problems = compare(local, remote)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2
    print(f"{len(local)} files on {args.repo}@{args.source} match the campaign "
          f"byte for byte in name and size")
    if args.check:
        return 0

    # Pin the branch to an immutable commit before copying from it, so a
    # concurrent upload cannot change what "staging" means mid-promotion.
    refs = api.list_repo_refs(repo_id=args.repo)
    pinned = next((branch.target_commit for branch in refs.branches
                   if branch.name == args.source), None)
    if pinned is None:
        print(f"error: {args.repo} has no branch {args.source}", file=sys.stderr)
        return 2

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
