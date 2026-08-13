#!/usr/bin/env python3
"""Promote a fully uploaded campaign from a staging branch to main, atomically.

A campaign is ~4,500 files across two bases, five stage trees and the signed
plans. Uploading them to `main` directly means many commits, and a consumer who
fetches between two of them gets a family that does not exist: a base whose
skeleton is missing, or a stage whose parent is not published yet.

So the campaign is uploaded to a staging branch and promoted here in **one
commit**, built from server-side copy operations. Nothing is re-uploaded: each
operation names a path and the revision to copy it from, so promotion costs one
API call regardless of the terabyte behind it.

    tools/promote_campaign.py --repo malaiwah/GLM-5.2-MSRT --from staging

`--check` verifies the staging branch carries everything the campaign summary
claims, and exits non-zero if not, without touching `main`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import CommitOperationCopy, HfApi


def staged_files(api: HfApi, repo: str, revision: str) -> list[str]:
    return [
        path for path in api.list_repo_files(repo_id=repo, revision=revision)
        if not path.startswith(".")
    ]


def expected_from_summary(summary: dict) -> set[str]:
    """Every path the campaign says it published, from its own summary."""
    wanted = {"campaign_summary.json"}
    for label in summary["bases"]:
        wanted.add(f"base/{label}/MANIFEST.sha256")
        wanted.add(f"base/{label}/config.json")
        wanted.add(f"base/{label}/model.safetensors.index.json")
    for assembly in summary["assemblies"]:
        wanted.add(f"assemblies/{assembly['label']}/assembly.jsonl")
    return wanted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="destination repo id")
    parser.add_argument("--from", dest="source", default="staging",
                        help="staging branch holding the uploaded campaign")
    parser.add_argument("--to", dest="target", default="main")
    parser.add_argument("--summary", type=Path,
                        help="local campaign_summary.json to check against")
    parser.add_argument("--check", action="store_true",
                        help="verify staging only; never touch --to")
    args = parser.parse_args(argv)

    api = HfApi()
    files = staged_files(api, args.repo, args.source)
    if not files:
        print(f"error: {args.repo}@{args.source} is empty", file=sys.stderr)
        return 2

    if args.summary:
        wanted = expected_from_summary(json.loads(args.summary.read_text()))
        missing = sorted(wanted - set(files))
        if missing:
            print(f"error: {len(missing)} published paths are absent from "
                  f"{args.source}, e.g. {missing[:5]}", file=sys.stderr)
            return 2
        print(f"staging carries every path {args.summary.name} names")

    print(f"{len(files)} files on {args.repo}@{args.source}")
    if args.check:
        return 0

    operations = [
        CommitOperationCopy(src_path_in_repo=path, path_in_repo=path,
                            src_revision=args.source)
        for path in files
    ]
    info = api.create_commit(
        repo_id=args.repo,
        operations=operations,
        revision=args.target,
        commit_message=f"publish MSRT campaign from {args.source}",
        commit_description=(
            f"One commit promoting {len(files)} files. Server-side copies: no "
            f"payload is re-uploaded, and no consumer can observe a partially "
            f"published family."),
    )
    print(f"promoted {len(files)} files to {args.target}: {info.commit_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
