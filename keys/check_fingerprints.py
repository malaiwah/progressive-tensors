#!/usr/bin/env python3
"""CI guard for the out-of-band trust root in keys/.

Checks, with the standard library only (no venv needed in CI):

1. every FINGERPRINTS record is well-formed: 64 lowercase hex chars, a key
   id, a known status, an ISO date, a non-empty role list;
2. no duplicate fingerprint and no duplicate key id;
3. every <key-id>.ed25519.pub file holds exactly that key id's fingerprint,
   and every active/retired record has such a file;
4. no stray .pub file that FINGERPRINTS does not know about.

Exit 0 when the trust root is internally consistent, 1 otherwise.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
KEYID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STATUSES = {"active", "retired", "revoked"}


def parse_fingerprints(path: Path) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{n}: expected 5 fields, got {len(parts)}: {raw!r}")
            continue
        fp, key_id, status, added, role = parts
        if not FINGERPRINT_RE.match(fp):
            errors.append(f"{path.name}:{n}: fingerprint is not 64 lowercase hex chars: {fp!r}")
        if not KEYID_RE.match(key_id):
            errors.append(f"{path.name}:{n}: bad key id {key_id!r}")
        if status not in STATUSES:
            errors.append(f"{path.name}:{n}: status {status!r} not in {sorted(STATUSES)}")
        if not DATE_RE.match(added):
            errors.append(f"{path.name}:{n}: date {added!r} is not YYYY-MM-DD")
        if not role.strip(","):
            errors.append(f"{path.name}:{n}: empty role list")
        records.append({"fingerprint": fp, "key_id": key_id, "status": status,
                        "added": added, "role": role, "line": n})
    return records, errors


def main() -> int:
    path = HERE / "FINGERPRINTS"
    if not path.exists():
        print(f"missing trust root: {path}", file=sys.stderr)
        return 1
    records, errors = parse_fingerprints(path)
    if not records:
        errors.append("FINGERPRINTS has no records — the trust root cannot be empty")

    seen_fp: dict[str, int] = {}
    seen_id: dict[str, int] = {}
    for r in records:
        if r["fingerprint"] in seen_fp:
            errors.append(f"duplicate fingerprint {r['fingerprint'][:16]}… "
                          f"(lines {seen_fp[r['fingerprint']]} and {r['line']})")
        if r["key_id"] in seen_id:
            errors.append(f"duplicate key id {r['key_id']!r} "
                          f"(lines {seen_id[r['key_id']]} and {r['line']})")
        seen_fp[r["fingerprint"]] = r["line"]
        seen_id[r["key_id"]] = r["line"]

    expected_files = set()
    for r in records:
        pub = HERE / f"{r['key_id']}.ed25519.pub"
        if r["status"] == "revoked" and not pub.exists():
            continue  # a revoked key need not keep a key file around
        expected_files.add(pub.name)
        if not pub.exists():
            errors.append(f"{r['key_id']}: missing key file {pub.name}")
            continue
        body = pub.read_text().strip()
        if body != r["fingerprint"]:
            errors.append(f"{pub.name}: contents {body[:16]}… do not match "
                          f"FINGERPRINTS record {r['fingerprint'][:16]}…")

    for pub in sorted(HERE.glob("*.ed25519.pub")):
        if pub.name not in expected_files:
            errors.append(f"{pub.name}: key file with no FINGERPRINTS record")

    for e in errors:
        print(f"trust-root: {e}", file=sys.stderr)
    if errors:
        return 1
    active = [r["key_id"] for r in records if r["status"] == "active"]
    print(f"trust root OK: {len(records)} record(s), active: {', '.join(active) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
