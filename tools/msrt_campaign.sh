#!/usr/bin/env bash
# Drive a whole MSRT campaign: stage, skeleton, encode, retire, finalize.
#
# The runbook (docs/MSRT-CAMPAIGN.md) explains why each step exists. This is the
# order and the flags, in one place, so an operator cannot forget the ones that
# only fail late: the pinned revision, the explicit identity a --local-dir source
# cannot infer, the shards that belong to no layer, and the shards a later window
# still needs.
#
#   REPO_ID=zai-org/GLM-5.2 REV=<40hex> SRC=/data/glm52-src \
#   CAMPAIGN=/data/glm52-msrt KEY=/data/keys/c.key ENC=/opt/exllamav3 \
#   RECIPE=/data/progressive-tensors/recipes/glm52-k2k3-lean.json \
#   DEVICES=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
#     /data/progressive-tensors/tools/msrt_campaign.sh
#
# Two phases. PHASE=encode (the default) stages, encodes and retires every
# window, then stops so the GPU fleet can be released. PHASE=finalize runs the
# publication pass on a CPU VM with the same volume attached.
#
# DRY_RUN=1 prints every command instead of running it.
set -euo pipefail

: "${REPO_ID:?set REPO_ID}" "${REV:?set REV}" "${SRC:?set SRC}"
: "${CAMPAIGN:?set CAMPAIGN}" "${KEY:?set KEY}" "${RECIPE:?set RECIPE}"
[[ $REV =~ ^[0-9a-f]{40}$ ]] || { echo "REV must be a 40-hex commit" >&2; exit 2; }

BLOCK_SIZE=${BLOCK_SIZE:-32}
if [[ ${PHASE:-encode} == encode ]]; then
  # The CPU phase needs neither of these, and the encoder bundle does not
  # survive the reattach; defaulting DEVICES would run one GPU while eight bill.
  : "${ENC:?set ENC}"
  : "${DEVICES:?set DEVICES to every GPU you are paying for, e.g. cuda:0,cuda:1}"
fi
[[ ${PHASE:-encode} == encode || ${PHASE:-encode} == finalize ]] ||
  { echo "PHASE must be encode or finalize" >&2; exit 2; }
WINDOWS=${WINDOWS:-"3-10 11-18 19-26 27-34 35-42 43-50 51-58 59-66 67-74 75-78"}
read -r -a FQ_CMD <<< "${FQ:-fq-assemble-lora}"
WORK=${WORK:-$CAMPAIGN/.driver}
DRY_RUN=${DRY_RUN:-0}

# $SRC is a --local-dir tree, not a Hugging Face snapshot path, so identity can
# never be inferred from it. Every command that signs anything gets it.
IDENTITY=(--base-model "$REPO_ID" --base-revision "$REV")
COMMON=(--source "$SRC" --recipe "$RECIPE" --out "$CAMPAIGN"
        --block-size "$BLOCK_SIZE")

run() {
  if [[ $DRY_RUN == 1 ]]; then printf '+ %s\n' "$*"; else "$@"; fi
}

stage() {  # stage(window) - the window's shards plus the ones no layer owns
  local window=$1
  local plan=$WORK/plan-$window.json
  run "${FQ_CMD[@]}" plan "${COMMON[@]}" --layers "$window" --out-plan "$plan"
  [[ $DRY_RUN == 1 ]] && { printf '+ hf download %s --revision %s --local-dir %s <shards of %s + skeleton_only>\n' \
      "$REPO_ID" "$REV" "$SRC" "$window"; return 0; }
  mapfile -t includes < <(python3 - "$plan" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
want = {s for layer in plan["layers"] for s in layer["shards"]}
want |= set(plan["skeleton_only_shards"])
for shard in sorted(want):
    print("--include")
    print(shard)
PY
)
  run hf download "$REPO_ID" --revision "$REV" --local-dir "$SRC" "${includes[@]}"
}

retire() {  # retire(window, remaining...) - delete only what no later window needs
  local window=$1; shift
  local remaining="$*"
  if [[ $DRY_RUN == 1 ]]; then
    printf '+ retire shards of %s not needed by [%s]\n' "$window" "${remaining:-none}"
    return 0
  fi
  local keep_plan=""
  if [[ -n $remaining ]]; then
    keep_plan=$WORK/keep.json
    "${FQ_CMD[@]}" plan "${COMMON[@]}" --layers "${remaining// /,}" --out-plan "$keep_plan" >/dev/null
  fi
  python3 - "$WORK/plan-$window.json" "$SRC" "$keep_plan" <<'PY' | xargs -r rm -f --
import json, os, sys
done, src, keep_path = sys.argv[1], sys.argv[2], sys.argv[3]
retire = {s for layer in json.load(open(done))["layers"] for s in layer["shards"]}
keep = set()
if keep_path:
    plan = json.load(open(keep_path))
    keep = {s for layer in plan["layers"] for s in layer["shards"]}
    keep |= set(plan["skeleton_only_shards"])
else:
    # Last window: nothing else will read a payload shard, but config.json and
    # the index must survive - finalize resolves every layer through them.
    keep = set(json.load(open(done))["skeleton_only_shards"])
for shard in sorted(retire - keep):
    print(os.path.join(src, shard))
PY
}

mkdir -p "$WORK"

# A done-marker means "this window of THIS run is finished". Bind it to the
# arguments that decide what a window contains, so a drifted rerun cannot skip
# work it never did. The campaign sentinel already refuses recipe, revision and
# block-size drift; the window partition is only known here.
digest_of() {  # sha256sum is GNU; shasum is the BSD/macOS spelling
  command sha256sum "$1" 2>/dev/null | cut -d' ' -f1 ||
    shasum -a 256 "$1" | cut -d' ' -f1
}
identity="recipe=$(digest_of "$RECIPE") rev=$REV \
block=$BLOCK_SIZE windows=$WINDOWS"
if [[ -e $WORK/identity ]]; then
  if [[ $(cat "$WORK/identity") != "$identity" ]]; then
    echo "error: $WORK holds markers from a different run:" >&2
    echo "  had:  $(cat "$WORK/identity")" >&2
    echo "  want: $identity" >&2
    echo "Use a fresh --out, or FORCE_WINDOWS=1 to ignore the markers." >&2
    [[ -z ${FORCE_WINDOWS:-} ]] && exit 2
  fi
elif [[ $DRY_RUN != 1 ]]; then
  printf '%s' "$identity" > "$WORK/identity"
fi

if [[ ${PHASE:-encode} == finalize ]]; then
  # CPU phase: no GPU, no source payload, just the campaign and the key.
  run "${FQ_CMD[@]}" finalize "${COMMON[@]}" "${IDENTITY[@]}" --sign-key "$KEY"
  echo "=== finalized; upload the bases and promote (docs/MSRT-CAMPAIGN.md 4.5)"
  exit 0
fi

# WINDOWS decides what gets encoded. A range that quietly omits one recipe layer
# would encode 75 of 76, retire the source those layers needed, and be discovered
# by finalize after the fleet was paid for. It must cover every recipe layer
# exactly once, and that is checked before a single byte is staged.
python3 - "$RECIPE" "$WINDOWS" <<'PY' || exit 2
import json, sys
recipe = json.load(open(sys.argv[1]))
want = list(recipe["moe_layers"])
seen = []
for part in sys.argv[2].split():
    if "-" in part:
        lo, hi = part.split("-", 1)
        seen += list(range(int(lo), int(hi) + 1))
    elif part.strip():
        seen.append(int(part))
problems = []
missing = sorted(set(want) - set(seen))
extra = sorted(set(seen) - set(want))
repeated = sorted({layer for layer in seen if seen.count(layer) > 1})
if missing:
    problems.append("never encodes recipe layers %s" % missing)
if extra:
    problems.append("names layers the recipe does not have: %s" % extra)
if repeated:
    problems.append("encodes layers twice: %s" % repeated)
if problems:
    print("error: WINDOWS " + "; ".join(problems), file=sys.stderr)
    raise SystemExit(2)
print("windows cover all %d recipe layers exactly once" % len(want))
PY

read -r -a windows <<< "$WINDOWS"
for index in "${!windows[@]}"; do
  window=${windows[$index]}
  if [[ -e $WORK/done-$window && -z ${FORCE_WINDOWS:-} ]]; then
    echo "=== window $window ($((index + 1))/${#windows[@]}): already complete ==="
    continue
  fi
  echo "=== window $window ($((index + 1))/${#windows[@]}) ==="
  stage "$window"
  # skeleton first: it mints the signing key, and eight encode workers racing to
  # create one would each sign with a different identity.
  # --source-digests lets the skeleton attest what it copied without re-hashing
  # source payloads. Optional: without it, each shard is hashed once, cached in
  # $CAMPAIGN/source-digests.json.
  skeleton_args=()
  [[ -n ${SOURCE_DIGESTS:-} ]] && skeleton_args+=(--source-digests "$SOURCE_DIGESTS")
  run "${FQ_CMD[@]}" skeleton "${COMMON[@]}" "${IDENTITY[@]}" --sign-key "$KEY" \
    "${skeleton_args[@]+"${skeleton_args[@]}"}"
  run "${FQ_CMD[@]}" encode "${COMMON[@]}" "${IDENTITY[@]}" --sign-key "$KEY" \
    --encoder-source "$ENC" --devices "$DEVICES" --layers "$window"
  [[ -n ${UPLOAD_HOOK:-} ]] && run "$UPLOAD_HOOK" "$window"
  retire "$window" "${windows[@]:index+1}"
  # Only now: `encode` above returns non-zero unless every block of this window
  # is committed on disk, so this marker means the window is finished, not
  # attempted. A rerun after a preemption skips it instead of re-downloading
  # ~166 GB of source for nothing.
  [[ $DRY_RUN == 1 ]] || : > "$WORK/done-$window"
done

cat <<'NOTE'
=== encode phase complete ===
Everything below needs no GPU. Before releasing the fleet, confirm the campaign
can be finished without it, then terminate the GPU instances, attach the volume
to a CPU VM, and re-run this script with PHASE=finalize.
NOTE
for path in "$KEY" "$RECIPE" "$SRC/model.safetensors.index.json" \
            "$SRC/config.json" "$CAMPAIGN"; do
  [[ -e $path ]] || { echo "NOT on the volume, finalize would fail: $path" >&2
                      exit 1; }
done
echo "volume holds the key, the recipe, the source metadata and the campaign"
findmnt -no SOURCE,TARGET --target "$CAMPAIGN" || true
