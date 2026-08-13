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
#   CAMPAIGN=/data/glm52-msrt KEY=~/.fq_keys/c.key ENC=/opt/exllamav3 \
#   RECIPE=recipes/glm52-k2k3-lean.json DEVICES=cuda:0,cuda:1 \
#     tools/msrt_campaign.sh
#
# DRY_RUN=1 prints every command instead of running it.
set -euo pipefail

: "${REPO_ID:?set REPO_ID}" "${REV:?set REV}" "${SRC:?set SRC}"
: "${CAMPAIGN:?set CAMPAIGN}" "${KEY:?set KEY}" "${RECIPE:?set RECIPE}"
: "${ENC:?set ENC}"
[[ $REV =~ ^[0-9a-f]{40}$ ]] || { echo "REV must be a 40-hex commit" >&2; exit 2; }

BLOCK_SIZE=${BLOCK_SIZE:-32}
DEVICES=${DEVICES:-cuda:0}
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
read -r -a windows <<< "$WINDOWS"
for index in "${!windows[@]}"; do
  window=${windows[$index]}
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
done

echo "=== release the GPU fleet now: finalize and the final upload need none ==="
run "${FQ_CMD[@]}" finalize "${COMMON[@]}" "${IDENTITY[@]}" --sign-key "$KEY"
