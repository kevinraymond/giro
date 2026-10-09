#!/usr/bin/env bash
# Score a Route C orbit LoRA checkpoint (board #3738) the way #3737 scored the plain Wan proxy orbit
# (data/sweeps/20261007-truth/consist.sh): per GSO truth-test object, giro's Wan 2.2 Fun Control orbit with the
# LoRA on the low-noise model (seed 7, 576x768x81, exact path cameras), plain 3DGS on its 65 training frames,
# then LPIPS against the true held-out views (eval) and against the orbit's own 16 held-out frames (selfeval).
#
#   eval_orbit_lora.sh CKPT.safetensors TAG GPU [NAMES...]      (CKPT: DiffSynth-Studio's output; "none": no LoRA)
#   EXPERT=high: CKPT goes on the high-noise model (default low); LOW_LORA=routec/X.safetensors: also that LoRA
#   (already under ComfyUI's loras) on the low-noise model, to score a high + low pair
#
# Baseline (no LoRA, #3737, same procedure): selfeval Wc 0.121 / 0.154 / 0.229, eval Wc 0.42 / 0.37 / 0.61 for
# Sonny_School_Bus / Reebok_REESCULPT_TRAINER_II / Android_Lego. Outputs go to OUT (default
# data/sweeps/20261008-routec): OUT/<name>/{wan-TAG, angles-wan-TAG[-held], W-TAG, eval-TAG.json, selfeval-TAG.json,
# sheet-TAG.jpg, selfeval-TAG.jpg}, one line per object in OUT/summary.tsv. Each object's work/ is linked from
# the truth test's sweep (TRUTH), whose own eval files stay untouched.
set -u
ckpt=$1 tag=$2 gpu=$3; shift 3
[ $# -gt 0 ] || set -- Sonny_School_Bus Reebok_REESCULPT_TRAINER_II Android_Lego
names=("$@")
G=/home/kevin/ai/giro
TRUTH=${TRUTH:-$G/data/sweeps/20261007-truth}
OUT=${OUT:-$G/data/sweeps/20261008-routec}
LORAS=${LORAS:-$HOME/models/loras}
cd "$G/scripts/proxy_spike"
mkdir -p "$OUT" "$LORAS/routec"
lora=()
if [ "$ckpt" != none ]; then
  uv run python ../train/wan_lora_to_comfy.py "$ckpt" "$LORAS/routec/$tag.safetensors" || exit 1
  lora=(--lora-${EXPERT:-low} "routec/$tag.safetensors")
fi
[ -n "${LOW_LORA:-}" ] && lora+=(--lora-low "$LOW_LORA")
st() { echo "$(date '+%F %T') $tag $*" >> "$OUT/status.log"; }
for n in "${names[@]}"; do
  mkdir -p "$OUT/$n"
  [ -e "$OUT/$n/work" ] || ln -s "$TRUTH/$n/work" "$OUT/$n/work"
  log=$OUT/$n-$tag.log
  uv run python gso_truth_test.py wan "$n" "$OUT" "$gpu" "${lora[@]}" --wan-tag="-$tag" > "$log" 2>&1 \
    && st "$n wan" || { st "$n wan FAILED"; continue; }
  (cd "$G" && uv run giro comfy down --gpu "$gpu") >> "$log" 2>&1
  uv run --group direct python direct_splat.py "$OUT/$n/work" "$OUT/$n/angles-wan-$tag" "$OUT/$n/W-$tag" "$gpu" --no-export \
    --views anchors --uniform --cameras "$OUT/$n/angles-wan-$tag/cameras.json" >> "$log" 2>&1 && st "$n W" || { st "$n W FAILED"; continue; }
  uv run --group direct python gso_truth_test.py eval "$n" "$OUT" "$gpu" --runs "W-$tag" >> "$log" 2>&1 \
    && mv "$OUT/$n/eval.json" "$OUT/$n/eval-$tag.json" && mv "$OUT/$n/sheet.jpg" "$OUT/$n/sheet-$tag.jpg"
  uv run --group direct python gso_truth_test.py selfeval "$n" "$OUT" "$gpu" --pairs "W-$tag:wan-$tag-held" >> "$log" 2>&1 \
    && mv "$OUT/$n/selfeval.json" "$OUT/$n/selfeval-$tag.json" && mv "$OUT/$n/selfeval.jpg" "$OUT/$n/selfeval-$tag.jpg"
  self=$(python3 -c "import json;print(json.load(open('$OUT/$n/selfeval-$tag.json'))['W-$tag:wan-$tag-held']['lpips'])" 2>/dev/null)
  truth=$(python3 -c "import json;print(json.load(open('$OUT/$n/eval-$tag.json'))['mean']['W-$tag']['lpips'])" 2>/dev/null)
  printf '%s\t%s\t%s\t%s\n' "$tag" "$n" "${self:-NA}" "${truth:-NA}" >> "$OUT/summary.tsv"
  st "$n self ${self:-NA} truth ${truth:-NA}"
done
st "done"
