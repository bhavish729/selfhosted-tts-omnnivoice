#!/usr/bin/env bash
# Pull WAVs from the RunPod pod into a local recordings/ folder, stamping each
# filename with the current UTC timestamp so you never lose provenance.
#
# Usage:
#   scripts/pull_recordings.sh ref       # corpus/ref_audio/*.wav  -> recordings/ref_<ts>/
#   scripts/pull_recordings.sh quality   # results/quality_samples/num_step_*/*.wav
#   scripts/pull_recordings.sh all       # both
set -euo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/runpod_claude}"
SSH_OPTS=(-o StrictHostKeyChecking=no -i "$SSH_KEY" -p 10883)
POD="root@103.207.149.105"
POD_BASE="/workspace/omnivoice-bench"
LOCAL_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/recordings"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

mode="${1:-all}"

pull_ref() {
  local out="$LOCAL_BASE/ref_${TS}"
  mkdir -p "$out"
  echo "[pull] ref_audio -> $out"
  # rsync into a tmp dir, then rename with timestamp prefix.
  local tmp; tmp="$(mktemp -d)"
  rsync -a -e "ssh ${SSH_OPTS[*]}" "$POD:$POD_BASE/corpus/ref_audio/" "$tmp/"
  shopt -s nullglob
  local n=0
  for f in "$tmp"/*.wav; do
    local base; base="$(basename "$f" .wav)"
    cp "$f" "$out/${base}_${TS}.wav"
    n=$((n+1))
  done
  rm -rf "$tmp"
  echo "[pull] copied $n ref WAVs"
}

pull_quality() {
  local out="$LOCAL_BASE/quality_${TS}"
  mkdir -p "$out"
  echo "[pull] quality_samples -> $out"
  local tmp; tmp="$(mktemp -d)"
  if ! rsync -a -e "ssh ${SSH_OPTS[*]}" "$POD:$POD_BASE/results/quality_samples/" "$tmp/" 2>/dev/null; then
    echo "[pull] (no quality_samples yet on pod)"
    rm -rf "$tmp"
    return 0
  fi
  shopt -s nullglob
  local n=0
  for subdir in "$tmp"/num_step_*; do
    [ -d "$subdir" ] || continue
    local ns; ns="$(basename "$subdir")"
    mkdir -p "$out/$ns"
    for f in "$subdir"/*.wav; do
      local base; base="$(basename "$f" .wav)"
      cp "$f" "$out/$ns/${base}_${TS}.wav"
      n=$((n+1))
    done
  done
  rm -rf "$tmp"
  echo "[pull] copied $n quality WAVs across num_step subdirs"
}

case "$mode" in
  ref) pull_ref ;;
  quality) pull_quality ;;
  all) pull_ref; pull_quality ;;
  *) echo "usage: $0 {ref|quality|all}"; exit 2 ;;
esac
echo "[pull] done at $TS"
