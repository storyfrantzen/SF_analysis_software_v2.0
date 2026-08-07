#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/visualize.sh ROOT_FILE EVENT_LIMIT [ROOT_FILTER]

Build and serve an ephemeral visualizer for a ROOT tree. The temporary HTML is
removed when the server stops.

Arguments:
  ROOT_FILE     Converter or selected ROOT file
  EVENT_LIMIT   Number of source events to sample; 0 keeps every matching event
  ROOT_FILTER   Optional ROOT expression applied before event sampling

Environment overrides:
  VISUALIZER_PORT        Server port (default: 8765)
  VISUALIZER_HOST        Bind address (default: 127.0.0.1)
  VISUALIZER_TREE        ROOT tree name (default: auto-detect canonical tree)
  VISUALIZER_SEED        Deterministic sampling seed (default: 12345)
  VISUALIZER_DICTIONARY  ROOT dictionary path
  VISUALIZER_PYTHON      Python executable (default: python3)

Examples:
  scripts/visualize.sh data.root 250000
  scripts/visualize.sh data.root 0 'event.runNum == 18480'
  scripts/visualize.sh data.root 100000 'event.runNum >= 18480 && event.runNum <= 18490'
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

ROOT_FILE=$1
EVENT_LIMIT=$2
ROOT_FILTER=${3:-}

if [[ ! -f "$ROOT_FILE" ]]; then
  echo "ROOT file does not exist: $ROOT_FILE" >&2
  exit 1
fi
if [[ ! "$EVENT_LIMIT" =~ ^[0-9]+$ ]]; then
  echo "EVENT_LIMIT must be a non-negative integer: $EVENT_LIMIT" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PORT=${VISUALIZER_PORT:-8765}
BIND_HOST=${VISUALIZER_HOST:-127.0.0.1}
TREE_NAME=${VISUALIZER_TREE:-}
SEED=${VISUALIZER_SEED:-12345}
PYTHON_BIN=${VISUALIZER_PYTHON:-python3}

DICTIONARY=${VISUALIZER_DICTIONARY:-}
if [[ -z "$DICTIONARY" ]]; then
  for candidate in \
    "$REPO_ROOT/build/libROOTBranchesDict.so" \
    "$REPO_ROOT/build/libROOTBranchesDict.dylib" \
    "$REPO_ROOT/work-build/libROOTBranchesDict.so" \
    "$REPO_ROOT/work-build/libROOTBranchesDict.dylib"; do
    if [[ -f "$candidate" ]]; then
      DICTIONARY=$candidate
      break
    fi
  done
elif [[ ! -f "$DICTIONARY" ]]; then
  echo "ROOT dictionary does not exist: $DICTIONARY" >&2
  exit 1
fi

TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/sf-visualizer.XXXXXX")
HTML_FILE="$TEMP_DIR/index.html"
cleanup() {
  if [[ -n "${TEMP_DIR:-}" && -d "$TEMP_DIR" ]]; then
    rm -rf -- "$TEMP_DIR"
  fi
}
trap cleanup EXIT INT TERM

visualizer_command=(
  "$PYTHON_BIN" -m visualizer "$ROOT_FILE"
  --format root
  --max-source-events "$EVENT_LIMIT"
  --seed "$SEED"
  --output "$HTML_FILE"
)
if [[ -n "$TREE_NAME" ]]; then
  visualizer_command+=(--tree "$TREE_NAME")
fi
if [[ -n "$DICTIONARY" ]]; then
  visualizer_command+=(--dictionary "$DICTIONARY")
fi
if [[ -n "$ROOT_FILTER" ]]; then
  visualizer_command+=(--root-filter "$ROOT_FILTER")
fi

echo "Generating ephemeral visualizer..."
(cd "$REPO_ROOT" && "${visualizer_command[@]}")
echo "The temporary HTML will be removed when this server stops."
"$REPO_ROOT/scripts/serve_visualizer.sh" \
  --host "$BIND_HOST" \
  --port "$PORT" \
  "$HTML_FILE"
