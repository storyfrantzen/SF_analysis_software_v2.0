#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/serve_visualizer.sh [--port PORT] [--host HOST] PATH.html

Serve a generated interactive_histograms.py HTML file for VS Code Remote SSH.

Options:
  -p, --port PORT   Port to bind locally on the remote host (default: 8765)
      --host HOST   Host/interface to bind (default: 127.0.0.1)
  -h, --help        Show this help

Example:
  scripts/serve_visualizer.sh results/selected_data_histograms_2.html
  scripts/serve_visualizer.sh --port 8877 results/selected_data_histograms_2.html
EOF
}

PORT="${PORT:-8765}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
HTML=""

while (($#)); do
  case "$1" in
    -p|--port)
      if (($# < 2)); then
        echo "Missing value for $1" >&2
        exit 2
      fi
      PORT="$2"
      shift 2
      ;;
    --host)
      if (($# < 2)); then
        echo "Missing value for $1" >&2
        exit 2
      fi
      BIND_HOST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$HTML" ]]; then
        echo "Only one HTML file may be served at a time" >&2
        usage >&2
        exit 2
      fi
      HTML="$1"
      shift
      ;;
  esac
done

if [[ -z "$HTML" ]]; then
  echo "Missing HTML file" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$HTML" ]]; then
  echo "HTML file does not exist: $HTML" >&2
  exit 1
fi

DIR="$(cd "$(dirname "$HTML")" && pwd -P)"
FILE="$(basename "$HTML")"
URL_FILE="$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$FILE")"
FORWARDED_URL="http://127.0.0.1:${PORT}/${URL_FILE}"
BIND_URL="http://${BIND_HOST}:${PORT}/${URL_FILE}"

cat <<EOF
Serving: ${DIR}/${FILE}
Binding: ${BIND_HOST}:${PORT}

In VS Code Remote SSH:
  1. Forward port ${PORT} if VS Code does not auto-detect it.
  2. Open "Simple Browser: Show" from the Command Palette.
  3. Paste:
     ${FORWARDED_URL}

Remote-side URL:
  ${BIND_URL}

Press Ctrl-C here to stop serving.

EOF

python3 -m http.server "$PORT" --bind "$BIND_HOST" --directory "$DIR"
