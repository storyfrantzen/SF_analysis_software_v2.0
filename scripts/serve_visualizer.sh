#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/serve_visualizer.sh [--port PORT] [--host HOST] [PATH.html|DIRECTORY]

Serve generated visualizer HTML files for VS Code Remote SSH.

With no path, the script serves results/ if it exists, otherwise the current
directory. Passing a directory opens a click-enabled file listing. Passing an
HTML file opens that file directly.

Options:
  -p, --port PORT   Port to bind locally on the remote host (default: 8765)
      --host HOST   Host/interface to bind (default: 127.0.0.1)
  -h, --help        Show this help

Example:
  scripts/serve_visualizer.sh
  scripts/serve_visualizer.sh results
  scripts/serve_visualizer.sh results/selected_data_histograms_2.html
  scripts/serve_visualizer.sh --port 8877 results/selected_data_histograms_2.html
EOF
}

PORT="${PORT:-8765}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
TARGET=""

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
      if [[ -n "$TARGET" ]]; then
        echo "Only one file or directory may be served at a time" >&2
        usage >&2
        exit 2
      fi
      TARGET="$1"
      shift
      ;;
  esac
done

if (($#)); then
  if [[ -n "$TARGET" ]]; then
    echo "Only one file or directory may be served at a time" >&2
    usage >&2
    exit 2
  fi
  TARGET="$1"
  shift
fi

if (($#)); then
  echo "Only one file or directory may be served at a time" >&2
  usage >&2
  exit 2
fi

if [[ -z "$TARGET" ]]; then
  if [[ -d results ]]; then
    TARGET="results"
  else
    TARGET="."
  fi
fi

if [[ -d "$TARGET" ]]; then
  DIR="$(cd "$TARGET" && pwd -P)"
  URL_PATH="/"
  SERVING="${DIR}/"
elif [[ -f "$TARGET" ]]; then
  DIR="$(cd "$(dirname "$TARGET")" && pwd -P)"
  FILE="$(basename "$TARGET")"
  URL_FILE="$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$FILE")"
  URL_PATH="/${URL_FILE}"
  SERVING="${DIR}/${FILE}"
else
  echo "File or directory does not exist: $TARGET" >&2
  exit 1
fi

FORWARDED_URL="http://127.0.0.1:${PORT}${URL_PATH}"
BIND_URL="http://${BIND_HOST}:${PORT}${URL_PATH}"

cat <<EOF
Serving: ${SERVING}
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
