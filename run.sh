#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/webapp"
exec python3 -m http.server "${1:-8644}" --bind 127.0.0.1
