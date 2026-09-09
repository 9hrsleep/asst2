#!/usr/bin/env bash

# profile.sh
# 
# Wrapper for ncu/nsys on ghc machines for 15-418/618.
#
# - Sets per-user environment variable TMPDIR to avoid Nsight locking conflicts
# - Preserves user flags/ arguments
# - Not necessary when analyzing existing reports
#
# Usage:
# ./profile.sh ncu [args...]
# ./profile nsys [args...]

set -euo pipefail

usage() {
  echo "Usage: gpu-prof-nolock <ncu|nsys> [args...]" >&2
  exit 2
}

# Check arg count
[[ $# -ge 1 ]] || usage
TOOL="$1"; shift

# Ensure we're running ncu or nsys
case "$TOOL" in
  ncu|nsys) ;;
  *) usage ;;
esac

USER_TMP="/tmp/$USER"
mkdir -p "$USER_TMP"
chmod 700 "$USER_TMP" 2>/dev/null || true

TMPDIR="$USER_TMP" TMP="$USER_TMP" TEMP="$USER_TMP" exec "$TOOL" "$@"
