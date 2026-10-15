#!/bin/sh
# Thin wrapper: run the full local reproduction pipeline.
# Usage: sh run_all.sh          (data + figures)
#        python3 run_all.py --list   to inspect the plan first.
set -e
cd "$(dirname "$0")"
exec python3 run_all.py "$@"
