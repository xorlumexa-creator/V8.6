#!/usr/bin/env bash
# Run once from the V8.6-main folder (git repo root).
set -e
rm -f "tests/test_assembly_fixes (1).py"        # its content is now tests/test_assembly_fixes.py (the old file there was empty)
rm -rf __pycache__ tests/__pycache__
echo "backend cleanup done"
