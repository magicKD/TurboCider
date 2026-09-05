#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BASE_PYTHON=${BASE_PYTHON:-${TURBOCIDER_BASE_PYTHON:-python3}}
DEST="$ROOT/Python"

if [ ! -x "$DEST/bin/python3" ]; then
    "$BASE_PYTHON" -m venv "$DEST"
fi

"$DEST/bin/python" -m pip install --upgrade pip setuptools wheel
"$DEST/bin/python" -m pip install \
    -r "$ROOT/requirements-flux2.txt" \
    modelscope

echo "vendored Python runtime to $DEST"
"$DEST/bin/python" --version
