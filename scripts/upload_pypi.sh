#!/bin/bash
# Wrapper script for uploading to PyPI with credentials from ~/.pypirc
#
# NOTE: Local linux_x86_64 wheels are rejected by PyPI (requires manylinux).
# This script still uploads only the sdist. For wheel publishing, use the
# cibuildwheel GitHub workflow artifacts (manylinux + macOS arm64).

set -e

REPO="${1:-pypi}"
DIST_DIR="dist"

VERSION="$(python - <<'PY'
import pathlib, tomllib
pyproject = pathlib.Path("pyproject.toml")
data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
print(data["project"]["version"])
PY
)"

CURRENT_SDIST="${DIST_DIR}/embodik-${VERSION}.tar.gz"

if [ ! -f "$CURRENT_SDIST" ]; then
    echo "Error: expected source distribution not found: $CURRENT_SDIST"
    echo "Run: pixi run build-sdist (or pixi run build-dist) before upload."
    exit 1
fi

echo "Preparing upload for version: $VERSION"
echo "Using source distribution: $CURRENT_SDIST"

# Use config file if it exists, otherwise use environment variables
if [ -f "$HOME/.pypirc" ]; then
    if [ "$REPO" = "testpypi" ]; then
        echo "Uploading source distribution to TestPyPI..."
        python -m twine upload --repository testpypi --config-file "$HOME/.pypirc" "$CURRENT_SDIST"
    else
        echo "Uploading source distribution to PyPI..."
        echo "(Note: this script uploads only sdist; use cibuildwheel workflow artifacts for wheels)"
        python -m twine upload --config-file "$HOME/.pypirc" "$CURRENT_SDIST"
    fi
else
    echo "Warning: ~/.pypirc not found. Using environment variables if set."
    echo "Set TWINE_USERNAME=__token__ and TWINE_PASSWORD=<token> if needed."
    if [ "$REPO" = "testpypi" ]; then
        python -m twine upload --repository testpypi "$CURRENT_SDIST"
    else
        python -m twine upload "$CURRENT_SDIST"
    fi
fi

# Cleanup: remove older source tarballs from previous releases.
for f in "$DIST_DIR"/embodik-*.tar.gz; do
    [ -e "$f" ] || continue
    if [ "$f" != "$CURRENT_SDIST" ]; then
        rm -f "$f"
    fi
done
echo "Cleaned old source tarballs from $DIST_DIR (kept $CURRENT_SDIST)."
