#!/bin/bash
# Wrapper script for uploading to PyPI with credentials from ~/.pypirc
#
# NOTE: PyPI rejects linux_x86_64 wheels (requires manylinux).
# For now, we upload only source distributions. Users will build from source
# or we'll add manylinux wheel builds via cibuildwheel in the future.

set -e

REPO="${1:-pypi}"

# Use config file if it exists, otherwise use environment variables
if [ -f "$HOME/.pypirc" ]; then
    if [ "$REPO" = "testpypi" ]; then
        echo "Uploading source distribution to TestPyPI..."
        twine upload --repository testpypi --config-file "$HOME/.pypirc" dist/*.tar.gz
    else
        echo "Uploading source distribution to PyPI..."
        echo "(Note: linux_x86_64 wheels are rejected; use cibuildwheel for manylinux wheels)"
        twine upload --config-file "$HOME/.pypirc" dist/*.tar.gz
    fi
else
    echo "Warning: ~/.pypirc not found. Using environment variables if set."
    echo "Set TWINE_USERNAME=__token__ and TWINE_PASSWORD=<token> if needed."
    if [ "$REPO" = "testpypi" ]; then
        twine upload --repository testpypi dist/*.tar.gz
    else
        twine upload dist/*.tar.gz
    fi
fi
