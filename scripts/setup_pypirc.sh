#!/bin/bash
# Setup script to create ~/.pypirc with API tokens
# Prompts for tokens interactively (tokens are not stored in this script)

set -e

echo "=== PyPI Credentials Setup ==="
echo ""
echo "Enter your PyPI API tokens when prompted."
echo "You can get tokens from:"
echo "  - TestPyPI: https://test.pypi.org/manage/account/token/"
echo "  - PyPI:     https://pypi.org/manage/account/token/"
echo ""

# Prompt for TestPyPI token
read -sp "Enter TestPyPI token (starts with pypi-): " TESTPYPI_TOKEN
echo ""

# Prompt for PyPI token
read -sp "Enter PyPI token (starts with pypi-): " PYPI_TOKEN
echo ""

# Create the config file
cat > "$HOME/.pypirc" << EOF
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = ${PYPI_TOKEN}

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = ${TESTPYPI_TOKEN}
EOF

chmod 600 "$HOME/.pypirc"
echo ""
echo "✓ Created ~/.pypirc with API tokens"
echo "✓ File permissions set to 600 (read/write for owner only)"
