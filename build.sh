#!/bin/bash
set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building embodiK with Pinocchio Integration ===${NC}"

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check for Pinocchio - first try local installation
PINOCCHIO_INSTALL_DIR="$HOME/Projects/repos/pinocchio/install"
if [ -d "$PINOCCHIO_INSTALL_DIR" ]; then
    echo -e "${GREEN}Found local Pinocchio installation at $PINOCCHIO_INSTALL_DIR${NC}"
    export CMAKE_PREFIX_PATH="$PINOCCHIO_INSTALL_DIR:$CMAKE_PREFIX_PATH"
    export PKG_CONFIG_PATH="$PINOCCHIO_INSTALL_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
elif ! pkg-config --exists pinocchio; then
    echo -e "${RED}Error: Pinocchio not found!${NC}"
    echo "Please install Pinocchio:"
    echo "  sudo apt-get install robotpkg-pinocchio"
    echo "  or build from source: https://github.com/stack-of-tasks/pinocchio"
    exit 1
fi

# Check for nanobind
if ! python3 -c "import nanobind" 2>/dev/null; then
    echo "Nanobind not found. Installing..."
    pip3 install --user nanobind
fi

# Check for Eigen
if ! pkg-config --exists eigen3; then
    echo -e "${RED}Error: Eigen3 not found!${NC}"
    echo "Please install Eigen3:"
    echo "  sudo apt-get install libeigen3-dev"
    exit 1
fi

echo -e "${GREEN}✓ All dependencies found${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo -e "${YELLOW}Configuring with CMake...${NC}"

# Add eigenpy path from pip installation
EIGENPY_PATH="/path/to/local/.local/lib/python3.10/site-packages/cmeel.prefix"
if [ -d "$EIGENPY_PATH" ]; then
    echo -e "${GREEN}Adding eigenpy path: $EIGENPY_PATH${NC}"
    export CMAKE_PREFIX_PATH="$EIGENPY_PATH:$CMAKE_PREFIX_PATH"
fi

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=OFF

# Build
echo -e "${YELLOW}Building C++ library and Python bindings...${NC}"
make -j$(nproc)

# Run C++ tests
if [ -f "test/test_robot_model" ]; then
    echo -e "${YELLOW}Running C++ tests...${NC}"
    ctest --output-on-failure || true
fi

cd ..

echo -e "${GREEN}✓ Build completed${NC}"

# Copy C++ extension to Python package for development use
echo -e "${BLUE}Copying C++ extension to Python package...${NC}"
EXTENSION_FILE=$(find build/python_bindings -name "_embodik_impl.cpython-*.so" | head -1)
if [ -f "$EXTENSION_FILE" ]; then
    cp "$EXTENSION_FILE" python/embodik/
    echo -e "${GREEN}✓ C++ extension copied to python/embodik/${NC}"
else
    echo -e "${YELLOW}⚠ Warning: C++ extension not found in build directory${NC}"
fi

# Set up Python path for testing
export PYTHONPATH="${SCRIPT_DIR}/build/python_bindings:${PYTHONPATH}"

# Run Python tests if pytest is available
if python3 -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}Running Python tests...${NC}"
    cd test
    python3 -m pytest test_robot_model.py -v || true
    cd ..
else
    echo -e "${YELLOW}Pytest not found. Skipping Python tests.${NC}"
    echo "Install with: pip3 install pytest"
fi

echo -e "${GREEN}🎉 Build completed successfully!${NC}"
echo ""
echo "You can now:"
echo "  - Import the module: ${YELLOW}import embodik${NC}"
echo "  - Run tests: ${YELLOW}cd build && ctest${NC}"
echo "  - Python tests: ${YELLOW}cd test && python3 -m pytest test_robot_model.py -v${NC}"