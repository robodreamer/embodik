#!/bin/bash

# SwiftIK Installation Script
# This script builds and installs SwiftIK as a proper Python package

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}SwiftIK Installation Script${NC}"
echo "================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in a virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo -e "${GREEN}✓ Virtual environment detected: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}⚠ Warning: No virtual environment detected. Consider using 'python -m venv venv && source venv/bin/activate'${NC}"
fi

# Check for required dependencies
echo -e "${BLUE}Checking dependencies...${NC}"

# Check for cmake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}✗ CMake not found. Please install cmake.${NC}"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3 not found. Please install Python 3.8+.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Install Python build dependencies
echo -e "${BLUE}Installing Python build dependencies...${NC}"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install numpy pin
python3 -m pip install nanobind scikit-build-core[pyproject]

# Option 1: Development installation (recommended for development)
if [[ "$1" == "--dev" ]]; then
    echo -e "${BLUE}Installing in development mode...${NC}"

    # Build the C++ extension
    echo -e "${BLUE}Building C++ extension...${NC}"
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd ..

    # Install in development mode using the recommended approach
    python3 -m pip install --no-build-isolation -Ceditable.rebuild=true -ve .

    echo -e "${GREEN}✓ SwiftIK installed in development mode${NC}"
    echo -e "${YELLOW}  Changes to Python files will be reflected immediately${NC}"
    echo -e "${YELLOW}  C++ changes require rebuilding: cd build && make${NC}"

# Option 2: Regular installation
else
    echo -e "${BLUE}Installing SwiftIK package...${NC}"

    # Install using the recommended approach with scikit-build-core
    echo -e "${GREEN}Using scikit-build-core with no build isolation...${NC}"
    python3 -m pip install --no-build-isolation -Ceditable.rebuild=true -v .

    echo -e "${GREEN}✓ SwiftIK installed${NC}"
fi

# Test the installation
echo -e "${BLUE}Testing installation...${NC}"
python3 -c "
import swift_ik
print(f'SwiftIK version: {swift_ik.__version__}')
print('Available classes:')
for name in swift_ik.__all__:
    if hasattr(swift_ik, name):
        obj = getattr(swift_ik, name)
        if hasattr(obj, '__module__'):
            print(f'  ✓ {name}')
        else:
            print(f'  ✓ {name} (function)')
" && echo -e "${GREEN}✓ Installation test passed${NC}" || echo -e "${RED}✗ Installation test failed${NC}"

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Usage:"
echo "  import swift_ik"
echo "  robot = swift_ik.RobotModel('path/to/robot.urdf')"
echo "  solver = swift_ik.KinematicsSolver(robot)"
echo ""
echo "Examples:"
echo "  python examples/01_basic_ik_simple.py"
echo ""
echo "For development:"
echo "  ./install.sh --dev"
