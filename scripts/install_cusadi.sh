#!/bin/bash
# Install CusADi from GitHub
# This script clones and installs CusADi for GPU-accelerated CasADi

CUSADI_DIR="${CUSADI_DIR:-$HOME/.local/cusadi}"

echo "Installing CusADi..."
echo "  Target directory: $CUSADI_DIR"

if [ ! -d "$CUSADI_DIR" ]; then
    echo "  Cloning CusADi repository..."
    git clone https://github.com/se-hwan/cusadi.git "$CUSADI_DIR"
else
    echo "  CusADi already cloned, pulling latest..."
    cd "$CUSADI_DIR" && git pull
fi

echo "  Installing CusADi in editable mode..."
pip install -e "$CUSADI_DIR"

echo ""
echo "CusADi installed successfully!"
echo ""
echo "To compile a CasADi function for GPU:"
echo "  1. Export the function:"
echo "     pixi run -e cuda export-casadi"
echo ""
echo "  2. Compile to CUDA kernel:"
echo "     mv fn_velocity_solve.casadi $CUSADI_DIR/src/casadi_functions/"
echo "     cd $CUSADI_DIR && python run_codegen.py --fn=fn_velocity_solve"
echo ""
echo "  3. Verify installation:"
echo "     pixi run -e cuda check-gpu"
