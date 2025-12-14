#!/bin/bash
# Example 9: Comprehensive Demo
# Runs multiple examples to showcase project capabilities

echo "========================================="
echo "Example 9: Comprehensive Demo"
echo "========================================="
echo "This script demonstrates various capabilities of the project."
echo ""

# Create output directory
OUTPUT_DIR="../output/examples/demo"
mkdir -p "$OUTPUT_DIR"

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "Running demonstration examples..."
echo ""

# Example 1: Basic model
echo "[1/5] Creating basic single-head model..."
python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "2" \
    --output-dir "$OUTPUT_DIR" \
    --output-name "demo_basic" \
    --no-save-keras > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Basic model created"
else
    echo "  ✗ Failed"
fi

# Example 2: Multi-head
echo "[2/5] Creating multi-head model..."
python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "5,2,3" \
    --head-names "object,person,age" \
    --output-dir "$OUTPUT_DIR" \
    --output-name "demo_multi" \
    --no-save-keras > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Multi-head model created"
else
    echo "  ✗ Failed"
fi

# Example 3: Grayscale
echo "[3/5] Creating grayscale model..."
python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "128x128x1" \
    --heads "2" \
    --output-dir "$OUTPUT_DIR" \
    --output-name "demo_grayscale" \
    --no-save-keras > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Grayscale model created"
else
    echo "  ✗ Failed"
fi

# Example 4: Different alpha
echo "[4/5] Creating medium-size model (alpha 0.50)..."
python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.50 \
    --input-shape "224x224x3" \
    --heads "10" \
    --output-dir "$OUTPUT_DIR" \
    --output-name "demo_medium" \
    --no-save-keras > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Medium model created"
else
    echo "  ✗ Failed"
fi

# Example 5: Custom calibration
echo "[5/5] Creating high-quality quantized model..."
python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "5,2" \
    --calibration-samples 200 \
    --output-dir "$OUTPUT_DIR" \
    --output-name "demo_high_quality" \
    --no-save-keras > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ High-quality model created"
else
    echo "  ✗ Failed"
fi

echo ""
echo "========================================="
echo "Demo Complete!"
echo "========================================="
echo "Models created in: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR"/*.tflite 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Explore the project capabilities:"
echo "  - See individual example scripts (01-08) for details"
echo "  - Check model reports in the output directory"
echo "  - Try training a model with example_training.py"
echo ""

