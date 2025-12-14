#!/bin/bash
# Script to create all experimental models

BASE_DIR="../outputs/experiment_models"
PROJECT_ROOT="../"

cd "$PROJECT_ROOT" || exit 1

# Activate virtual environment
source venv_mobilenet/bin/activate

# Create base directory
mkdir -p "$BASE_DIR"

echo "Creating experimental models..."
echo "================================"

# Case 1: 96x96x1 -> 2
echo ""
echo "[1/8] Creating 96x96x1 grayscale single head (2 classes)..."
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "96x96x1" \
    --heads "2" \
    --output-dir "$BASE_DIR/96x96x1_1" \
    --output-name "mnv3_96x96x1_1" \
    --no-save-keras

# Case 2: 96x96x1 -> 5,2,5,3,2
echo ""
echo "[2/8] Creating 96x96x1 grayscale multi-head (5,2,5,3,2)..."
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "96x96x1" \
    --heads "5,2,5,3,2" \
    --output-dir "$BASE_DIR/96x96x1_5" \
    --output-name "mnv3_96x96x1_5" \
    --no-save-keras

# Case 3: 96x96x3 -> 2
echo ""
echo "[3/8] Creating 96x96x3 RGB single head (2 classes)..."
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "96x96x3" \
    --heads "2" \
    --output-dir "$BASE_DIR/96x96x3_1" \
    --output-name "mnv3_96x96x3_1" \
    --no-save-keras

# Case 4: 96x96x3 -> 5,2,5,3,2
echo ""
echo "[4/8] Creating 96x96x3 RGB multi-head (5,2,5,3,2)..."
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "96x96x3" \
    --heads "5,2,5,3,2" \
    --output-dir "$BASE_DIR/96x96x3_5" \
    --output-name "mnv3_96x96x3_5" \
    --no-save-keras

# Case 5: 128x128x3 -> 2
echo ""
echo "[5/8] Creating 128x128x3 RGB single head (2 classes)..."
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "128x128x3" \
    --heads "2" \
    --output-dir "$BASE_DIR/128x128x3_1" \
    --output-name "mnv3_128x128x3_1" \
    --no-save-keras

# Case 6: 128x128x3 -> 5,2,5,3,2
echo ""
echo "[6/8] Creating 128x128x3 RGB multi-head (5,2,5,3,2)..."
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "128x128x3" \
    --heads "5,2,5,3,2" \
    --output-dir "$BASE_DIR/128x128x3_5" \
    --output-name "mnv3_128x128x3_5" \
    --no-save-keras

# Case 7: 224x224x3 -> 5,2,5,3,2
echo ""
echo "[7/8] Creating 224x224x3 RGB multi-head (5,2,5,3,2)..."
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "5,2,5,3,2" \
    --output-dir "$BASE_DIR/224x224x3_5" \
    --output-name "mnv3_224x224x3_5" \
    --no-save-keras

# Case 8: 256x256x3 -> 5,2,5,3,2
echo ""
echo "[8/8] Creating 256x256x3 RGB multi-head (5,2,5,3,2)..."
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "256x256x3" \
    --heads "5,2,5,3,2" \
    --output-dir "$BASE_DIR/256x256x3_5" \
    --output-name "mnv3_256x256x3_5" \
    --no-save-keras

echo ""
echo "================================"
echo "All models created successfully!"
echo ""
echo "Summary:"
echo "  - All models use alpha=0.25 (smallest/fastest)"
echo "  - All models are fully quantized (uint8)"
echo "  - Keras models not saved (only TFLite)"
echo ""
echo "Models are in: $BASE_DIR/"
ls -lh "$BASE_DIR" | grep "^d" | awk '{print "  "$NF}'
