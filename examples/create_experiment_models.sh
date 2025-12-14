#!/bin/bash
# Create Experimental Models
# Creates all experimental models with res_ch_head naming format
# Format: {resolution}_{channels}_{num_heads}

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT" || exit 1

# Activate virtual environment if it exists
if [ -d "venv_test" ]; then
    source venv_test/bin/activate
fi

echo "========================================="
echo "Creating Experimental Models"
echo "========================================="
echo "Creating models with res_ch_head naming format..."
echo ""

BASE_DIR="output/mnv3_experiments"
COPY_DIR="output/mnv3_exp"

# Create base directories
mkdir -p "$BASE_DIR"
mkdir -p "$COPY_DIR"

# Case 1: 96x96x1 -> 2 (96 resolution, 1 channel, 1 head)
echo "[1/8] Creating 96_1_1 (96x96x1 grayscale single head)..."
python src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "96x96x1" \
    --heads "2" \
    --output-dir "$BASE_DIR/96_1_1" \
    --output-name "96_1_1" \
    --no-save-keras

if [ $? -eq 0 ]; then
    echo "  96_1_1 model created"
else
    echo "  Failed to create 96_1_1 model"
fi

# Case 2: 96x96x1 -> 5,2,5,3,2 (96 resolution, 1 channel, 5 heads)
echo ""
echo "[2/8] Creating 96_1_5 (96x96x1 grayscale multi-head)..."
python src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "96x96x1" \
    --heads "5,2,5,3,2" \
    --output-dir "$BASE_DIR/96_1_5" \
    --output-name "96_1_5" \
    --no-save-keras

if [ $? -eq 0 ]; then
    echo "  96_1_5 model created"
else
    echo "  Failed to create 96_1_5 model"
fi

# Case 3: 96x96x3 -> 2 (96 resolution, 3 channels, 1 head)
echo ""
echo "[3/8] Creating 96_3_1 (96x96x3 RGB single head)..."
python src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "96x96x3" \
    --heads "2" \
    --output-dir "$BASE_DIR/96_3_1" \
    --output-name "96_3_1" \
    --no-save-keras

if [ $? -eq 0 ]; then
    echo "  96_3_1 model created"
else
    echo "  Failed to create 96_3_1 model"
fi

# Case 4: 96x96x3 -> 5,2,5,3,2 (96 resolution, 3 channels, 5 heads)
echo ""
echo "[4/8] Creating 96_3_5 (96x96x3 RGB multi-head)..."
python src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "96x96x3" \
    --heads "5,2,5,3,2" \
    --output-dir "$BASE_DIR/96_3_5" \
    --output-name "96_3_5" \
    --no-save-keras

if [ $? -eq 0 ]; then
    echo "  96_3_5 model created"
else
    echo "  Failed to create 96_3_5 model"
fi

# Case 5: 128x128x3 -> 2 (128 resolution, 3 channels, 1 head)
echo ""
echo "[5/8] Creating 128_3_1 (128x128x3 RGB single head)..."
python src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "128x128x3" \
    --heads "2" \
    --output-dir "$BASE_DIR/128_3_1" \
    --output-name "128_3_1" \
    --no-save-keras

if [ $? -eq 0 ]; then
    echo "  128_3_1 model created"
else
    echo "  Failed to create 128_3_1 model"
fi

# Case 6: 128x128x3 -> 5,2,5,3,2 (128 resolution, 3 channels, 5 heads)
echo ""
echo "[6/8] Creating 128_3_5 (128x128x3 RGB multi-head)..."
python src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "128x128x3" \
    --heads "5,2,5,3,2" \
    --output-dir "$BASE_DIR/128_3_5" \
    --output-name "128_3_5" \
    --no-save-keras

if [ $? -eq 0 ]; then
    echo "  128_3_5 model created"
else
    echo "  Failed to create 128_3_5 model"
fi

# Case 7: 224x224x3 -> 5,2,5,3,2 (224 resolution, 3 channels, 5 heads)
echo ""
echo "[7/8] Creating 224_3_5 (224x224x3 RGB multi-head)..."
python src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "5,2,5,3,2" \
    --output-dir "$BASE_DIR/224_3_5" \
    --output-name "224_3_5" \
    --no-save-keras

if [ $? -eq 0 ]; then
    echo "  224_3_5 model created"
else
    echo "  Failed to create 224_3_5 model"
fi

# Case 8: 256x256x3 -> 5,2,5,3,2 (256 resolution, 3 channels, 5 heads)
echo ""
echo "[8/8] Creating 256_3_5 (256x256x3 RGB multi-head)..."
python src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "256x256x3" \
    --heads "5,2,5,3,2" \
    --output-dir "$BASE_DIR/256_3_5" \
    --output-name "256_3_5" \
    --no-save-keras

if [ $? -eq 0 ]; then
    echo "  256_3_5 model created"
else
    echo "  Failed to create 256_3_5 model"
fi

echo ""
echo "========================================="
echo "Copying quantized models..."
echo "========================================="

# Copy all quantized TFLite models to the copy directory with stripped names
for model_dir in "$BASE_DIR"/*/; do
    if [ -d "$model_dir" ]; then
        # Find the TFLite file (should be *_int8.tflite)
        tflite_file=$(find "$model_dir" -name "*_int8.tflite" | head -1)
        if [ -n "$tflite_file" ]; then
            # Extract the base name (directory name) and copy with .tflite extension
            dir_name=$(basename "$model_dir")
            cp "$tflite_file" "$COPY_DIR/${dir_name}.tflite"
            echo "  Copied: ${dir_name}.tflite"
        fi
    fi
done

echo ""
echo "========================================="
echo "Summary"
echo "========================================="
echo "All models use:"
echo "  - Alpha: 0.25 (smallest/fastest)"
echo "  - Quantization: Fully quantized (uint8)"
echo "  - Vela compatible: Yes (logits output, softmax in post-processing)"
echo "  - Keras models: Not saved (only TFLite)"
echo ""
echo "Naming format: {resolution}_{channels}_{num_heads}"
echo "  Example: 96_3_5 = 96x96 resolution, 3 channels (RGB), 5 heads"
echo ""
echo "Models saved in: $PROJECT_ROOT/$BASE_DIR/"
if [ -d "$BASE_DIR" ]; then
    ls -lh "$BASE_DIR" 2>/dev/null | grep "^d" | awk '{print "  " $NF}' || echo "  (no directories found)"
fi
echo ""
echo "Copied models in: $PROJECT_ROOT/$COPY_DIR/"
if [ -d "$COPY_DIR" ]; then
    ls -lh "$COPY_DIR" 2>/dev/null | grep "\.tflite$" | awk '{print "  " $NF " (" $5 ")"}' || echo "  (no .tflite files found)"
fi
echo ""
