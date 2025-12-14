#!/bin/bash
# Example 4: Different Model Sizes (Alpha Values)
# Demonstrates the size/accuracy trade-off with different alpha values

echo "========================================="
echo "Example 4: Different Model Sizes (Alpha Values)"
echo "========================================="
echo "Creating models with different alpha values..."
echo ""

ALPHAS=(0.25 0.50 0.75)
NAMES=("small" "medium" "large")

for i in "${!ALPHAS[@]}"; do
    ALPHA=${ALPHAS[$i]}
    NAME=${NAMES[$i]}
    
    echo ""
    echo "[$((i+1))/3] Creating alpha $ALPHA model ($NAME)..."
    
    python ../src/create_quantized_mobilenet_v3.py \
        --alpha "$ALPHA" \
        --input-shape "224x224x3" \
        --heads "10" \
        --output-dir "../output/examples/different_sizes" \
        --output-name "model_alpha_${ALPHA}_${NAME}" \
        --no-save-keras
    
    if [ $? -eq 0 ]; then
        echo "  Alpha $ALPHA model created"
    else
        echo "  Failed to create alpha $ALPHA model"
    fi
done

echo ""
echo "========================================="
echo "Summary:"
echo "========================================="
echo "Alpha 0.25 (Small):  ~100KB quantized  - Fastest, smallest"
echo "Alpha 0.50 (Medium): ~400KB quantized  - Balanced"
echo "Alpha 0.75 (Large):  ~900KB quantized  - Most accurate, supports pretrained"
echo ""
echo "Models saved in: ../output/examples/different_sizes/"
echo ""

