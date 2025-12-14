#!/bin/bash
# Example 5: High-Resolution Model
# Creates a model for detailed recognition tasks

echo "========================================="
echo "Example 5: High-Resolution Model"
echo "========================================="
echo "Creating a high-resolution model for detailed tasks..."
echo ""

python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.50 \
    --input-shape "320x320x3" \
    --heads "50,10" \
    --head-names "object_class,action_class" \
    --output-dir ../output/examples/high_resolution \
    --output-name "high_res_multitask" \
    --calibration-samples 200

echo ""
echo "High-resolution model created successfully!"
echo "  Output: ../output/examples/high_resolution/high_res_multitask_int8.tflite"
echo ""
echo "High-resolution models are useful for:"
echo "  - Fine-grained classification (50 object classes)"
echo "  - Action recognition (10 action classes)"
echo "  - Tasks requiring detailed visual information"
echo ""
echo "Trade-offs:"
echo "  - Larger model size"
echo "  - Slower inference"
echo "  - More memory usage"
echo "  - Better accuracy on detailed tasks"
echo ""

