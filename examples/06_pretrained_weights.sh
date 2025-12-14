#!/bin/bash
# Example 6: Using Pretrained ImageNet Weights
# Demonstrates transfer learning with pretrained weights

echo "========================================="
echo "Example 6: Pretrained ImageNet Weights"
echo "========================================="
echo "Creating model with pretrained weights..."
echo ""

python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.75 \
    --input-shape "224x224x3" \
    --heads "100,20" \
    --head-names "object_class,scene_class" \
    --use-pretrained \
    --output-dir ../output/examples/pretrained \
    --output-name "pretrained_model" \
    --calibration-samples 150

echo ""
echo "Pretrained model created successfully!"
echo "  Output: ../output/examples/pretrained/pretrained_model_int8.tflite"
echo ""
echo "Pretrained weights provide:"
echo "  - Better accuracy with limited training data"
echo "  - Faster convergence during training"
echo "  - Useful when your task is similar to ImageNet classification"
echo ""
echo "Requirements:"
echo "  - Alpha must be 0.75 or 1.0"
echo "  - Input must be RGB (3 channels)"
echo "  - Standard ImageNet preprocessing"
echo ""

