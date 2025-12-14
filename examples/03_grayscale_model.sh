#!/bin/bash
# Example 3: Grayscale Model for Constrained Devices
# Creates a smaller model using grayscale input

echo "========================================="
echo "Example 3: Grayscale Model"
echo "========================================="
echo "Creating a grayscale model for memory-constrained devices..."
echo ""

python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "128x128x1" \
    --heads "2" \
    --output-dir ../output/examples/grayscale \
    --output-name "grayscale_person_detection" \
    --no-save-keras

echo ""
echo "Model created successfully!"
echo "  Output: ../output/examples/grayscale/grayscale_person_detection_int8.tflite"
echo ""
echo "Benefits of grayscale models:"
echo "  - 3x smaller input size (1 channel vs 3 channels)"
echo "  - Smaller overall model size"
echo "  - Faster inference"
echo "  - Lower memory usage"
echo ""
echo "Use case: Resource-constrained devices like microcontrollers"
echo ""

