#!/bin/bash
# Example 1: Basic Single-Head Model
# Creates a simple binary classification model (e.g., person/no-person detection)

echo "========================================="
echo "Example 1: Basic Single-Head Model"
echo "========================================="
echo "Creating a binary classification model..."
echo ""

python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "2" \
    --output-dir ../output/examples/basic_single_head \
    --output-name "person_detection_binary"

echo ""
echo "✓ Model created successfully!"
echo "  Output: ../output/examples/basic_single_head/person_detection_binary_int8.tflite"
echo ""
echo "Use case: Binary classification tasks like:"
echo "  - Person detection (person/no-person)"
echo "  - Object presence detection"
echo "  - Quality checks (good/defective)"
echo ""

