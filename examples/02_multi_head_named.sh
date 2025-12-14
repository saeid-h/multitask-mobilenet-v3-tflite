#!/bin/bash
# Example 2: Multi-Head Model with Named Heads
# Demonstrates a model that performs multiple tasks simultaneously

echo "========================================="
echo "Example 2: Multi-Head Model with Named Heads"
echo "========================================="
echo "Creating a multi-task model..."
echo ""

python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "5,2,3" \
    --head-names "object_class,person_detection,age_group" \
    --output-dir ../output/examples/multi_head_named \
    --output-name "multitask_model"

echo ""
echo "Model created successfully!"
echo "  Output: ../output/examples/multi_head_named/multitask_model_int8.tflite"
echo ""
echo "This model performs 3 tasks simultaneously:"
echo "  - Head 1: Object classification (5 classes)"
echo "  - Head 2: Person detection (2 classes: person/no-person)"
echo "  - Head 3: Age group classification (3 classes)"
echo ""
echo "All tasks share the same backbone, reducing model size!"
echo ""

