#!/bin/bash
# Example 8: Quantize a Trained Model
# Demonstrates loading and quantizing an existing trained Keras model

echo "========================================="
echo "Example 8: Quantize a Trained Model"
echo "========================================="
echo ""

# Check if a trained model exists
TRAINED_MODEL="../output/examples/basic_single_head/person_detection_binary.keras"

if [ ! -f "$TRAINED_MODEL" ]; then
    echo "ℹ️  Trained model not found at: $TRAINED_MODEL"
    echo ""
    echo "First, train a model using example_training.py:"
    echo "  python examples/example_training.py"
    echo ""
    echo "Or create an untrained model first:"
    echo "  bash examples/01_basic_single_head.sh"
    echo ""
    echo "Then update TRAINED_MODEL path in this script and run again."
    exit 0
fi

echo "Loading and quantizing trained model..."
echo "  Model: $TRAINED_MODEL"
echo ""

python ../src/create_quantized_mobilenet_v3.py \
    --keras-model-path "$TRAINED_MODEL" \
    --output-dir ../output/examples/quantized_trained \
    --output-name "quantized_from_trained" \
    --calibration-samples 200

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Trained model quantized successfully!"
    echo "  Output: ../output/examples/quantized_trained/quantized_from_trained_int8.tflite"
    echo ""
    echo "Workflow:"
    echo "  1. Train your model with example_training.py or custom training script"
    echo "  2. Save the trained model (.keras format)"
    echo "  3. Use --keras-model-path to quantize the trained model"
    echo ""
    echo "Benefits:"
    echo "  - Quantize models trained on your specific dataset"
    echo "  - Maintain training accuracy through quantization"
    echo "  - Deploy trained models to edge devices"
    echo ""
else
    echo ""
    echo "✗ Failed to quantize trained model"
    echo "  Check that the model file exists and is a valid Keras model"
    exit 1
fi

