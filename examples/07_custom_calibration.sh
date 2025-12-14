#!/bin/bash
# Example 7: Custom Calibration Samples
# Demonstrates using more calibration samples for better quantization

echo "========================================="
echo "Example 7: Custom Calibration Samples"
echo "========================================="
echo "Creating model with high-quality quantization..."
echo ""

python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "10,5,3" \
    --head-names "category,subcategory,type" \
    --calibration-samples 300 \
    --output-dir ../output/examples/custom_calibration \
    --output-name "high_quality_quantized"

echo ""
echo "✓ High-quality quantized model created successfully!"
echo "  Output: ../output/examples/custom_calibration/high_quality_quantized_int8.tflite"
echo ""
echo "Custom calibration (300 samples) provides:"
echo "  - Better quantization accuracy"
echo "  - More accurate activation range estimates"
echo "  - Potentially lower accuracy loss from quantization"
echo ""
echo "Use when:"
echo "  - Quantization quality is critical"
echo "  - Model has complex activation patterns"
echo "  - You can afford longer quantization time"
echo ""
echo "Default: 100 samples (faster, usually sufficient)"
echo ""

