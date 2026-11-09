#!/bin/bash
# Example 11: Separate TFLite Export
# Export backbone and heads as separate TFLite files with different precisions

echo "========================================="
echo "Example 11: Separate TFLite Export"
echo "========================================="
echo "Creating model with separate TFLite exports..."
echo ""

python ../src/create_quantized_mobilenet_v3.py \
    --alpha 0.75 \
    --input-shape "224x224x3" \
    --heads "5,2,3" \
    --head-names "object_class,person_detection,age_group" \
    --use-pretrained \
    --separable-weights \
    --export-separate-tflite \
    --backbone-format int8 \
    --head-format fp16 \
    --output-dir ../output/examples/separate_tflite \
    --output-name "multi_precision_model" \
    --calibration-samples 150

echo ""
echo "Separate TFLite models created successfully!"
echo "  Output directory: ../output/examples/separate_tflite/"
echo ""
echo "Files created:"
echo "  1. Unified model: multi_precision_model_int8.tflite"
echo "  2. Separate models in multi_precision_model_separate_tflite/:"
echo "     - multi_precision_model_backbone_int8.tflite (quantized backbone)"
echo "     - multi_precision_model_object_class_fp16.tflite (fp16 head)"
echo "     - multi_precision_model_person_detection_fp16.tflite (fp16 head)"
echo "     - multi_precision_model_age_group_fp16.tflite (fp16 head)"
echo ""
echo "Use cases:"
echo "  - NPU deployment: int8 backbone on NPU, fp16 heads on CPU"
echo "  - Better accuracy: avoid quantization loss in final classification layers"
echo "  - Modular loading: load only needed heads for specific tasks"
echo "  - Mixed deployment: different precision requirements per component"
echo ""
echo "Inference workflow:"
echo "  1. Run backbone (int8) -> get feature map (quantized)"
echo "  2. Dequantize feature map using output quantization parameters"
echo "  3. Run head(s) (fp16) with dequantized features"
echo ""
