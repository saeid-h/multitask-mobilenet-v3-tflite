#!/bin/bash
# Example usage scripts for creating quantized MobileNet V3 models

# Example 1: Single head model (2 classes)
echo "Creating single-head model..."
python examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "2" \
    --output-dir ./output/single_head

# Example 2: Multi-head model with 3 heads
echo "Creating multi-head model..."
python examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "5,2,3" \
    --head-names "object_class,person_detection,age_group" \
    --output-dir ./output/multi_head

# Example 3: Larger model with alpha 0.5
echo "Creating larger model..."
python examples/create_quantized_mobilenet_v3.py \
    --alpha 0.5 \
    --input-shape "224x224x3" \
    --heads "10,5,2" \
    --output-dir ./output/larger_model

# Example 4: Grayscale input model
echo "Creating grayscale model..."
python examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "96x96x1" \
    --heads "2" \
    --output-dir ./output/grayscale

# Example 5: Using pretrained weights (alpha 0.75 or 1.0 only)
echo "Creating model with pretrained weights..."
python examples/create_quantized_mobilenet_v3.py \
    --alpha 0.75 \
    --input-shape "224x224x3" \
    --heads "100,10" \
    --use-pretrained \
    --output-dir ./output/pretrained

# Example 6: Custom calibration samples
echo "Creating model with custom calibration..."
python examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "5,2" \
    --calibration-samples 200 \
    --output-dir ./output/custom_calibration
