#!/bin/bash
# Create 128x128 grayscale multi-head model with heads: 5,2,5,3,3

python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "128x128x1" \
    --heads "5,2,5,3,3" \
    --output-dir ./multi_task_mobilenet_v3/output \
    --output-name "mnv3_128x128_grayscale_5_2_5_3_3"

echo ""
echo "Model created in: ./multi_task_mobilenet_v3/output/"
echo "TFLite file: mnv3_128x128_grayscale_5_2_5_3_3_int8.tflite"
