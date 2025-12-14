# Quick Create: 128x128 Grayscale Multi-Head Model

To create the model with heads [5, 2, 5, 3, 3]:

```bash
python src/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "128x128x1" \
    --heads "5,2,5,3,3" \
    --output-dir ./output \
    --output-name "mnv3_128x128_grayscale_5_2_5_3_3"
```

Or use the convenience script:
```bash
bash examples/10_create_128x128_model.sh
```

**Output file**: `./output/mnv3_128x128_grayscale_5_2_5_3_3_int8.tflite`

This will create a fully uint8 quantized model with:
- Input: 128x128 grayscale (1 channel)
- Output: 5 heads with 5, 2, 5, 3, 3 classes respectively
- Alpha: 0.25 (smallest model size)
- Fully quantized: uint8 input and output
- Vela-compatible: Outputs logits (softmax applied in post-processing)
