# Tutorial

This tutorial walks you through creating quantized MobileNet V3 models with different configurations.

## Step 1: Basic Single-Head Model

Start with a simple model that has one classification head:

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --heads "2" \
    --output-dir ./tutorial_output
```

This creates a model with 2 classes. By default, it uses:
- Alpha 0.25 (smallest, fastest model)
- Input shape 224x224x3 (RGB images)
- No pretrained weights (alpha 0.25 doesn't support them)

The model will be saved as a TFLite file ready for deployment.

## Step 2: Multi-Head Model

Now create a model with multiple heads. This is useful when you need to predict multiple things from the same image:

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --heads "5,2,3" \
    --head-names "object_class,person_detection,age_group" \
    --output-dir ./tutorial_output
```

This creates three heads:
- First head: 5 classes (object classification)
- Second head: 2 classes (person detection)
- Third head: 3 classes (age group)

All heads share the same MobileNet V3 backbone, making the model efficient.

## Step 3: Custom Input Shape

You can change the input resolution and channels. For grayscale images:

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --input-shape "96x96x1" \
    --heads "2" \
    --output-dir ./tutorial_output
```

For smaller RGB images:

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --input-shape "128x128x3" \
    --heads "5,2" \
    --output-dir ./tutorial_output
```

Smaller input sizes create smaller models and faster inference, but may reduce accuracy.

## Step 4: Using Pretrained Weights

If you want to use ImageNet pretrained weights (only available for alpha 0.75 or 1.0):

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.75 \
    --heads "100,10" \
    --use-pretrained \
    --output-dir ./tutorial_output
```

Pretrained weights can improve accuracy but create larger models. Alpha 0.25 and 0.50 don't support pretrained weights in Keras.

## Step 5: Adjusting Model Size

Choose the alpha parameter based on your needs:

- Alpha 0.25: Smallest model (~100KB quantized), fastest inference
- Alpha 0.50: Medium model (~400KB quantized), balanced performance
- Alpha 0.75: Larger model (~900KB quantized), better accuracy, supports pretrained
- Alpha 1.0: Largest model (~1.5MB quantized), best accuracy, supports pretrained

```bash
# Small model
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --heads "2" \
    --output-dir ./tutorial_output/small

# Larger model
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.5 \
    --heads "2" \
    --output-dir ./tutorial_output/large
```

## Step 6: Custom Calibration

The quantization process uses a representative dataset for calibration. You can control how many samples are used:

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --heads "5,2" \
    --calibration-samples 200 \
    --output-dir ./tutorial_output
```

More samples can improve quantization quality but take longer. The default of 100 is usually sufficient.

## Step 7: Saving Intermediate Models

By default, the script saves the Keras model (.keras file) as well as the quantized TFLite model. To skip the Keras model:

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --heads "2" \
    --no-save-keras \
    --output-dir ./tutorial_output
```

## Validating Your Model

After creation, check the summary file to verify:

1. Model size matches expectations
2. All heads are present with correct class counts
3. Quantization is fully applied (uint8)
4. Model is Vela-compatible (outputs logits, softmax in post-processing)

Open the `*_summary.txt` file to see all model details. The `*_report.json` file contains machine-readable statistics.

## Next Steps

- See [Examples](examples.md) for real-world use cases
- Read [Architecture](architecture.md) to understand how it works
- Check [Quantization Guide](quantization_guide.md) for quantization details
