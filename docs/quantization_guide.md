# Quantization Guide

Understanding how uint8 quantization works and why it matters.

## What is Quantization?

Quantization converts floating-point weights and activations to integers. Instead of storing each weight as a 32-bit float (4 bytes), we store it as an 8-bit integer (1 byte). This reduces model size by ~4x and can speed up inference on hardware that supports integer operations.

## Why Quantize?

**Size Reduction**: A 1MB FP32 model becomes ~250KB when quantized to uint8/int8.

**Speed**: Many edge devices (like microcontrollers) run integer math faster than floating-point.

**Deployment**: Smaller models fit in limited storage and memory.

**Trade-off**: Small accuracy loss (typically 1-3%) in exchange for significant size and speed benefits.

## How Quantization Works

The quantization process maps floating-point values to integers using:

```
quantized_value = round(float_value / scale) + zero_point
```

Where:
- `scale`: The quantization step size
- `zero_point`: The integer value representing zero

To convert back:
```
float_value = (quantized_value - zero_point) * scale
```

## Calibration

During quantization, the converter needs to know the range of values in your model. This is where the representative dataset comes in.

The calibration process:
1. Runs your model on representative samples
2. Observes the range of values in each layer
3. Chooses appropriate scales and zero points
4. Converts weights and activation ranges to uint8

More calibration samples give better estimates of value ranges, potentially improving quantization quality.

## Input/Output Quantization

The quantized models use uint8 for both inputs and outputs:

**Input**: Your preprocessing should produce uint8 values (0-255). The model expects these directly, no float conversion needed.

**Output**: Model outputs are uint8 but represent quantized logits (not probabilities). Models are Vela-compatible by default. To use them:

1. Get quantization parameters from the model interpreter
2. Convert uint8 outputs to float logits: `logits = (uint8_value - zero_point) * scale`
3. Apply softmax: `probabilities = softmax(logits)` (required for Vela-compatible models)

## Checking Quantization

After creating a model, check the `*_quantization_info.json` file to verify:

- `fully_quantized: true` means all tensors are quantized (uint8 or int8)
- `uint8_tensors` count should include input/output tensors
- `int8_tensors` count should be high (internal weights)
- `float32_tensors` should be 0 (or very low if some operations can't be quantized)

## Quantization Quality

Good quantization means:
- Minimal accuracy loss (< 3% typical)
- All operations quantized (no float fallbacks)
- Appropriate value ranges (no saturation)

If quantization quality is poor:
- Try more calibration samples (200-300)
- Check if your model has unusual activation patterns
- Consider using a larger alpha for more capacity
- Some operations may inherently require float (rare)

## Model Size Calculation

Rough estimates for quantized model sizes:

- Alpha 0.25 backbone: ~25KB (quantized)
- Alpha 0.50 backbone: ~100KB (quantized)
- Alpha 0.75 backbone: ~225KB (quantized)
- Alpha 1.0 backbone: ~375KB (quantized)

Plus:
- Each head: ~250 bytes per class (quantized)
- Quantization overhead: ~5-10KB for scales/zero points
- Model structure: ~10-20KB

Total size ≈ backbone_size + (sum of head_classes * 250) + overhead

## Deployment Considerations

When deploying quantized models:

1. **Input preprocessing**: Must produce uint8 values matching the model's expected input quantization
2. **Output post-processing**: Convert uint8 outputs back to float using quantization parameters
3. **Hardware support**: Ensure your target device supports uint8/int8 operations efficiently (including Arm Ethos-U NPU via Vela compiler)
4. **Testing**: Always test quantized models on real data to verify accuracy

Most modern edge devices (ARM processors, microcontrollers with ML accelerators, Arm Ethos-U NPU) support uint8/int8 inference efficiently. Models are generated Vela-compatible by default for optimal hardware accelerator support.
