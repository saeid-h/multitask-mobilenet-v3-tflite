# Architecture

How multi-head MobileNet V3 models work and what makes them efficient.

## Overview

Multi-head MobileNet V3 uses a shared backbone (feature extractor) with multiple classification heads. The backbone processes the input image once, and each head makes predictions based on the same features.

## Backbone: MobileNet V3

The backbone is a MobileNet V3 Small architecture, which is optimized for mobile and edge devices. Key features:

- Depthwise separable convolutions reduce computation
- Squeeze-and-Excitation blocks improve feature quality
- Hard-swish activations for efficiency
- Width multiplier (alpha) controls model capacity

The backbone extracts features from the input image and produces a feature map. This happens once, regardless of how many heads you have.

## Classification Heads

Each head is a small neural network attached to the backbone output:

1. Global Average Pooling - reduces spatial dimensions to a single vector
2. Optional Dropout - prevents overfitting
3. Dense layer - final classification with linear activation (outputs logits; softmax applied in post-processing for Vela compatibility)

All heads share the same backbone features but have separate final layers. This means:
- Adding more heads adds minimal computation
- All predictions use the same underlying features
- Model size grows linearly with number of heads

**Note**: Models default to linear activation (no softmax) for Vela compiler compatibility. Apply softmax in post-processing when interpreting outputs.

## Model Size

Model size depends on:

**Backbone**: Determined by alpha parameter
- Alpha 0.25: ~100K parameters
- Alpha 0.50: ~400K parameters
- Alpha 0.75: ~900K parameters
- Alpha 1.0: ~1.5M parameters

**Heads**: Each head adds roughly ~1000 parameters per class
- Head with 2 classes: ~2K parameters
- Head with 10 classes: ~10K parameters
- Head with 100 classes: ~100K parameters

**Quantization**: Reduces size by ~4x
- FP32 weights: 4 bytes per parameter
- uint8/int8 weights: 1 byte per parameter
- Plus quantization overhead (scales, zero points)

Total quantized size ≈ (backbone_params + head_params) / 4 + overhead

## Input Processing

The model expects:
- Input shape: (batch, height, width, channels)
- Channels: 1 (grayscale) or 3 (RGB)
- Pixel values: Typically normalized to [0, 1] or [0, 255]

After quantization, inputs must be uint8. Preprocessing should convert your input images to match the expected format.

## Output Structure

For multi-head models, outputs are a dictionary:
```python
{
    'head_1': <tensor with shape (batch, num_classes_1)>,
    'head_2': <tensor with shape (batch, num_classes_2)>,
    ...
}
```

Each output tensor contains logits (linear activation, no softmax). After quantization, outputs are uint8 but represent quantized float logits. Use quantization parameters to convert back to float logits, then apply softmax to get probabilities. Models are Vela-compatible by default.

## Performance Characteristics

**Inference Speed**:
- Backbone processing: ~90% of total time
- Head processing: ~10% of total time (shared across all heads)
- Adding heads has minimal impact on speed

**Memory Usage**:
- Backbone features: Main memory consumer
- Head weights: Small compared to backbone
- Activation memory: Depends on input size

**Accuracy Trade-offs**:
- Smaller alpha: Faster, smaller, less accurate
- Larger alpha: Slower, larger, more accurate
- Pretrained weights: Better accuracy, larger model, requires RGB input

## Comparison with Separate Models

Using one multi-head model instead of multiple single-head models:

**Advantages**:
- Shared feature extraction (compute once)
- Smaller total model size
- Consistent features across tasks
- Single inference pass

**Disadvantages**:
- Tasks must be compatible (similar image types)
- All heads use same feature resolution
- Less flexibility in architecture per task

Multi-head models work well when tasks are related (e.g., person detection + gender classification). For unrelated tasks, separate models might be better.
