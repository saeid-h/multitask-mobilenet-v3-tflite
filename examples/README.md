# Example Scripts

This directory contains example scripts demonstrating various capabilities of the Multi-Head MobileNet V3 quantization tool.

## Quick Start

Run any example script to see the tool in action:

```bash
bash examples/01_basic_single_head.sh
```

## Available Examples

### 01_basic_single_head.sh
Creates a simple binary classification model (e.g., person/no-person detection).
- **Use case**: Binary classification tasks
- **Model**: Single head, 2 classes
- **Size**: Smallest (alpha 0.25)

### 02_multi_head_named.sh
Demonstrates a model that performs multiple tasks simultaneously.
- **Use case**: Multi-task learning
- **Model**: 3 heads with named outputs
- **Tasks**: Object classification, person detection, age grouping

### 03_grayscale_model.sh
Creates a smaller model using grayscale input for memory-constrained devices.
- **Use case**: Resource-constrained devices
- **Input**: 128x128 grayscale (1 channel)
- **Benefits**: 3x smaller input, faster inference

### 04_different_sizes.sh
Demonstrates the size/accuracy trade-off with different alpha values.
- **Creates**: 3 models (alpha 0.25, 0.50, 0.75)
- **Shows**: Size vs accuracy trade-offs
- **Sizes**: ~100KB, ~400KB, ~900KB quantized

### 05_high_resolution.sh
Creates a model for detailed recognition tasks with higher resolution.
- **Input**: 320x320 RGB (high resolution)
- **Use case**: Fine-grained classification
- **Trade-off**: Larger size, better detail recognition

### 06_pretrained_weights.sh
Demonstrates transfer learning with ImageNet pretrained weights.
- **Requirements**: Alpha 0.75 or 1.0, RGB input
- **Benefits**: Better accuracy with limited data
- **Use case**: Similar tasks to ImageNet classification

### 07_custom_calibration.sh
Demonstrates using more calibration samples for better quantization.
- **Calibration**: 300 samples (default: 100)
- **Benefits**: Better quantization accuracy
- **Use when**: Quantization quality is critical

### 08_quantize_trained_model.sh
Demonstrates loading and quantizing an existing trained Keras model.
- **Workflow**: Train → Save → Quantize
- **Use case**: Quantize models trained on your dataset
- **Requirements**: Trained .keras model file

### 09_comprehensive_demo.sh
Runs multiple examples to showcase project capabilities.
- **Creates**: 5 different model types
- **Shows**: Full range of capabilities
- **Purpose**: Quick overview of all features

## Running Examples

### Individual Examples

Run any script directly:

```bash
cd examples
bash 01_basic_single_head.sh
```

### All Examples

Run the comprehensive demo:

```bash
bash examples/09_comprehensive_demo.sh
```

### Custom Examples

Use the scripts as templates for your own use cases:

```bash
# Edit a script to customize parameters
nano examples/01_basic_single_head.sh
```

## Output Location

All example outputs are saved to:
```
../output/examples/<example_name>/
```

Each output directory contains:
- `*_int8.tflite` - Quantized TensorFlow Lite model
- `*_report.json` - Detailed model statistics
- `*_summary.txt` - Human-readable summary
- `*_quantization_info.json` - Quantization analysis

## Common Use Cases

### Smallest Model for Microcontrollers
```bash
bash examples/03_grayscale_model.sh
```

### Multi-Task Learning
```bash
bash examples/02_multi_head_named.sh
```

### Best Accuracy with Pretrained Weights
```bash
bash examples/06_pretrained_weights.sh
```

### Custom Training + Quantization
```bash
# First train
python examples/example_training.py

# Then quantize (update path in script)
bash examples/08_quantize_trained_model.sh
```

## Next Steps

1. **Explore Examples**: Run different scripts to see capabilities
2. **Modify Scripts**: Customize for your specific needs
3. **Train Models**: Use `example_training.py` to train custom models
4. **Read Documentation**: See `docs/` directory for detailed guides

## Notes

- All scripts are executable and can be run from the repository root or examples directory
- Scripts create output directories automatically
- Use `--no-save-keras` to skip saving intermediate Keras files (saves disk space)
- Models are Vela-compatible by default for Arm Ethos-U NPU deployment

