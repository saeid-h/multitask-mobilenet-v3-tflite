# Getting Started

This guide helps you set up and create your first quantized MobileNet V3 model.

## Prerequisites

You'll need:
- Python 3.7 or later
- TensorFlow 2.x
- NumPy

The tool uses the models package from this repository, so make sure you're running it from the repository root or have the models package in your Python path.

## Installation

No separate installation needed. The script uses the existing dependencies from the repository. Just ensure TensorFlow and NumPy are installed:

```bash
pip install tensorflow numpy
```

## Creating Your First Model

The simplest command creates a model with a single classification head:

```bash
python src/create_quantized_mobilenet_v3.py \
    --heads "2" \
    --output-dir ./my_models
```

Or use one of the example scripts:

```bash
bash examples/01_basic_single_head.sh
```

This creates a model with 2 output classes (like person/no-person detection). The output includes:

- A quantized TFLite model file (`.tflite`)
- A JSON report with model statistics
- A text summary file
- Quantization analysis information

## Understanding the Output

After running the script, you'll find several files in your output directory:

- `{model_name}_int8.tflite` - The quantized model ready for deployment (Vela-compatible, outputs logits)
- `{model_name}_report.json` - Detailed model statistics in JSON format
- `{model_name}_summary.txt` - Human-readable summary of the model
- `{model_name}_quantization_info.json` - Quantization analysis details

The model name is auto-generated from your configuration. For example, a model with alpha 0.25, heads [5,2], and input shape 224x224x3 would be named `mnv3_0_25_5_2_224x224x3`.

## Example Scripts

Ready-to-use example scripts are available in the `examples/` directory. Run them to quickly create different model types:

```bash
# Basic single-head model
bash examples/01_basic_single_head.sh

# Multi-head model
bash examples/02_multi_head_named.sh

# See all examples
ls examples/*.sh
```

Check `examples/README.md` for descriptions of each example script.

## Next Steps

- Follow the [Tutorial](tutorial.md) for a detailed walkthrough
- Check [Examples](examples.md) for different use cases
- Read the [API Reference](api_reference.md) for all available options
