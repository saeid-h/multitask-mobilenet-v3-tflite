# Multi-Head MobileNet V3 Quantization Tool

Create quantized MobileNet V3 models with custom multi-head classification outputs. This tool generates fully quantized TensorFlow Lite models (uint8 input/output) from MobileNet V3 architectures with your choice of classification heads. Models are Vela-compatible by default for Arm Ethos-U NPU deployment.

## Quick Start

Install dependencies:
```bash
pip install tensorflow numpy
```

Create a model with multiple heads:
```bash
python examples/create_quantized_mobilenet_v3.py \
    --heads "5,2,3" \
    --output-dir ./models
```

This creates a model with three heads: 5 classes, 2 classes, and 3 classes, saved as a quantized TFLite file.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy

The script uses the existing models package from this repository.

See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed setup instructions.

## Documentation

Full documentation is available in the [docs/](docs/) directory:

- [Getting Started](docs/getting_started.md) - Installation and first steps
- [Tutorial](docs/tutorial.md) - Step-by-step guide with examples
- [Training Guide](docs/training_guide.md) - Using as a module and training with multiple datasets
- [API Reference](docs/api_reference.md) - Complete parameter documentation
- [Examples](docs/examples.md) - Various use cases and configurations
- [Architecture](docs/architecture.md) - How multi-head MobileNet V3 works
- [Quantization Guide](docs/quantization_guide.md) - Understanding quantization
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## Basic Usage

```bash
python examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "2,5" \
    --output-dir ./output
```

See [docs/getting_started.md](docs/getting_started.md) for more details.
