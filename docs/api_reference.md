# API Reference

Complete documentation of all command-line parameters and options.

## Command-Line Arguments

### Required Arguments

#### `--heads` OR `--keras-model-path`

Either `--heads` (for new models) or `--keras-model-path` (for trained models) must be provided.

##### `--heads`

Comma-separated list of class counts per head. Required when creating a new model.

**Format**: `"num1,num2,num3,..."`

**Example**: `--heads "5,2,3"` creates three heads with 5, 2, and 3 classes respectively.

**Requirements**:
- At least one head must be specified
- Each head must have at least 1 class
- Values must be positive integers
- Ignored when `--keras-model-path` is provided

##### `--keras-model-path`

Path to a trained Keras model file (.keras) to quantize. Required when quantizing an existing trained model.

**Example**: `--keras-model-path ./trained_models/best_model.keras`

**Behavior**:
- Loads the trained model from the specified path
- Automatically detects input shape from the model
- Skips model creation and goes straight to quantization
- Useful for quantizing models trained on your own data

**Requirements**:
- File must be a valid Keras model (.keras format)
- Model structure must be compatible with the quantization process

#### `--output-dir`

Directory where output files will be saved.

**Example**: `--output-dir ./models`

The directory will be created if it doesn't exist.

### Optional Arguments

#### `--alpha`

Width multiplier controlling model size and capacity.

**Options**: `0.25`, `0.50`, `0.75`, `1.0`

**Default**: `0.25`

**Guidelines**:
- 0.25: Smallest model (~100KB quantized), fastest inference
- 0.50: Medium model (~400KB quantized)
- 0.75: Larger model (~900KB quantized), supports pretrained weights
- 1.0: Largest model (~1.5MB quantized), supports pretrained weights

#### `--input-shape`

Input image dimensions and channels.

**Format**: `"HxWxC"` where H=height, W=width, C=channels

**Default**: `"224x224x3"` (RGB images at 224x224 resolution)

**Examples**:
- `"224x224x3"` - RGB images, standard ImageNet size
- `"96x96x1"` - Grayscale images, smaller size
- `"128x128x3"` - RGB images, smaller size

**Requirements**:
- Height and width must be at least 32 pixels
- Channels must be 1 (grayscale) or 3 (RGB)
- Format must match exactly: numbers separated by 'x'

#### `--head-names`

Optional names for each head. If not provided, heads are named automatically (head_1, head_2, etc.).

**Format**: `"name1,name2,name3,..."`

**Example**: `--head-names "object_class,person_detection,age_group"`

**Requirements**:
- Number of names must match number of heads
- Names are used in model output and reports

#### `--output-name`

Base name for output files. If not provided, generated automatically from configuration.

**Example**: `--output-name "my_custom_model"`

Output files will be:
- `my_custom_model_int8.tflite`
- `my_custom_model_report.json`
- `my_custom_model_summary.txt`
- etc.

#### `--use-pretrained`

Use ImageNet pretrained weights for the backbone.

**Default**: Not used (False)

**Requirements**:
- Only works with alpha 0.75 or 1.0
- Requires RGB input (3 channels)
- Backbone will be frozen (non-trainable)

**Example**: `--alpha 0.75 --use-pretrained`

#### `--calibration-samples`

Number of samples used for quantization calibration.

**Default**: `100`

**Range**: Typically 50-500

**Guidelines**:
- More samples can improve quantization quality
- More samples take longer to process
- 100 is usually sufficient for most cases
- Use more (200-300) for complex models

#### `--save-keras` / `--no-save-keras`

Whether to save the intermediate Keras model file.

**Default**: `--save-keras` (saves the .keras file)

Use `--no-save-keras` to skip saving the Keras model and only keep the TFLite file.

#### `--vela-compatible` / `--with-softmax`

Control whether the model includes softmax activation.

**Default**: `--vela-compatible` (enabled by default)

- **`--vela-compatible`** (default): Model outputs logits (linear activation). Softmax must be applied in post-processing. Compatible with Arm Vela compiler for Ethos-U NPU deployment.
- **`--with-softmax`**: Model includes softmax activation. Outputs probabilities directly. Not compatible with Vela compilation.

For deployment on Arm Ethos-U NPU or when using Vela compiler, use the default Vela-compatible mode.

## Output Files

All output files are saved in the specified `--output-dir` directory.

### `{output_name}_int8.tflite`

The quantized TensorFlow Lite model file. This is the main output, ready for deployment.

**Format**: TensorFlow Lite FlatBuffer
**Input**: uint8 tensor(s)
**Output**: uint8 tensor(s) representing logits (softmax applied in post-processing for Vela-compatible models)

### `{output_name}.keras`

The Keras model file (if `--save-keras` is used).

**Format**: Keras SavedModel format
**Use**: Can be loaded in Python for further training or inspection

### `{output_name}_report.json`

Machine-readable model statistics and metadata.

**Contains**:
- Model architecture information
- Parameter counts (total, trainable, backbone, heads)
- Head configurations
- Quantization status
- Output shape information

### `{output_name}_summary.txt`

Human-readable text summary of the model.

**Contains**:
- Model name and configuration
- Parameter statistics
- Head details
- Quantization status

### `{output_name}_quantization_info.json`

Detailed quantization analysis.

**Contains**:
- Input/output tensor details
- Quantization parameters (scales, zero points)
- Tensor type distribution (uint8, int8 vs FP32)
- Full quantization status

## Return Codes

- `0`: Success
- `1`: Error (invalid arguments, model creation failed, etc.)

## Usage Modes

The script supports two modes:

### Mode 1: Create New Model

Create a new model from scratch and quantize it:

```bash
python src/create_quantized_mobilenet_v3.py \
    --heads "5,2,3" \
    --output-dir ./models
```

### Mode 2: Quantize Trained Model

Load an existing trained model and quantize it:

```bash
python src/create_quantized_mobilenet_v3.py \
    --keras-model-path ./trained_model.keras \
    --output-dir ./models
```

## Examples

See [Examples](examples.md) for complete usage examples with different configurations. Also check the `examples/` directory for ready-to-run example scripts.
