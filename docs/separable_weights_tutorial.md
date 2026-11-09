# Separable Weights Tutorial

A comprehensive guide to using the separable weights feature for modular model management, transfer learning, and efficient inference.

## Overview

The **separable weights** feature allows you to:

1. **Save/Load Independently**: Save backbone and heads as separate weight files
2. **Transfer Learning**: Freeze backbone, train only heads
3. **Dynamic Extension**: Add new heads without retraining backbone
4. **Feature Caching**: Extract features once, run multiple heads
5. **Modular Deployment**: Share backbones across different configurations

## Quick Start

### Enable Separable Weights

```python
from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import create_head_config_from_list
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture

# Create configuration with separable_weights=True
head_configs = create_head_config_from_list(
    [5, 2, 3],
    ["object_class", "person_detection", "age_group"]
)

config = MultiHeadModelConfig(
    input_shape=(128, 128, 3),
    head_configs=head_configs,
    arch_params={'alpha': 0.25},
    separable_weights=True  # Enable the feature
)

architecture = MultiHeadMobileNetV3QATArchitecture(config)
model = architecture.get_model()

# Verify it's enabled
print(f"Separable weights: {architecture.is_separable}")  # True
```

### Command-Line Usage

```bash
# Create model with separable weights
python src/create_quantized_mobilenet_v3.py \
    --heads "5,2,3" \
    --head-names "object_class,person_detection,age_group" \
    --separable-weights \
    --save-separate-weights \
    --output-dir ./models

# Output files:
# ./models/xxx_int8.tflite
# ./models/xxx_weights/backbone_weights.h5
# ./models/xxx_weights/object_class_weights.h5
# ./models/xxx_weights/person_detection_weights.h5
# ./models/xxx_weights/age_group_weights.h5
```

### Separate TFLite Export

Export backbone and heads as separate TFLite files with different precision formats:

```bash
# Export with mixed precision (int8 backbone, fp16 heads)
python src/create_quantized_mobilenet_v3.py \
    --heads "5,2,3" \
    --head-names "object_class,person_detection,age_group" \
    --separable-weights \
    --export-separate-tflite \
    --backbone-format int8 \
    --head-format fp16 \
    --output-dir ./models

# Output files:
# ./models/xxx_int8.tflite (unified model)
# ./models/xxx_separate_tflite/xxx_backbone_int8.tflite
# ./models/xxx_separate_tflite/xxx_object_class_fp16.tflite
# ./models/xxx_separate_tflite/xxx_person_detection_fp16.tflite
# ./models/xxx_separate_tflite/xxx_age_group_fp16.tflite
```

## Saving and Loading Weights

### Save All Weights Separately

```python
# Save backbone and all heads to a directory
saved_files = architecture.save_all_weights_separately('./model_weights')

# Creates:
# ./model_weights/backbone_weights.h5
# ./model_weights/object_class_weights.h5
# ./model_weights/person_detection_weights.h5
# ./model_weights/age_group_weights.h5
```

### Load All Weights

```python
# Create a new architecture with same configuration
new_architecture = MultiHeadMobileNetV3QATArchitecture(config)
new_model = new_architecture.get_model()

# Load all weights
new_architecture.load_all_weights_separately('./model_weights')
```

### Save/Load Individual Components

```python
# Save just the backbone
architecture.save_backbone_weights('./backbone_only.h5')

# Save a specific head
architecture.save_head_weights('object_class', './object_class.h5')

# Load into new model
new_architecture.load_backbone_weights('./backbone_only.h5')
new_architecture.load_head_weights('object_class', './object_class.h5')
```

## Transfer Learning

### Freeze Backbone, Train Heads

```python
# Freeze the backbone (shared feature extractor)
architecture.freeze_backbone()

# Compile and train - only heads will be updated
model.compile(
    optimizer='adam',
    loss={h.name: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
          for h in architecture.head_configs},
    metrics=['accuracy']
)

model.fit(train_data, epochs=10)

# Unfreeze for fine-tuning
architecture.unfreeze_backbone()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), ...)
model.fit(train_data, epochs=5)
```

### Freeze Specific Heads

```python
# Freeze some heads, train others
architecture.freeze_backbone()
architecture.freeze_head('object_class')
architecture.freeze_head('person_detection')
# age_group head remains trainable

# Check trainable status
status = architecture.get_trainable_status()
# {'backbone': False, 'object_class': False, 'person_detection': False, 'age_group': True}
```

### Pretrained Backbone with New Heads

```python
# Step 1: Train and save a backbone
original_config = MultiHeadModelConfig(
    input_shape=(128, 128, 3),
    head_configs=create_head_config_from_list([5, 2], ['task1', 'task2']),
    arch_params={'alpha': 0.25},
    separable_weights=True
)
original_arch = MultiHeadMobileNetV3QATArchitecture(original_config)
original_model = original_arch.get_model()

# Train...
original_model.fit(original_data, epochs=20)

# Save backbone
original_arch.save_backbone_weights('./pretrained_backbone.h5')

# Step 2: Create new model with different heads
new_config = MultiHeadModelConfig(
    input_shape=(128, 128, 3),
    head_configs=create_head_config_from_list([3, 4, 2], ['new_task1', 'new_task2', 'new_task3']),
    arch_params={'alpha': 0.25},
    separable_weights=True
)
new_arch = MultiHeadMobileNetV3QATArchitecture(new_config)
new_model = new_arch.get_model()

# Load pretrained backbone
new_arch.load_backbone_weights('./pretrained_backbone.h5')

# Freeze backbone, train only new heads
new_arch.freeze_backbone()
new_model.compile(...)
new_model.fit(new_data, epochs=10)
```

## Dynamic Head Extension

Add new classification heads to an existing model without retraining the backbone.

### Add a New Head

```python
# Original model with 3 heads
print(f"Current heads: {architecture.head_names}")
# ['object_class', 'person_detection', 'age_group']

# Add a new head
new_model = architecture.add_head_dynamically(
    num_classes=4,
    head_name='emotion',
    activation='linear',
    freeze_backbone=True  # Don't modify backbone
)

print(f"Heads after adding: {architecture.head_names}")
# ['object_class', 'person_detection', 'age_group', 'emotion']
```

### Train Only the New Head

```python
# Freeze existing heads
architecture.freeze_head('object_class')
architecture.freeze_head('person_detection')
architecture.freeze_head('age_group')
# 'emotion' head remains trainable

# Compile with weights that only train new head
new_model.compile(
    optimizer='adam',
    loss={
        'object_class': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'person_detection': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'age_group': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'emotion': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    },
    loss_weights={
        'object_class': 0.0,      # Frozen, don't contribute to loss
        'person_detection': 0.0,
        'age_group': 0.0,
        'emotion': 1.0            # Only train this
    }
)

# Train with data that includes emotion labels
new_model.fit(train_data_with_emotion, epochs=10)
```

## Feature Caching

The `FeatureCacheManager` enables efficient inference by extracting features once and reusing them with multiple heads.

### Basic Usage

```python
from models.utils import FeatureCacheManager

# Create cache manager
cache_manager = FeatureCacheManager(architecture)

# Extract features from images
images = np.random.rand(16, 128, 128, 3).astype(np.float32)
features = cache_manager.extract_features(images)
print(f"Features shape: {features.shape}")  # (16, 4, 4, 192)
```

### Run Inference with Cached Features

```python
# Load and run one head
cache_manager.load_head('object_class')
predictions = cache_manager.predict_with_cached_features('object_class')
print(f"Predictions: {predictions.shape}")  # (16, 5)

# Load another head and run
cache_manager.load_head('person_detection')
predictions = cache_manager.predict_with_cached_features('person_detection')

# Run all loaded heads at once
all_preds = cache_manager.predict_all_loaded_heads()
# {'object_class': array(...), 'person_detection': array(...)}
```

### Memory Management

```python
# Unload heads to free memory
cache_manager.unload_head('object_class')

# Or unload all
cache_manager.unload_all_heads()

# Get memory usage info
memory_info = cache_manager.get_memory_usage()
print(f"Cached features: {memory_info['cached_features']['size_mb']:.2f} MB")
print(f"Backbone: {memory_info['backbone']['parameters']:,} params")
```

### Save/Load Features

```python
# Save features to disk
cache_manager.save_features('./features.npy')

# Load later
cache_manager.load_features('./features.npy')

# Clear in-memory cache
cache_manager.clear_cache()
```

### Performance Comparison

```python
# Compare full model vs cached features
results = cache_manager.compare_performance(
    test_images,
    head_names=['object_class', 'person_detection'],
    num_runs=10
)

print(f"Full model time: {results['full_model']['time_per_batch']*1000:.2f} ms")
print(f"Cached time: {results['speedup']['total_cached_time']*1000:.2f} ms")
print(f"Speedup: {results['speedup']['speedup_factor']:.2f}x")
```

## Separate TFLite Inference

When using `--export-separate-tflite`, inference requires two steps: backbone → heads.

### Python API for Separate Export

```python
from src.utils import export_separate_tflite_models

# Export separate models
results = export_separate_tflite_models(
    architecture,
    input_shape=(224, 224, 3),
    output_dir="./separate_models",
    base_name="my_model",
    backbone_format="int8",
    head_format="fp16",
    calibration_samples=100
)

print(f"Backbone: {results['backbone']['file']}")
for head_name, head_info in results['heads'].items():
    print(f"Head '{head_name}': {head_info['file']}")
```

### Two-Step Inference

```python
import tensorflow as tf
import numpy as np

# Load interpreters
backbone = tf.lite.Interpreter("my_model_backbone_int8.tflite")
backbone.allocate_tensors()

head = tf.lite.Interpreter("my_model_object_class_fp16.tflite")
head.allocate_tensors()

# Get tensor details
bb_input = backbone.get_input_details()[0]
bb_output = backbone.get_output_details()[0]
head_input = head.get_input_details()[0]
head_output = head.get_output_details()[0]

# Step 1: Run backbone (uint8 input/output)
image_uint8 = preprocess_image_to_uint8(image)  # Your preprocessing
backbone.set_tensor(bb_input['index'], image_uint8)
backbone.invoke()
features_quantized = backbone.get_tensor(bb_output['index'])

# Step 2: Dequantize backbone output for fp16 head
scale, zero_point = bb_output['quantization_parameters']['scales'][0], bb_output['quantization_parameters']['zero_points'][0]
features_float = (features_quantized.astype(np.float32) - zero_point) * scale

# Step 3: Run head (fp32 input/output)
head.set_tensor(head_input['index'], features_float)
head.invoke()
logits = head.get_tensor(head_output['index'])

# Step 4: Apply softmax for probabilities
probabilities = tf.nn.softmax(logits).numpy()
```

### Deployment Benefits

- **NPU + CPU**: Run int8 backbone on NPU, fp16 heads on CPU
- **Better accuracy**: Avoid quantization loss in classification layers
- **Modular loading**: Load only needed heads for specific tasks
- **Memory efficiency**: Keep only active heads in memory

## Demo Applications

### Jupyter Notebook Demo

Run the interactive notebook demo:

```bash
cd examples
jupyter notebook demo_separable_weights.ipynb
```

The notebook covers:
1. Model creation with separable weights
2. Separate weight saving/loading
3. Feature caching
4. Dynamic head operations
5. Training control with frozen components
6. Memory and performance analysis

### Gradio Web UI Demo

Run the web-based demo:

```bash
cd examples
python demo_separable_weights_ui.py
```

Then open the displayed URL in your browser. The UI provides:
- Model setup and configuration
- Image upload and feature extraction
- Dynamic head inference
- Performance metrics visualization

## API Reference

### Architecture Methods

| Method | Description |
|--------|-------------|
| `architecture.is_separable` | Property: Check if separable weights enabled |
| `architecture.backbone` | Property: Get backbone model for direct access |
| `architecture.head(name)` | Get a specific head model |
| `architecture.save_backbone_weights(path)` | Save backbone weights |
| `architecture.load_backbone_weights(path)` | Load backbone weights |
| `architecture.save_head_weights(name, path)` | Save specific head weights |
| `architecture.load_head_weights(name, path)` | Load specific head weights |
| `architecture.save_all_weights_separately(dir)` | Save all weights |
| `architecture.load_all_weights_separately(dir)` | Load all weights |
| `architecture.add_head_dynamically(...)` | Add new head |
| `architecture.freeze_backbone()` | Freeze backbone weights |
| `architecture.unfreeze_backbone()` | Unfreeze backbone weights |
| `architecture.freeze_head(name)` | Freeze specific head |
| `architecture.unfreeze_head(name)` | Unfreeze specific head |
| `architecture.get_trainable_status()` | Get trainable status of all components |

### FeatureCacheManager Methods

| Method | Description |
|--------|-------------|
| `cache_manager.extract_features(images)` | Extract and cache features |
| `cache_manager.save_features(path)` | Save features to disk |
| `cache_manager.load_features(path)` | Load features from disk |
| `cache_manager.clear_cache()` | Clear cached features |
| `cache_manager.load_head(name)` | Load a head for inference |
| `cache_manager.unload_head(name)` | Unload a head |
| `cache_manager.unload_all_heads()` | Unload all heads |
| `cache_manager.predict_with_cached_features(name)` | Run prediction |
| `cache_manager.predict_all_loaded_heads()` | Run all loaded heads |
| `cache_manager.get_memory_usage()` | Get memory info |
| `cache_manager.compare_performance(...)` | Compare performance |

### Separate TFLite Export Methods

| Function | Description |
|----------|-------------|
| `export_separate_tflite_models(...)` | Export backbone and heads as separate TFLite files |
| `convert_to_int8_tflite(model, input_shape, ...)` | Convert Keras model to int8 TFLite |
| `convert_to_fp16_tflite(model, ...)` | Convert Keras model to fp16 TFLite |
| `convert_to_fp32_tflite(model, ...)` | Convert Keras model to fp32 TFLite |

## Best Practices

1. **Always enable for transfer learning**: Use `separable_weights=True` when you plan to freeze/unfreeze components
2. **Save backbone after initial training**: Create reusable backbone checkpoints
3. **Use feature caching for multi-head inference**: Avoids redundant backbone computation
4. **Freeze backbone when adding new heads**: Prevents forgetting existing knowledge
5. **Monitor trainable status**: Use `get_trainable_status()` before training
6. **Unload unused heads**: Free memory with `unload_head()` when done

## Troubleshooting

### Error: "separable_weights not enabled"

Make sure you create the model with `separable_weights=True`:

```python
config = MultiHeadModelConfig(..., separable_weights=True)
```

### Error: "Head not found"

Check available heads:

```python
print(architecture.head_names)
```

### Weights not transferring correctly

Ensure configurations match:
- Same `alpha` value
- Same `input_shape`
- Architecture type (e.g., MobileNetV3QAT)

### Memory issues with feature caching

- Clear cache when done: `cache_manager.clear_cache()`
- Unload unused heads: `cache_manager.unload_head(name)`
- Process images in smaller batches


