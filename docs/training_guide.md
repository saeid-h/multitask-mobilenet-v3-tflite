# Training Guide

How to use the model programmatically and train it with multiple datasets.

## Using as a Module

Import and create models programmatically instead of using the command-line script.

### Basic Import

```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import HeadConfiguration, create_head_config_from_list
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture
```

### Creating a Model Programmatically

```python
# Create head configurations
head_configs = create_head_config_from_list(
    [5, 2, 3],  # 5 classes, 2 classes, 3 classes
    head_names=["object_class", "person_detection", "age_group"]  # Optional names
)

# Or create manually for more control
# Note: For Vela compatibility, use 'linear' activation (default)
head_configs = [
    HeadConfiguration(name="object_class", num_classes=5, activation="linear"),
    HeadConfiguration(name="person_detection", num_classes=2, activation="linear"),
    HeadConfiguration(name="age_group", num_classes=3, activation="linear"),
]

# Create model configuration
config = MultiHeadModelConfig(
    input_shape=(224, 224, 3),
    head_configs=head_configs,
    arch_params={
        'alpha': 0.25,
        'use_pretrained': False,
    },
    training_mode='joint',  # or 'sequential', 'hybrid'
    inference_mode='all_active',
    loss_weights={'object_class': 1.0, 'person_detection': 1.0, 'age_group': 1.0}  # Optional
)

# Create architecture and build model
architecture = MultiHeadMobileNetV3QATArchitecture(config)
model = architecture.get_model()
```

## Training with Multiple Datasets

Multi-head models need training data for each head. There are two approaches: separate datasets or combined datasets.

### Approach 1: Separate Datasets (Recommended)

Each dataset provides labels for one head. The model processes the same images through the shared backbone, but each head learns from its own labels.

```python
import tensorflow as tf
import numpy as np

# Assume you have separate datasets
# dataset1: images + labels for head 1 (object_class)
# dataset2: images + labels for head 2 (person_detection)
# dataset3: images + labels for head 3 (age_group)

def create_multi_head_dataset(datasets_dict, batch_size=32):
    """
    Create a dataset that yields (images, {head_name: labels}) format.
    
    Args:
        datasets_dict: Dict mapping head names to (images, labels) tuples
        batch_size: Batch size for training
    """
    # Get the first dataset to extract image dataset
    first_head = list(datasets_dict.keys())[0]
    images_dataset = datasets_dict[first_head][0]
    
    # Create labels dictionary
    labels_dict = {}
    for head_name, (_, labels) in datasets_dict.items():
        labels_dict[head_name] = labels
    
    # Zip images with labels dictionary
    dataset = tf.data.Dataset.zip({
        'images': images_dataset,
        **labels_dict
    })
    
    # Rename 'images' key if needed, or restructure
    def map_fn(x):
        images = x.pop('images') if 'images' in x else list(x.values())[0]
        return images, x
    
    dataset = dataset.map(map_fn)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Example: Load your datasets
# These should be tf.data.Dataset objects
object_dataset = ...  # (images, object_labels)
person_dataset = ...  # (images, person_labels)
age_dataset = ...     # (images, age_labels)

# Combine into multi-head format
train_data = create_multi_head_dataset({
    'object_class': object_dataset,
    'person_detection': person_dataset,
    'age_group': age_dataset,
}, batch_size=32)
```

### Approach 2: Combined Dataset

If your dataset already has labels for all heads, structure it as:

```python
def load_combined_dataset():
    """
    Load dataset where each sample has labels for all heads.
    
    Returns:
        Dataset yielding (images, {
            'head_name_1': labels_1,
            'head_name_2': labels_2,
            ...
        })
    """
    # Your data loading code here
    images = ...  # np array or tf.data.Dataset
    labels_dict = {
        'object_class': object_labels,
        'person_detection': person_labels,
        'age_group': age_labels,
    }
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels_dict))
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
```

## Compiling for Training

Multi-head models need a loss function for each head and optional loss weights.

### Basic Compilation

```python
# Define loss functions for each head
# Note: Use from_logits=True since models default to linear activation (logits)
losses = {
    'object_class': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'person_detection': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'age_group': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
}

# Optional: Set loss weights (default is 1.0 for each)
loss_weights = {
    'object_class': 1.0,
    'person_detection': 2.0,  # Give more weight to person detection
    'age_group': 1.0,
}

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=losses,
    loss_weights=loss_weights,  # Optional
    metrics=['accuracy']  # Applied to each head
)
```

### Using Loss Weights from Configuration

If you set loss weights in the model config:

```python
# Loss weights are stored in the config
loss_weights = architecture.get_loss_weights()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={head.name: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
          for head in architecture.head_configs},
    loss_weights=loss_weights,
    metrics=['accuracy']
)
```

## Training the Model

Training works the same as single-head models, but the data format is different.

```python
# Prepare datasets
train_dataset = create_multi_head_dataset({
    'object_class': train_object_data,
    'person_detection': train_person_data,
    'age_group': train_age_data,
})

val_dataset = create_multi_head_dataset({
    'object_class': val_object_data,
    'person_detection': val_person_data,
    'age_group': val_age_data,
})

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks
)
```

## Monitoring Training

The history object contains metrics for each head:

```python
# Access individual head metrics
print(history.history['object_class_accuracy'])
print(history.history['person_detection_accuracy'])
print(history.history['age_group_accuracy'])

# Overall loss
print(history.history['loss'])  # Combined weighted loss
print(history.history['val_loss'])
```

Metric names follow the pattern: `{head_name}_{metric_name}`.

## Complete Example

```python
import tensorflow as tf
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import create_head_config_from_list
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture

# 1. Create model
head_configs = create_head_config_from_list(
    [5, 2, 3],
    ["object_class", "person_detection", "age_group"]
)

config = MultiHeadModelConfig(
    input_shape=(224, 224, 3),
    head_configs=head_configs,
    arch_params={'alpha': 0.25, 'use_pretrained': False},
    training_mode='joint',
    loss_weights={'object_class': 1.0, 'person_detection': 2.0, 'age_group': 1.0}
)

architecture = MultiHeadMobileNetV3QATArchitecture(config)
model = architecture.get_model()

# 2. Load datasets (implement your loading logic)
def load_datasets():
    # Return (train_images, train_labels_dict), (val_images, val_labels_dict)
    train_images = ...
    train_labels = {
        'object_class': ...,
        'person_detection': ...,
        'age_group': ...,
    }
    val_images = ...
    val_labels = {
        'object_class': ...,
        'person_detection': ...,
        'age_group': ...,
    }
    return (train_images, train_labels), (val_images, val_labels)

(train_images, train_labels), (val_images, val_labels) = load_datasets()

# 3. Create tf.data.Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# 4. Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={head.name: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
          for head in architecture.head_configs},
    loss_weights=architecture.get_loss_weights(),
    metrics=['accuracy']
)

# 5. Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    ]
)

# 6. Save
model.save('trained_model.keras')
```

## Handling Mismatched Datasets

If your datasets have different sizes or images don't align perfectly:

1. **Use the same images**: Ensure all datasets use the same image set
2. **Handle missing labels**: Use dummy labels or mask losses for missing data
3. **Data alignment**: Create a mapping between datasets to ensure images match

```python
# Example: Aligning datasets by image IDs
def align_datasets(image_ids, datasets_dict):
    """
    Align multiple datasets to use the same images.
    
    Args:
        image_ids: List of image identifiers
        datasets_dict: Dict mapping head names to (id_to_image, id_to_label) dicts
    """
    aligned_images = []
    aligned_labels = {head: [] for head in datasets_dict.keys()}
    
    for img_id in image_ids:
        # Assume first dataset provides images
        first_head = list(datasets_dict.keys())[0]
        aligned_images.append(datasets_dict[first_head][0][img_id])
        
        # Get labels from each dataset
        for head_name, (_, id_to_label) in datasets_dict.items():
            if img_id in id_to_label:
                aligned_labels[head_name].append(id_to_label[img_id])
            else:
                # Handle missing label (use dummy or skip)
                aligned_labels[head_name].append(0)  # or handle appropriately
    
    return np.array(aligned_images), aligned_labels
```

## Tips

- **Start with equal loss weights**: Begin training with all weights at 1.0, then adjust based on performance
- **Monitor each head separately**: Check individual head accuracies to ensure all tasks are learning
- **Freeze backbone initially**: For transfer learning, consider freezing the backbone for a few epochs
- **Use pretrained weights when possible**: If using alpha 0.75 or 1.0, `use_pretrained=True` can help
- **Balance datasets**: If one head has much more data than others, consider upsampling or adjusting loss weights
