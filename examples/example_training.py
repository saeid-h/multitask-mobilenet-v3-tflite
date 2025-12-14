#!/usr/bin/env python3
"""
Example: Training a multi-head MobileNet V3 model with multiple datasets.

This script demonstrates how to:
1. Create a multi-head model programmatically
2. Prepare data for training
3. Compile and train the model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import create_head_config_from_list
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture


def create_synthetic_dataset(num_samples=1000, input_shape=(224, 224, 3)):
    """Create synthetic datasets for demonstration."""
    
    images = np.random.random((num_samples,) + input_shape).astype(np.float32)
    
    # Create labels for each head
    object_labels = np.random.randint(0, 5, num_samples)  # 5 classes
    person_labels = np.random.randint(0, 2, num_samples)  # 2 classes
    age_labels = np.random.randint(0, 3, num_samples)     # 3 classes
    
    return images, {
        'object_class': object_labels,
        'person_detection': person_labels,
        'age_group': age_labels,
    }


def create_training_dataset(images, labels_dict, batch_size=32, shuffle=True):
    """Create a tf.data.Dataset for multi-head training."""
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels_dict))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def main():
    print("Creating multi-head MobileNet V3 model...")
    
    # Step 1: Create model configuration
    head_configs = create_head_config_from_list(
        [5, 2, 3],  # 5 classes, 2 classes, 3 classes
        head_names=["object_class", "person_detection", "age_group"]
    )
    
    config = MultiHeadModelConfig(
        input_shape=(224, 224, 3),
        head_configs=head_configs,
        arch_params={
            'alpha': 0.25,
            'use_pretrained': False,
        },
        training_mode='joint',
        loss_weights={
            'object_class': 1.0,
            'person_detection': 2.0,  # Give more weight to person detection
            'age_group': 1.0,
        }
    )
    
    # Step 2: Build model
    architecture = MultiHeadMobileNetV3QATArchitecture(config)
    model = architecture.get_model()
    
    print(f"Model created: {architecture.name}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Step 3: Prepare datasets
    print("\nPreparing datasets...")
    train_images, train_labels = create_synthetic_dataset(num_samples=1000)
    val_images, val_labels = create_synthetic_dataset(num_samples=200)
    
    train_dataset = create_training_dataset(train_images, train_labels, batch_size=32)
    val_dataset = create_training_dataset(val_images, val_labels, batch_size=32, shuffle=False)
    
    # Step 4: Compile model
    print("\nCompiling model...")
    
    losses = {
        head.name: tf.keras.losses.SparseCategoricalCrossentropy()
        for head in architecture.head_configs
    }
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=losses,
        loss_weights=architecture.get_loss_weights(),
        metrics=['accuracy']
    )
    
    # Step 5: Train model
    print("\nTraining model...")
    print("(Using synthetic data - this is just a demonstration)")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'example_trained_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 6: Show results
    print("\nTraining complete!")
    print("\nFinal metrics:")
    for head in architecture.head_configs:
        acc_key = f"{head.name}_accuracy"
        val_acc_key = f"val_{head.name}_accuracy"
        if acc_key in history.history:
            print(f"  {head.name}:")
            print(f"    Train accuracy: {history.history[acc_key][-1]:.4f}")
            if val_acc_key in history.history:
                print(f"    Val accuracy: {history.history[val_acc_key][-1]:.4f}")
    
    print(f"\nModel saved to: example_trained_model.keras")
    print("\nTo use with your own data, replace create_synthetic_dataset()")
    print("with your actual data loading logic.")


if __name__ == '__main__':
    main()
