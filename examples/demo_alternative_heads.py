#!/usr/bin/env python3
"""
Demo: Training with Alternative Head Types

This script demonstrates how to create, train, and use models with different
head types for various tasks: classification, multi-label, regression, 
embedding, and ordinal regression.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import (
    create_classification_head,
    create_multilabel_head, 
    create_regression_head,
    create_embedding_head,
    create_ordinal_head
)
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture
from src.utils import (
    get_losses_for_heads,
    get_metrics_for_heads,
    OrdinalCrossEntropy,
    ordinal_accuracy,
    ordinal_mae
)


def create_demo_data(input_shape, batch_size=32):
    """Create synthetic demo data for different tasks."""
    images = np.random.random((batch_size,) + input_shape).astype(np.float32)
    
    # Different label formats for different tasks
    labels = {
        'classification': np.random.randint(0, 5, (batch_size,)),  # Single class labels
        'multilabel': np.random.randint(0, 2, (batch_size, 3)).astype(np.float32),  # Multi-label binary
        'regression': np.random.random((batch_size, 2)).astype(np.float32),  # Continuous values
        'embedding': np.random.randint(0, 10, (batch_size,)),  # Identity labels for embedding
        'ordinal': np.random.randint(0, 4, (batch_size,))  # Ordinal class labels
    }
    
    return images, labels


def demo_classification_model():
    """Demo: Standard multi-class classification."""
    print("=" * 60)
    print("DEMO 1: Multi-Class Classification")
    print("=" * 60)
    
    # Create classification heads
    head_configs = [
        create_classification_head('scene_type', num_classes=5, activation='linear'),
        create_classification_head('weather', num_classes=3, activation='linear')
    ]
    
    config = MultiHeadModelConfig(
        input_shape=(96, 96, 3),
        head_configs=head_configs,
        arch_params={'alpha': 0.25, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    # Get appropriate losses and metrics
    losses = get_losses_for_heads(head_configs, from_logits=True)
    metrics = get_metrics_for_heads(head_configs)
    
    print(f"Model: {len(head_configs)} classification heads")
    print(f"Losses: {list(losses.keys())}")
    print(f"Metrics: {metrics}")
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss=losses,
        metrics=metrics
    )
    
    # Demo training data
    images, labels = create_demo_data((96, 96, 3))
    train_labels = {
        'scene_type': labels['classification'],
        'weather': np.random.randint(0, 3, (32,))
    }
    
    print("Training for 1 epoch...")
    model.fit(images, train_labels, epochs=1, verbose=0)
    print("✓ Classification model trained successfully!")
    return model


def demo_multilabel_model():
    """Demo: Multi-label classification."""
    print("\n" + "=" * 60)
    print("DEMO 2: Multi-Label Classification")
    print("=" * 60)
    
    # Create multi-label head
    head_configs = [
        create_multilabel_head('image_attributes', num_labels=5, dropout_rate=0.3)
    ]
    
    config = MultiHeadModelConfig(
        input_shape=(128, 128, 3),
        head_configs=head_configs,
        arch_params={'alpha': 0.25, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    # Get losses and metrics
    losses = get_losses_for_heads(head_configs, from_logits=False)  # Sigmoid outputs
    metrics = get_metrics_for_heads(head_configs)
    
    print(f"Model: Multi-label head with {head_configs[0].num_classes} labels")
    print(f"Losses: {list(losses.keys())}")
    print(f"Metrics: {metrics}")
    
    model.compile(
        optimizer='adam',
        loss=losses,
        metrics=metrics
    )
    
    # Demo training data
    images, labels = create_demo_data((128, 128, 3))
    # Create correct multi-label data (5 labels to match head config)
    train_labels = {'image_attributes': np.random.randint(0, 2, (32, 5)).astype(np.float32)}
    
    print("Training for 1 epoch...")
    model.fit(images, train_labels, epochs=1, verbose=0)
    print("✓ Multi-label model trained successfully!")
    return model


def demo_regression_model():
    """Demo: Regression tasks."""
    print("\n" + "=" * 60)
    print("DEMO 3: Regression")
    print("=" * 60)
    
    # Create regression heads
    head_configs = [
        create_regression_head('age_estimation', n_outputs=1, output_scale=100.0),
        create_regression_head('pose_angles', n_outputs=3),  # yaw, pitch, roll
        create_regression_head('quality_score', n_outputs=1, output_activation='sigmoid')
    ]
    
    config = MultiHeadModelConfig(
        input_shape=(128, 128, 3),
        head_configs=head_configs,
        arch_params={'alpha': 0.25, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    # Get losses and metrics (specify MSE for regression)
    loss_overrides = {head.name: {'loss_type': 'mse'} for head in head_configs}
    losses = get_losses_for_heads(head_configs, loss_overrides=loss_overrides)
    metrics = get_metrics_for_heads(head_configs)
    
    print(f"Model: {len(head_configs)} regression heads")
    for head in head_configs:
        n_out = head.custom_params.get('n_outputs', 1)
        activation = head.custom_params.get('output_activation', 'linear')
        print(f"  - {head.name}: {n_out} outputs, {activation} activation")
    
    model.compile(
        optimizer='adam',
        loss=losses,
        metrics=metrics
    )
    
    # Demo training data
    images, labels = create_demo_data((128, 128, 3))
    train_labels = {
        'age_estimation': np.random.uniform(0, 1, (32, 1)),  # Normalized age
        'pose_angles': np.random.uniform(-1, 1, (32, 3)),   # Angles in radians
        'quality_score': np.random.uniform(0, 1, (32, 1))   # Quality 0-1
    }
    
    print("Training for 1 epoch...")
    model.fit(images, train_labels, epochs=1, verbose=0)
    print("✓ Regression model trained successfully!")
    return model


def demo_embedding_model():
    """Demo: Embedding for similarity matching."""
    print("\n" + "=" * 60)
    print("DEMO 4: Embedding Learning")
    print("=" * 60)
    
    # Create embedding head
    head_configs = [
        create_embedding_head('face_embedding', embed_dim=128, projection_layers=[256], use_bn=True)
    ]
    
    config = MultiHeadModelConfig(
        input_shape=(160, 160, 3),
        head_configs=head_configs,
        arch_params={'alpha': 0.5, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    # For embedding, we typically use cosine similarity or triplet loss
    # Here we'll use MSE as a simple example
    loss_overrides = {'face_embedding': {'loss_type': 'cosine'}}
    losses = get_losses_for_heads(head_configs, loss_overrides=loss_overrides)
    metrics = get_metrics_for_heads(head_configs)
    
    embed_dim = head_configs[0].custom_params['embed_dim']
    print(f"Model: {embed_dim}D embedding head")
    print(f"Losses: {list(losses.keys())}")
    print(f"Metrics: {metrics}")
    
    model.compile(
        optimizer='adam',
        loss=losses,
        metrics=metrics
    )
    
    # Demo: predict embeddings (no training data format for demo)
    images, _ = create_demo_data((160, 160, 3))
    embeddings = model.predict(images, verbose=0)
    
    print(f"Predicted embeddings shape: {embeddings['face_embedding'].shape}")
    
    # Check L2 normalization
    norms = np.linalg.norm(embeddings['face_embedding'], axis=1)
    print(f"Embedding norms (should be ~1.0): {norms[:5]}")
    print("✓ Embedding model created successfully!")
    return model


def demo_ordinal_model():
    """Demo: Ordinal regression."""
    print("\n" + "=" * 60)
    print("DEMO 5: Ordinal Regression")
    print("=" * 60)
    
    # Create ordinal heads
    head_configs = [
        create_ordinal_head('age_group', num_classes=5, threshold_init='ascending'),
        create_ordinal_head('severity', num_classes=4, threshold_init='ascending')
    ]
    
    config = MultiHeadModelConfig(
        input_shape=(224, 224, 3),
        head_configs=head_configs,
        arch_params={'alpha': 0.25, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    # Ordinal regression uses special loss and metrics
    losses = {
        'age_group': OrdinalCrossEntropy(num_classes=5),
        'severity': OrdinalCrossEntropy(num_classes=4)
    }
    
    metrics = {
        'age_group': [ordinal_accuracy, ordinal_mae],
        'severity': [ordinal_accuracy, ordinal_mae]
    }
    
    print(f"Model: {len(head_configs)} ordinal regression heads")
    for head in head_configs:
        n_thresholds = head.num_classes - 1
        print(f"  - {head.name}: {head.num_classes} ordinal classes, {n_thresholds} thresholds")
    
    model.compile(
        optimizer='adam',
        loss=losses,
        metrics=metrics
    )
    
    # Demo training data
    images, labels = create_demo_data((224, 224, 3))
    train_labels = {
        'age_group': labels['ordinal'],
        'severity': np.random.randint(0, 4, (32,))
    }
    
    print("Training for 1 epoch...")
    model.fit(images, train_labels, epochs=1, verbose=0)
    print("✓ Ordinal regression model trained successfully!")
    return model


def main():
    """Run all demos."""
    print("Alternative Head Types Demo")
    print("This demo shows how to create and train models with different head types.")
    
    # Run demos
    models = []
    models.append(demo_classification_model())
    models.append(demo_multilabel_model())
    models.append(demo_regression_model())
    models.append(demo_embedding_model())
    models.append(demo_ordinal_model())
    
    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nSummary of head types:")
    print("1. standard: Multi-class classification (softmax/linear)")
    print("2. multilabel: Multi-label classification (sigmoid)")
    print("3. regression: Continuous value prediction (linear + scaling)")
    print("4. embedding: L2-normalized feature vectors (cosine similarity)")
    print("5. ordinal: Ordered categorical regression (CORAL thresholds)")
    
    print("\nKey utilities:")
    print("- get_losses_for_heads(): Automatic loss function selection")
    print("- get_metrics_for_heads(): Appropriate metrics per head type")
    print("- create_*_head(): Convenient head configuration builders")
    print("- Head builders are extensible via @register_head decorator")
    
    return models


if __name__ == "__main__":
    main()