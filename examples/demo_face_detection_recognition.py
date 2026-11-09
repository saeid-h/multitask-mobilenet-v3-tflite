#!/usr/bin/env python3
"""
Demo: Face Detection + Recognition System

This script demonstrates how to create a complete face detection and recognition
system using the new SSD detection head and embedding head.
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
    create_ssd_detection_head,
    create_embedding_head,
    create_classification_head
)
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture
from src.utils import get_losses_for_heads, get_metrics_for_heads, SSDLoss


def demo_face_detection_only():
    """Demo: Face detection using SSD head."""
    print("=" * 60)
    print("DEMO 1: Face Detection (SSD)")
    print("=" * 60)
    
    # Create SSD detection head for face detection
    face_detection = create_ssd_detection_head(
        name="face_detector",
        num_classes=1,  # Binary: face/no-face
        num_anchors=3,
        anchor_scales=[0.1, 0.2, 0.37],
        dropout_rate=0.1
    )
    
    config = MultiHeadModelConfig(
        input_shape=(224, 224, 3),
        head_configs=[face_detection],
        arch_params={'alpha': 0.75, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    print(f"Face detection model created:")
    print(f"  - Input: {config.input_shape}")
    print(f"  - Head: {face_detection.name} (SSD detection)")
    print(f"  - Outputs: boxes + confidence scores")
    
    # Demo prediction
    batch_size = 2
    dummy_images = np.random.random((batch_size,) + config.input_shape).astype(np.float32)
    
    predictions = model.predict(dummy_images, verbose=0)
    print(f"\nPrediction shapes:")
    print(f"  - Boxes: {predictions['face_detector']['boxes'].shape}")  # [batch, num_anchors, 4]
    print(f"  - Scores: {predictions['face_detector']['scores'].shape}")  # [batch, num_anchors, 1]
    
    print("✓ Face detection model created successfully!")
    return model


def demo_face_recognition_only():
    """Demo: Face recognition using embedding head."""
    print("\n" + "=" * 60)
    print("DEMO 2: Face Recognition (Embedding)")
    print("=" * 60)
    
    # Create embedding head for face recognition
    face_embedding = create_embedding_head(
        name="face_embedding",
        embed_dim=512,  # Standard for face recognition
        projection_layers=[1024, 512],
        use_bn=True,
        dropout_rate=0.1
    )
    
    config = MultiHeadModelConfig(
        input_shape=(112, 112, 3),  # Standard face crop size
        head_configs=[face_embedding],
        arch_params={'alpha': 0.75, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    print(f"Face recognition model created:")
    print(f"  - Input: {config.input_shape} (cropped faces)")
    print(f"  - Head: {face_embedding.name} (512D embedding)")
    print(f"  - Output: L2-normalized feature vectors")
    
    # Demo: Compare face embeddings
    batch_size = 4
    dummy_faces = np.random.random((batch_size,) + config.input_shape).astype(np.float32)
    
    embeddings = model.predict(dummy_faces, verbose=0)['face_embedding']
    print(f"\nEmbedding shape: {embeddings.shape}")
    
    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"L2 norms (should be ~1.0): {norms}")
    
    # Demo similarity computation
    similarity_matrix = np.dot(embeddings, embeddings.T)
    print(f"Cosine similarity matrix:\n{similarity_matrix}")
    
    print("✓ Face recognition model created successfully!")
    return model


def demo_combined_face_system():
    """Demo: Combined face detection + recognition + attributes."""
    print("\n" + "=" * 60)
    print("DEMO 3: Complete Face Analysis System")
    print("=" * 60)
    
    # Create multiple heads for comprehensive face analysis
    heads = [
        # Face detection
        create_ssd_detection_head(
            name="face_detector",
            num_classes=1,  # Binary face detection
            num_anchors=3,
            dropout_rate=0.1
        ),
        
        # Face recognition (for cropped faces)
        create_embedding_head(
            name="face_identity",
            embed_dim=256,
            projection_layers=[512],
            use_bn=True,
            dropout_rate=0.1
        ),
        
        # Face attributes
        create_classification_head(
            name="age_group", 
            num_classes=5,  # child, teen, adult, middle-aged, senior
            activation="linear",
            dropout_rate=0.2
        ),
        
        create_classification_head(
            name="gender",
            num_classes=2,  # male, female
            activation="linear",
            dropout_rate=0.2
        )
    ]
    
    config = MultiHeadModelConfig(
        input_shape=(224, 224, 3),
        head_configs=heads,
        arch_params={'alpha': 0.75, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    print(f"Complete face analysis system created:")
    print(f"  - Input: {config.input_shape}")
    print(f"  - {len(heads)} heads:")
    for head in heads:
        if head.head_type == 'ssd_detection':
            print(f"    * {head.name}: SSD detection (bboxes + confidence)")
        elif head.head_type == 'embedding':
            embed_dim = head.custom_params['embed_dim']
            print(f"    * {head.name}: {embed_dim}D embedding")
        else:
            print(f"    * {head.name}: {head.num_classes}-class classification")
    
    # Get appropriate losses (will use SSDLoss for detection automatically)
    try:
        losses = get_losses_for_heads(heads, from_logits=True)
        metrics = get_metrics_for_heads(heads)
        
        print(f"\nAutomatic loss selection:")
        for head_name, loss in losses.items():
            print(f"  - {head_name}: {type(loss).__name__}")
            
        # Note: For SSDLoss, you'd need custom training loop since it expects
        # specific data format (boxes + labels). This is just for demonstration.
        print("\nNote: SSD head requires custom training data format and loss handling.")
        
    except Exception as e:
        print(f"Loss selection (demonstration only): {e}")
    
    # Demo prediction on full images
    batch_size = 2
    dummy_images = np.random.random((batch_size,) + config.input_shape).astype(np.float32)
    
    predictions = model.predict(dummy_images, verbose=0)
    
    print(f"\nPrediction structure:")
    for head_name, pred in predictions.items():
        if isinstance(pred, dict):
            # SSD detection head returns dict
            print(f"  - {head_name}:")
            for key, value in pred.items():
                print(f"    * {key}: {value.shape}")
        else:
            print(f"  - {head_name}: {pred.shape}")
    
    print("✓ Complete face analysis system created successfully!")
    return model


def demo_dense_prediction_heads():
    """Demo: Additional dense prediction head types."""
    print("\n" + "=" * 60)
    print("DEMO 4: Dense Prediction Heads (Segmentation + Keypoints)")
    print("=" * 60)
    
    from models.components.head_configuration import (
        create_segmentation_head,
        create_keypoint_detection_head
    )
    
    # Create dense prediction heads
    heads = [
        # Semantic segmentation (face parsing)
        create_segmentation_head(
            name="face_parsing",
            num_classes=7,  # background, skin, hair, eyes, nose, mouth, other
            upsample_factor=8,
            intermediate_channels=[256, 128],
            dropout_rate=0.2
        ),
        
        # Facial landmark detection
        create_keypoint_detection_head(
            name="face_landmarks",
            num_keypoints=68,  # 68 facial landmarks
            upsample_factor=4,
            heatmap_sigma=1.0,
            dropout_rate=0.1
        )
    ]
    
    config = MultiHeadModelConfig(
        input_shape=(224, 224, 3),
        head_configs=heads,
        arch_params={'alpha': 0.5, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    print(f"Dense prediction model created:")
    print(f"  - Input: {config.input_shape}")
    for head in heads:
        if head.head_type == 'segmentation':
            print(f"    * {head.name}: Pixel-wise segmentation ({head.num_classes} classes)")
        elif head.head_type == 'keypoint_detection':
            print(f"    * {head.name}: Heatmap keypoint detection ({head.num_classes} points)")
    
    # Demo prediction
    batch_size = 2
    dummy_images = np.random.random((batch_size,) + config.input_shape).astype(np.float32)
    
    predictions = model.predict(dummy_images, verbose=0)
    
    print(f"\nDense prediction outputs:")
    for head_name, pred in predictions.items():
        print(f"  - {head_name}: {pred.shape}")
        if head_name == "face_parsing":
            print(f"    → Pixel-wise class predictions")
        elif head_name == "face_landmarks":
            print(f"    → Heatmaps for each keypoint")
    
    print("✓ Dense prediction heads created successfully!")
    return model


def main():
    """Run all face detection/recognition demos."""
    print("Face Detection + Recognition Demo")
    print("This demo shows the new head types for computer vision tasks.")
    
    models = []
    models.append(demo_face_detection_only())
    models.append(demo_face_recognition_only())
    models.append(demo_combined_face_system())
    models.append(demo_dense_prediction_heads())
    
    print("\n" + "=" * 60)
    print("ALL FACE DEMOS COMPLETED!")
    print("=" * 60)
    
    print("\nNew head types demonstrated:")
    print("1. ssd_detection: SSD-style object/face detection")
    print("2. yolo_detection: YOLO-style detection (implemented)")
    print("3. segmentation: Pixel-wise semantic segmentation")
    print("4. keypoint_detection: Heatmap-based keypoint detection")
    print("5. embedding: L2-normalized features (face recognition)")
    
    print("\nComplete computer vision pipeline:")
    print("• Face Detection → SSD head finds faces in images")
    print("• Face Recognition → Embedding head creates face features")
    print("• Face Parsing → Segmentation head segments face regions")
    print("• Landmark Detection → Keypoint head finds facial landmarks")
    print("• Attribute Classification → Standard heads predict age, gender, etc.")
    
    print("\nProduction considerations:")
    print("• SSD/YOLO heads need custom training loops and data formats")
    print("• Detection heads output multiple predictions per image")
    print("• Post-processing (NMS, thresholding) needed for detection")
    print("• Dense prediction heads require full-resolution ground truth")
    
    return models


if __name__ == "__main__":
    main()