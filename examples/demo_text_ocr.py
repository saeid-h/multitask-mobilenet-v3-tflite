#!/usr/bin/env python3
"""
Demo: Text Detection and OCR System

This script demonstrates the new text & OCR head types for detecting and
reading text in natural scene images.
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
    create_text_detection_head,
    create_text_recognition_head,
    create_scene_text_head,
    create_classification_head
)
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture
from src.utils import get_losses_for_heads, get_metrics_for_heads


def demo_text_detection():
    """Demo: Text detection for locating text regions."""
    print("=" * 60)
    print("DEMO 1: Text Detection")
    print("=" * 60)
    
    # Create text detection head
    text_detector = create_text_detection_head(
        name="text_detector",
        detect_orientation=True,
        min_text_size=8,
        text_threshold=0.7,
        dropout_rate=0.1
    )
    
    config = MultiHeadModelConfig(
        input_shape=(320, 320, 3),
        head_configs=[text_detector],
        arch_params={'alpha': 0.75, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    print(f"Text detection model created:")
    print(f"  - Input: {config.input_shape}")
    print(f"  - Head: {text_detector.name} (text detection)")
    print(f"  - Features: orientation detection, min size {text_detector.custom_params['min_text_size']}px")
    
    # Demo prediction
    batch_size = 2
    dummy_images = np.random.random((batch_size,) + config.input_shape).astype(np.float32)
    
    predictions = model.predict(dummy_images, verbose=0)
    print(f"\nText detection outputs:")
    for key, value in predictions['text_detector'].items():
        print(f"  - {key}: {value.shape}")
        if key == 'text_scores':
            print(f"    → Text confidence maps (0-1)")
        elif key == 'text_boxes':
            print(f"    → Bounding box predictions (x,y,w,h)")
        elif key == 'text_angles':
            print(f"    → Text orientation angles (-π to π)")
    
    print("✓ Text detection model created successfully!")
    return model


def demo_text_recognition():
    """Demo: Text recognition for reading cropped text regions."""
    print("\n" + "=" * 60)
    print("DEMO 2: Text Recognition (OCR)")
    print("=" * 60)
    
    # Character vocabulary: digits + letters + punctuation + blank
    vocab_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?-' "
    vocab_size = len(vocab_chars) + 1  # +1 for CTC blank token
    
    # Create text recognition head
    text_recognizer = create_text_recognition_head(
        name="text_recognizer",
        vocab_size=vocab_size,
        max_text_length=32,
        use_attention=True,
        rnn_units=256,
        num_rnn_layers=2,
        dropout_rate=0.2
    )
    
    config = MultiHeadModelConfig(
        input_shape=(32, 128, 3),  # Typical text crop: height=32, width=128
        head_configs=[text_recognizer],
        arch_params={'alpha': 0.5, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    print(f"Text recognition model created:")
    print(f"  - Input: {config.input_shape} (cropped text regions)")
    print(f"  - Head: {text_recognizer.name} (CTC-based OCR)")
    print(f"  - Vocabulary: {vocab_size} characters")
    print(f"  - Max length: {text_recognizer.custom_params['max_text_length']} chars")
    print(f"  - Features: attention, {text_recognizer.custom_params['num_rnn_layers']} BiLSTM layers")
    
    # Demo prediction
    batch_size = 4
    dummy_text_crops = np.random.random((batch_size,) + config.input_shape).astype(np.float32)
    
    predictions = model.predict(dummy_text_crops, verbose=0)
    ctc_logits = predictions['text_recognizer']
    print(f"\nText recognition output:")
    print(f"  - CTC logits: {ctc_logits.shape}")
    print(f"    → [batch, time_steps, vocab_size] for sequence decoding")
    
    # Demo CTC decoding (simplified)
    predicted_chars = np.argmax(ctc_logits, axis=-1)
    print(f"  - Predicted char indices: {predicted_chars.shape}")
    print(f"    → Example sequence: {predicted_chars[0][:10]}")
    
    print("✓ Text recognition model created successfully!")
    return model


def demo_scene_text_end_to_end():
    """Demo: End-to-end scene text reading."""
    print("\n" + "=" * 60)
    print("DEMO 3: End-to-End Scene Text Reading")
    print("=" * 60)
    
    # Character vocabulary
    vocab_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?-' "
    vocab_size = len(vocab_chars) + 1
    
    # Create scene text head
    scene_text = create_scene_text_head(
        name="scene_text_reader",
        char_vocab_size=vocab_size,
        max_detections=50,
        max_chars_per_text=20,
        detection_threshold=0.5,
        dropout_rate=0.1
    )
    
    config = MultiHeadModelConfig(
        input_shape=(384, 384, 3),
        head_configs=[scene_text],
        arch_params={'alpha': 0.75, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    print(f"End-to-end scene text model created:")
    print(f"  - Input: {config.input_shape}")
    print(f"  - Head: {scene_text.name} (detection + recognition)")
    print(f"  - Max detections: {scene_text.custom_params['max_detections']}")
    print(f"  - Max chars per text: {scene_text.custom_params['max_chars_per_text']}")
    print(f"  - Vocabulary: {vocab_size} characters")
    
    # Demo prediction
    batch_size = 2
    dummy_images = np.random.random((batch_size,) + config.input_shape).astype(np.float32)
    
    predictions = model.predict(dummy_images, verbose=0)
    scene_outputs = predictions['scene_text_reader']
    
    print(f"\nEnd-to-end scene text outputs:")
    for key, value in scene_outputs.items():
        print(f"  - {key}: {value.shape}")
        if key == 'text_instances':
            print(f"    → Text locations: [x1, y1, x2, y2, angle, confidence]")
        elif key == 'text_sequences':
            print(f"    → Character predictions for each detection")
    
    print("✓ End-to-end scene text model created successfully!")
    return model


def demo_document_analysis_system():
    """Demo: Complete document analysis with multiple text tasks."""
    print("\n" + "=" * 60)
    print("DEMO 4: Document Analysis System")
    print("=" * 60)
    
    # Vocabulary for text recognition
    vocab_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?-'() "
    vocab_size = len(vocab_chars) + 1
    
    # Create comprehensive document analysis heads
    heads = [
        # Text detection: Find all text regions
        create_text_detection_head(
            name="text_detector",
            detect_orientation=True,
            text_threshold=0.6,
            dropout_rate=0.1
        ),
        
        # Document classification: Document type/category
        create_classification_head(
            name="document_type",
            num_classes=10,  # receipt, invoice, form, letter, etc.
            activation="linear",
            dropout_rate=0.2
        ),
        
        # Layout analysis: Text region classification
        create_classification_head(
            name="text_region_type", 
            num_classes=8,  # header, paragraph, list, table, etc.
            activation="linear",
            dropout_rate=0.2
        ),
        
        # Document orientation
        create_classification_head(
            name="page_orientation",
            num_classes=4,  # 0°, 90°, 180°, 270°
            activation="linear",
            dropout_rate=0.1
        )
    ]
    
    config = MultiHeadModelConfig(
        input_shape=(512, 512, 3),  # High resolution for documents
        head_configs=heads,
        arch_params={'alpha': 1.0, 'use_pretrained': False}  # Larger model for complex analysis
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    print(f"Document analysis system created:")
    print(f"  - Input: {config.input_shape} (high-res documents)")
    print(f"  - {len(heads)} analysis heads:")
    
    for head in heads:
        if head.head_type == 'text_detection':
            print(f"    * {head.name}: Locate text regions with orientation")
        else:
            print(f"    * {head.name}: {head.num_classes}-class classification")
    
    # Get appropriate losses
    losses = get_losses_for_heads(heads, from_logits=True)
    metrics = get_metrics_for_heads(heads)
    
    print(f"\nAutomatic loss selection:")
    for head_name, loss in losses.items():
        print(f"  - {head_name}: {type(loss).__name__}")
    
    # Demo prediction
    batch_size = 2
    dummy_documents = np.random.random((batch_size,) + config.input_shape).astype(np.float32)
    
    predictions = model.predict(dummy_documents, verbose=0)
    
    print(f"\nDocument analysis outputs:")
    for head_name, pred in predictions.items():
        if isinstance(pred, dict):
            print(f"  - {head_name}:")
            for key, value in pred.items():
                print(f"    * {key}: {value.shape}")
        else:
            print(f"  - {head_name}: {pred.shape}")
    
    print("✓ Document analysis system created successfully!")
    return model


def demo_multilingual_ocr():
    """Demo: Multilingual OCR with language detection."""
    print("\n" + "=" * 60)
    print("DEMO 5: Multilingual OCR System") 
    print("=" * 60)
    
    # Extended vocabulary for multiple languages
    # English + numbers + common punctuation
    base_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    # Add some common non-English characters (simplified example)
    extended_chars = "àáâãäåæçèéêëìíîïñòóôõöøùúûüý"
    vocab_chars = base_chars + extended_chars + ".,!?-'()[]{}@#$%&*+/\\|~`"
    vocab_size = len(vocab_chars) + 1  # +1 for CTC blank
    
    heads = [
        # Language detection
        create_classification_head(
            name="language_detection",
            num_classes=5,  # English, Spanish, French, German, Italian
            activation="linear",
            dropout_rate=0.2
        ),
        
        # Script detection (writing system)
        create_classification_head(
            name="script_type", 
            num_classes=3,  # Latin, Arabic, Asian
            activation="linear",
            dropout_rate=0.1
        ),
        
        # Multilingual text recognition
        create_text_recognition_head(
            name="multilingual_ocr",
            vocab_size=vocab_size,
            max_text_length=40,
            use_attention=True,
            rnn_units=512,  # Larger for multilingual
            num_rnn_layers=3,
            dropout_rate=0.2
        )
    ]
    
    config = MultiHeadModelConfig(
        input_shape=(64, 256, 3),  # Larger text crops for multilingual
        head_configs=heads,
        arch_params={'alpha': 0.75, 'use_pretrained': False}
    )
    
    arch = MultiHeadMobileNetV3QATArchitecture(config)
    model = arch.get_model()
    
    print(f"Multilingual OCR system created:")
    print(f"  - Input: {config.input_shape} (text crop images)")
    print(f"  - Extended vocabulary: {vocab_size} characters")
    print(f"  - Languages: 5 (English, Spanish, French, German, Italian)")
    print(f"  - Scripts: 3 (Latin, Arabic, Asian)")
    print(f"  - Features: attention-based OCR, language auto-detection")
    
    # Demo prediction
    batch_size = 3
    dummy_text_images = np.random.random((batch_size,) + config.input_shape).astype(np.float32)
    
    predictions = model.predict(dummy_text_images, verbose=0)
    
    print(f"\nMultilingual OCR outputs:")
    for head_name, pred in predictions.items():
        print(f"  - {head_name}: {pred.shape}")
        if head_name == "language_detection":
            # Demo language prediction
            lang_probs = tf.nn.softmax(pred, axis=-1).numpy()
            languages = ["English", "Spanish", "French", "German", "Italian"]
            print(f"    → Example language scores: {dict(zip(languages, lang_probs[0]))}")
        elif head_name == "multilingual_ocr":
            print(f"    → Character sequence logits for CTC decoding")
    
    print("✓ Multilingual OCR system created successfully!")
    return model


def main():
    """Run all text & OCR demos."""
    print("Text Detection & OCR Demo")
    print("This demo shows the new text & OCR head types for document analysis.")
    
    models = []
    models.append(demo_text_detection())
    models.append(demo_text_recognition())
    models.append(demo_scene_text_end_to_end())
    models.append(demo_document_analysis_system())
    models.append(demo_multilingual_ocr())
    
    print("\n" + "=" * 60)
    print("ALL TEXT & OCR DEMOS COMPLETED!")
    print("=" * 60)
    
    print("\nNew text head types demonstrated:")
    print("1. text_detection: Locate text regions with orientation")
    print("2. text_recognition: CTC-based OCR for cropped text")
    print("3. scene_text: End-to-end detection + recognition")
    
    print("\nComplete text processing pipeline:")
    print("• Text Detection → Find text regions in images")
    print("• Text Recognition → Read cropped text regions (OCR)")
    print("• Scene Text Reading → Direct text extraction from scenes")
    print("• Document Analysis → Type classification + layout analysis")
    print("• Multilingual OCR → Language detection + multilingual reading")
    
    print("\nUse cases covered:")
    print("• Document digitization (receipts, invoices, forms)")
    print("• Scene text reading (signs, menus, product labels)")
    print("• Multilingual text processing")
    print("• Document layout analysis")
    print("• Real-time text extraction from camera feeds")
    
    print("\nTechnical features:")
    print("• CTC loss for variable-length sequences")
    print("• Attention mechanisms for better accuracy")
    print("• Orientation detection for rotated text")
    print("• Combined detection + recognition losses")
    print("• Extensible vocabulary support")
    
    return models


if __name__ == "__main__":
    main()