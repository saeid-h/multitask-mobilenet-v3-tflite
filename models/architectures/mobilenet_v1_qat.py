"""
MobileNetV1 QAT-optimized architecture implementation.

This module implements MobileNetV1 with QAT optimization using Keras applications,
following the successful approach from the reference project.

Key features:
- Direct use of Keras MobileNetV1 with ImageNet pre-trained weights (RGB only)
- Configurable alpha values (0.25, 0.50, 0.75, 1.0) - Keras supported
- QAT-compatible architecture design
- Support for person detection with custom head
- Optimized for ultra-lightweight models (250KB+ target)
"""

from typing import Dict, Any, Tuple, List
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers, Model

from ..base import MobileNetArchitecture, ModelConfig
from ..factory import ModelArchitectureFactory


class MobileNetV1QATArchitecture(MobileNetArchitecture):
    """MobileNetV1 QAT-optimized architecture implementation.
    
    This class implements MobileNetV1 with QAT optimization using Keras applications,
    providing a direct path to ultra-lightweight models through proper quantization.
    
    Supports alpha values: 0.25, 0.50, 0.75, 1.0 (Keras MobileNetV1 supported)
    """
    
    @property
    def name(self) -> str:
        """Return the architecture name."""
        alpha = self.config.arch_params.get('alpha', 0.25)
        # Map alpha values to consistent naming
        alpha_mapping = {
            0.25: '0_25',
            0.50: '0_50', 
            0.75: '0_75',
            1.0: '1_0'
        }
        alpha_str = alpha_mapping.get(alpha, str(alpha).replace('.', '_'))
        return f"mobilenet_v1_qat_{alpha_str}"
    
    @property
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        """Return list of supported input shapes for MobileNetV1 QAT."""
        # MobileNetV1 supports various input resolutions
        return [
            (96, 96, 1),    # Person detection optimized size, grayscale
            (128, 128, 1),  # Alternative resolution, grayscale
            (160, 160, 1),  # Higher resolution, grayscale
            (224, 224, 1),  # Standard ImageNet resolution, grayscale
            (96, 96, 3),    # Person detection optimized size, RGB
            (128, 128, 3),  # Alternative resolution, RGB
            (160, 160, 3),  # Higher resolution, RGB
            (224, 224, 3),  # Standard ImageNet resolution, RGB
        ]
    
    @property
    def parameter_count_range(self) -> Tuple[int, int]:
        """Return expected parameter count range for MobileNetV1 QAT."""
        alpha = self.config.arch_params.get('alpha', 0.25)
        
        # Approximate parameter counts for MobileNetV1 with different alpha values
        # Based on Keras MobileNetV1 implementation
        alpha_to_params = {
            0.25: (200_000, 300_000),   # ~250K params
            0.50: (800_000, 1_200_000), # ~1M params  
            0.75: (1_800_000, 2_700_000), # ~2.25M params
            1.0: (3_200_000, 4_800_000),  # ~4M params
        }
        
        return alpha_to_params.get(alpha, (100_000, 5_000_000))
    
    def validate_config(self) -> None:
        """Validate MobileNetV1 QAT-specific configuration parameters."""
        alpha = self.config.arch_params.get('alpha', 0.25)
        valid_alphas = [0.25, 0.50, 0.75, 1.0]
        
        if alpha not in valid_alphas:
            raise ValueError(f"alpha must be one of {valid_alphas} (Keras MobileNetV1 supported values), got {alpha}")
        
        self.validate_input_shape(self.config.input_shape)
        
        # Validate minimum image size for MobileNetV1
        height, width, channels = self.config.input_shape
        if height < 32 or width < 32:
            raise ValueError(f"MobileNetV1 requires minimum input size of 32x32, got {height}x{width}")
        
        # Validate channels
        if channels not in [1, 3]:
            raise ValueError(f"MobileNetV1 supports 1 (grayscale) or 3 (RGB) channels, got {channels}")
    
    def build_model(self) -> tf.keras.Model:
        """Build and return the MobileNetV1 QAT TensorFlow model."""
        alpha = self.config.arch_params.get('alpha', 0.25)
        use_pretrained = self.config.arch_params.get('use_pretrained', True)
        num_classes = self.config.num_classes
        input_shape = self.config.input_shape
        height, width, channels = input_shape
        
        # Check if we can use ImageNet weights (only for RGB inputs)
        can_use_imagenet = use_pretrained and channels == 3
        weights = 'imagenet' if can_use_imagenet else None
        
        # Create MobileNetV1 backbone
        backbone = MobileNet(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            alpha=alpha,
            pooling=None
        )
        
        # Freeze backbone if using ImageNet weights
        if can_use_imagenet:
            backbone.trainable = False
        
        # Create person detection head (2 classes: person/no-person)
        head = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax', name='person_detection_output')
        ])
        
        # Create the complete model
        model = tf.keras.Sequential([backbone, head], name=f'mobilenet_v1_qat_alpha_{alpha}')
        
        return model


# Factory functions for creating MobileNetV1 QAT configurations
def create_mobilenet_v1_qat_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Create default configurations for MobileNetV1 QAT variants.
    
    Returns:
        List of (architecture_name, default_config) tuples
    """
    configs = []
    
    # Define alpha values and their target sizes (Keras supported values)
    alpha_configs = [
        (0.25, "Ultra-lightweight MCU (~200KB)"),
        (0.50, "Lightweight MCU (~800KB)"),
        (0.75, "Balanced mobile (~2.2MB)"),
        (1.0, "Standard mobile (~4.2MB)")
    ]
    
    for alpha, description in alpha_configs:
        # Map alpha values to proper string representations
        alpha_mapping = {
            0.25: '0_25',
            0.50: '0_50',
            0.75: '0_75',
            1.0: '1_0'
        }
        alpha_str = alpha_mapping.get(alpha, str(alpha).replace('.', '_'))
        arch_name = f"mobilenet_v1_qat_{alpha_str}"
        
        default_config = {
            'input_shape': (96, 96, 1),  # Default for person detection
            'num_classes': 2,  # Person detection: person/no-person
            'arch_params': {
                'alpha': alpha,
                'use_pretrained': True,
            },
            'description': f"MobileNetV1 QAT with alpha={alpha} - {description}"
        }
        
        configs.append((arch_name, default_config))
    
    return configs


# Register MobileNetV1 QAT variants
def _register_mobilenet_v1_qat_variants():
    """Register all MobileNetV1 QAT variants with the factory."""
    configs = create_mobilenet_v1_qat_configs()
    
    for arch_name, default_config in configs:
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MobileNetV1QATArchitecture,
            default_config=default_config
        )


# Auto-register when module is imported
_register_mobilenet_v1_qat_variants()
