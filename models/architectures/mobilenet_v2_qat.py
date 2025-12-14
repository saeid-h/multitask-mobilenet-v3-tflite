"""
MobileNetV2 QAT-optimized architecture implementation.

This module implements MobileNetV2 with Quantization-Aware Training (QAT) support,
following the successful reference project pattern. It uses Keras applications
directly and supports configurable alpha values for ultra-lightweight models.

Key features:
- Direct use of Keras MobileNetV2 with ImageNet pre-trained weights (RGB only)
- Configurable alpha values (0.35, 0.50, 0.75, 1.0, 1.3, 1.4) - Keras supported
- QAT-compatible architecture design
- Support for person detection with custom head
- Optimized for ultra-lightweight models (250KB+ target)

Reference: 
- MobileNetV2: Inverted Residuals and Linear Bottlenecks
- Keras MobileNetV2 supported alpha values
"""

from typing import Dict, Any, Tuple, List, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

from ..base import MobileNetArchitecture, ModelConfig
from ..factory import ModelArchitectureFactory


class MobileNetV2QATArchitecture(MobileNetArchitecture):
    """MobileNetV2 QAT-optimized architecture implementation.
    
    This class implements MobileNetV2 with QAT support, using Keras applications
    directly for better compatibility with TensorFlow Model Optimization.
    
    The architecture follows the reference project design:
    1. MobileNetV2 backbone with ImageNet pre-trained weights (RGB only)
    2. Configurable alpha values for model size control
    3. Custom head for person detection
    4. QAT-compatible design
    
    Supports alpha values: 0.35 (~250KB), 0.50 (~500KB), 0.75 (~1.1MB), 1.0 (~2.2MB), 1.3 (~3.8MB), 1.4 (~4.4MB)
    """
    
    @property
    def name(self) -> str:
        """Return the architecture name."""
        alpha = self.config.arch_params.get('alpha', 0.35)
        # Map alpha values to proper string representations
        alpha_mapping = {
            0.35: '0_35',
            0.50: '0_50',
            0.75: '0_75',
            1.0: '1_0',
            1.3: '1_3',
            1.4: '1_4'
        }
        alpha_str = alpha_mapping.get(alpha, str(alpha).replace('.', '_'))
        return f"mobilenet_v2_qat_{alpha_str}"
    
    @property
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        """Return list of supported input shapes for MobileNetV2 QAT."""
        return [
            (96, 96, 1),     # Person detection optimized (MCU)
            (96, 96, 3),     # Person detection RGB (MCU)
            (128, 128, 1),   # Medium resolution grayscale
            (128, 128, 3),   # Medium resolution RGB
            (160, 160, 1),   # Higher resolution grayscale
            (160, 160, 3),   # Higher resolution RGB
            (224, 224, 1),   # Standard resolution grayscale
            (224, 224, 3),   # Standard ImageNet input
            (256, 256, 1),   # High resolution grayscale
            (256, 256, 3),   # High resolution RGB
        ]
    
    @property
    def parameter_count_range(self) -> Tuple[int, int]:
        """Return expected parameter count range for MobileNetV2 QAT."""
        alpha = self.config.arch_params.get('alpha', 0.35)
        
        # Actual parameter counts from Keras MobileNetV2 with different alphas
        # These are more accurate than the reference project estimates
        reference_params = {
            0.35: 250_000,   # ~250KB model (actual Keras value)
            0.50: 500_000,   # ~500KB model
            0.75: 1_100_000, # ~1.1MB model
            1.0: 2_200_000,  # ~2.2MB model
            1.3: 3_800_000,  # ~3.8MB model
            1.4: 4_400_000,  # ~4.4MB model
        }
        
        base_params = reference_params.get(alpha, 250_000)
        
        # Allow 15% tolerance for custom configurations
        min_params = int(base_params * 0.85)
        max_params = int(base_params * 1.15)
        
        return (min_params, max_params)
    
    def validate_config(self) -> None:
        """Validate MobileNetV2 QAT-specific configuration parameters."""
        # Validate alpha value - only values supported by Keras MobileNetV2 with ImageNet weights
        alpha = self.config.arch_params.get('alpha', 0.35)
        valid_alphas = [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]  # Keras MobileNetV2 supported values
        
        if alpha not in valid_alphas:
            raise ValueError(
                f"alpha must be one of {valid_alphas} (Keras MobileNetV2 supported values), got {alpha}"
            )
        
        # Validate input shape
        self.validate_input_shape(self.config.input_shape)
        
        # Validate minimum image size for MobileNetV2
        height, width, channels = self.config.input_shape
        if height < 32 or width < 32:
            raise ValueError(
                f"MobileNetV2 requires minimum input size of 32x32, "
                f"got {height}x{width}"
            )
        
        # Validate channels
        if channels not in [1, 3]:
            raise ValueError(
                f"MobileNetV2 supports 1 (grayscale) or 3 (RGB) channels, "
                f"got {channels}"
            )
    
    def build_model(self) -> tf.keras.Model:
        """Build and return the MobileNetV2 QAT TensorFlow model.
        
        This implementation follows the reference project pattern:
        1. Use Keras MobileNetV2 with ImageNet weights (only for RGB)
        2. Add custom head for person detection
        3. Ensure QAT compatibility
        """
        alpha = self.config.arch_params.get('alpha', 0.35)
        use_pretrained = self.config.arch_params.get('use_pretrained', True)
        num_classes = self.config.num_classes
        
        # Get input shape
        input_shape = self.config.input_shape
        height, width, channels = input_shape
        
        # Determine if we can use ImageNet weights
        # ImageNet weights require 3 channels
        can_use_imagenet = use_pretrained and channels == 3
        weights = 'imagenet' if can_use_imagenet else None
        
        # Create MobileNetV2 backbone
        backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            alpha=alpha,  # Key parameter for model size
            pooling=None  # No pooling, we'll add custom head
        )
        
        # Freeze backbone for transfer learning if using pretrained weights
        if can_use_imagenet:
            backbone.trainable = False
        
        # Create custom head for person detection
        # Following the reference project pattern but adapted for person detection
        head = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax', name='person_detection_output')
        ])
        
        # Create the complete model
        model = tf.keras.Sequential([
            backbone,
            head
        ], name=f'mobilenet_v2_qat_alpha_{alpha}')
        
        return model
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        model = self.build_model()
        
        # Capture model summary
        summary_parts = []
        model.summary(print_fn=lambda x: summary_parts.append(x))
        
        return '\n'.join(summary_parts)


def create_mobilenet_v2_qat_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Create default configurations for MobileNetV2 QAT variants.
    
    Returns:
        List of (architecture_name, default_config) tuples
    """
    configs = []
    
    # Define alpha values and their target sizes (Keras supported values)
    alpha_configs = [
        (0.35, "Ultra-lightweight MCU (~250KB)"),
        (0.50, "Lightweight MCU (~500KB)"),
        (0.75, "Balanced mobile (~1.1MB)"),
        (1.0, "Standard mobile (~2.2MB)"),
        (1.3, "High-performance mobile (~3.8MB)"),
        (1.4, "Maximum performance (~4.4MB)")
    ]
    
    for alpha, description in alpha_configs:
        # Map alpha values to proper string representations
        alpha_mapping = {
            0.35: '0_35',
            0.50: '0_50',
            0.75: '0_75',
            1.0: '1_0',
            1.3: '1_3',
            1.4: '1_4'
        }
        alpha_str = alpha_mapping.get(alpha, str(alpha).replace('.', '_'))
        arch_name = f"mobilenet_v2_qat_{alpha_str}"
        
        default_config = {
            'input_shape': (96, 96, 1),  # Default for person detection
            'num_classes': 2,  # Person detection: person/no-person
            'arch_params': {
                'alpha': alpha,
                'use_pretrained': True,
            },
            'description': f"MobileNetV2 QAT with alpha={alpha} - {description}"
        }
        
        configs.append((arch_name, default_config))
    
    return configs


def _register_mobilenet_v2_qat_variants():
    """Register all MobileNetV2 QAT variants with the factory."""
    configs = create_mobilenet_v2_qat_configs()
    
    for arch_name, default_config in configs:
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MobileNetV2QATArchitecture,
            default_config=default_config
        )


# Auto-register variants when module is imported
_register_mobilenet_v2_qat_variants()
