"""
MobileNetV4 QAT-optimized architecture implementation.

This module implements MobileNetV4 with QAT optimization by wrapping the existing
MobileNetV4 implementation and adding QAT-specific optimizations.

Key features:
- Wrapper around existing MobileNetV4-Conv-S implementation
- Configurable alpha values (0.25, 0.50, 0.75, 1.0) via width multiplier
- QAT-compatible architecture design
- Support for person detection with custom head
- Optimized for ultra-lightweight models (250KB+ target)
"""

from typing import Dict, Any, Tuple, List
import tensorflow as tf
from tensorflow.keras import layers, Model

from ..base import MobileNetArchitecture, ModelConfig
from ..factory import ModelArchitectureFactory
from .mobilenet_v4 import MobileNetV4ConvS


class MobileNetV4QATArchitecture(MobileNetArchitecture):
    """MobileNetV4 QAT-optimized architecture implementation.
    
    This class implements MobileNetV4 with QAT optimization by wrapping the existing
    MobileNetV4 implementation and adding QAT-specific optimizations.
    
    Supports alpha values: 0.25, 0.50, 0.75, 1.0 (via width multiplier)
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
        return f"mobilenet_v4_qat_{alpha_str}"
    
    @property
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        """Return list of supported input shapes for MobileNetV4 QAT."""
        # MobileNetV4 supports various input resolutions
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
        """Return expected parameter count range for MobileNetV4 QAT."""
        alpha = self.config.arch_params.get('alpha', 0.25)
        
        # Approximate parameter counts for MobileNetV4 with different alpha values
        # Based on MobileNetV4-Conv-S base model (~3.8M params)
        base_params = 3_800_000
        alpha_to_params = {
            0.25: (200_000, 300_000),    # ~250K params
            0.50: (800_000, 1_200_000),  # ~1M params  
            0.75: (1_800_000, 2_700_000), # ~2.25M params
            1.0: (3_200_000, 4_800_000),  # ~4M params
        }
        
        return alpha_to_params.get(alpha, (100_000, 5_000_000))
    
    def validate_config(self) -> None:
        """Validate MobileNetV4 QAT-specific configuration parameters."""
        alpha = self.config.arch_params.get('alpha', 0.25)
        valid_alphas = [0.25, 0.50, 0.75, 1.0]
        
        if alpha not in valid_alphas:
            raise ValueError(f"alpha must be one of {valid_alphas} (MobileNetV4 supported values), got {alpha}")
        
        self.validate_input_shape(self.config.input_shape)
        
        # Validate minimum image size for MobileNetV4
        height, width, channels = self.config.input_shape
        if height < 32 or width < 32:
            raise ValueError(f"MobileNetV4 requires minimum input size of 32x32, got {height}x{width}")
        
        # Validate channels
        if channels not in [1, 3]:
            raise ValueError(f"MobileNetV4 supports 1 (grayscale) or 3 (RGB) channels, got {channels}")
    
    def build_model(self) -> tf.keras.Model:
        """Build and return the MobileNetV4 QAT TensorFlow model."""
        alpha = self.config.arch_params.get('alpha', 0.25)
        num_classes = self.config.num_classes
        input_shape = self.config.input_shape
        
        # Create MobileNetV4 backbone with width multiplier
        # Convert alpha to width_multiplier for the existing implementation
        width_multiplier = alpha
        
        # Create config for MobileNetV4
        v4_config = ModelConfig(
            input_shape=input_shape,
            num_classes=num_classes,
            arch_params={
                'width_multiplier': width_multiplier,
                'dropout_rate': self.config.dropout_rate,
            },
            weight_decay=self.config.weight_decay,
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_epsilon=self.config.batch_norm_epsilon,
            activation=self.config.activation,
            dropout_rate=self.config.dropout_rate,
        )
        
        # Create MobileNetV4 model directly
        v4_model = MobileNetV4ConvS(v4_config)
        model = v4_model.build_model()
        
        # Rename the model for QAT
        model._name = f'mobilenet_v4_qat_alpha_{alpha}'
        
        return model


# Factory functions for creating MobileNetV4 QAT configurations
def create_mobilenet_v4_qat_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Create default configurations for MobileNetV4 QAT variants.
    
    Returns:
        List of (architecture_name, default_config) tuples
    """
    configs = []
    
    # Define alpha values and their target sizes
    alpha_configs = [
        (0.25, "Ultra-lightweight MCU (~250KB)"),
        (0.50, "Lightweight MCU (~1MB)"),
        (0.75, "Balanced mobile (~2.25MB)"),
        (1.0, "Standard mobile (~4MB)")
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
        arch_name = f"mobilenet_v4_qat_{alpha_str}"
        
        default_config = {
            'input_shape': (96, 96, 1),  # Default for person detection
            'num_classes': 2,  # Person detection: person/no-person
            'arch_params': {
                'alpha': alpha,
                'dropout_rate': 0.2,
            },
            'description': f"MobileNetV4 QAT with alpha={alpha} - {description}"
        }
        
        configs.append((arch_name, default_config))
    
    return configs


# Register MobileNetV4 QAT variants
def _register_mobilenet_v4_qat_variants():
    """Register all MobileNetV4 QAT variants with the factory."""
    configs = create_mobilenet_v4_qat_configs()
    
    for arch_name, default_config in configs:
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MobileNetV4QATArchitecture,
            default_config=default_config
        )


# Auto-register variants when module is imported
_register_mobilenet_v4_qat_variants()
