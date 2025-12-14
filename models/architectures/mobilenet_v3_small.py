"""
MobileNetV3-Small Architecture Implementation

This module implements the MobileNetV3-Small architecture as described in:
"Searching for MobileNetV3" (Howard et al., 2019)

The architecture is adapted for person detection with:
- Input: 96×96×1 (grayscale instead of 224×224×3)
- Output: 2 classes (person/no-person)
- Width multipliers: 0.25, 0.5, 0.75, 1.0
"""

import tensorflow as tf
from typing import Optional, List, Tuple
from ..base import MobileNetArchitecture, ModelConfig
from ..components import (
    InvertedResidualBlock,
    StandardConvBlock, 
    GlobalAveragePoolingBlock,
    DenseClassificationHead,
    apply_activation,
    ComponentConfig
)


class MobileNetV3Small(MobileNetArchitecture):
    """MobileNetV3-Small architecture implementation.
    
    This implementation follows the exact specifications from the paper
    but adapts them for person detection on 96×96 grayscale images.
    
    Key features:
    - Inverted residual blocks with SE attention
    - Hard-swish and ReLU activations
    - Efficient network architecture search optimizations
    - Support for various width multipliers
    """
    
    # MobileNetV3-Small layer specification (width multiplier 1.0)
    # Format: (input_channels, expansion_ratio, output_channels, use_se, activation, stride, kernel_size)
    LAYER_SPECS = [
        # Initial conv: 96×96×1 -> 48×48×16
        # First inverted residual block
        (16, 1, 16, True, 'relu6', 2, 3),      # SE-InvRes: 48×48×16 -> 24×24×16
        # Second and third blocks  
        (16, 4.5, 24, False, 'relu6', 2, 3),  # InvRes: 24×24×16 -> 12×12×24
        (24, 3.67, 24, False, 'relu6', 1, 3), # InvRes: 12×12×24 -> 12×12×24
        # Fourth and fifth blocks
        (24, 4, 40, True, 'hard_swish', 2, 5),   # SE-InvRes: 12×12×24 -> 6×6×40
        (40, 6, 40, True, 'hard_swish', 1, 5),   # SE-InvRes: 6×6×40 -> 6×6×40
        (40, 6, 40, True, 'hard_swish', 1, 5),   # SE-InvRes: 6×6×40 -> 6×6×40
        # Sixth through eighth blocks
        (40, 3, 48, True, 'hard_swish', 1, 5),   # SE-InvRes: 6×6×40 -> 6×6×48
        (48, 3, 48, True, 'hard_swish', 1, 5),   # SE-InvRes: 6×6×48 -> 6×6×48
        # Final blocks
        (48, 6, 96, True, 'hard_swish', 2, 5),   # SE-InvRes: 6×6×48 -> 3×3×96
        (96, 6, 96, True, 'hard_swish', 1, 5),   # SE-InvRes: 3×3×96 -> 3×3×96
        (96, 6, 96, True, 'hard_swish', 1, 5),   # SE-InvRes: 3×3×96 -> 3×3×96
    ]
    
    def __init__(self, config: ModelConfig):
        """Initialize MobileNetV3-Small architecture.
        
        Args:
            config: Model configuration including width multiplier
        """
        super().__init__(config)
        self.width_multiplier = config.arch_params.get('width_multiplier', 1.0)
        self._name = f"mobilenet_v3_small_{str(self.width_multiplier).replace('.', '_')}"
        
        # Validate width multiplier
        if self.width_multiplier not in [0.25, 0.5, 0.75, 1.0]:
            raise ValueError(
                f"Width multiplier {self.width_multiplier} not supported. "
                f"Supported values: [0.25, 0.5, 0.75, 1.0]"
            )
    
    @property
    def name(self) -> str:
        """Return the architecture name."""
        return self._name
    
    @property
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        """Return list of supported input shapes for this architecture."""
        return [
            (96, 96, 1),    # Primary target for person detection
            (128, 128, 1),  # Medium resolution grayscale
            (224, 224, 1),  # Larger input for better accuracy
            (256, 256, 1),  # High resolution grayscale
            (96, 96, 3),    # RGB variant
            (128, 128, 3),  # Medium resolution RGB
            (224, 224, 3),  # Original paper input size
            (256, 256, 3),  # High resolution RGB
        ]
    
    @property
    def parameter_count_range(self) -> Tuple[int, int]:
        """Return expected parameter count range (min, max) for this architecture."""
        # Parameter ranges based on width multiplier
        # These are estimates based on the original MobileNetV3-Small paper
        base_min = 50_000    # Width 0.25
        base_max = 3_000_000 # Width 1.0
        
        # Adjust for our smaller input size and 2 classes vs 1000
        scale_factor = 0.8  # Slightly fewer parameters due to fewer classes
        
        return (int(base_min * scale_factor), int(base_max * scale_factor))
    
    def validate_config(self) -> None:
        """Validate architecture-specific configuration parameters."""
        # Validate input shape
        if self.config.input_shape not in self.supported_input_shapes:
            raise ValueError(
                f"Input shape {self.config.input_shape} not supported. "
                f"Supported shapes: {self.supported_input_shapes}"
            )
        
        # Validate width multiplier
        width_mult = self.config.arch_params.get('width_multiplier', 1.0)
        if width_mult not in [0.25, 0.5, 0.75, 1.0]:
            raise ValueError(
                f"Width multiplier {width_mult} not supported. "
                f"Supported values: [0.25, 0.5, 0.75, 1.0]"
            )
        
        # Validate architecture type if specified
        arch_type = self.config.arch_params.get('architecture_type', 'mobilenet_v3_small')
        if arch_type != 'mobilenet_v3_small':
            raise ValueError(
                f"Architecture type '{arch_type}' not supported by MobileNetV3Small. "
                f"Expected: 'mobilenet_v3_small'"
            )
        
        # Validate optimization parameters
        if 'hard_activations' in self.config.optimization_params:
            if not isinstance(self.config.optimization_params['hard_activations'], bool):
                raise ValueError("hard_activations must be a boolean")
        
        if 'se_blocks' in self.config.optimization_params:
            if not isinstance(self.config.optimization_params['se_blocks'], bool):
                raise ValueError("se_blocks must be a boolean")
    
    def _make_divisible(self, channels: int, divisor: int = 8) -> int:
        """Make channel count divisible by divisor for efficient computation.
        
        Args:
            channels: Input channel count
            divisor: Divisor for channel count (default: 8)
            
        Returns:
            Adjusted channel count
        """
        new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%
        if new_channels < 0.9 * channels:
            new_channels += divisor
        return new_channels
    
    def _apply_width_multiplier(self, channels: int) -> int:
        """Apply width multiplier to channel count.
        
        Args:
            channels: Base channel count
            
        Returns:
            Scaled channel count
        """
        return self._make_divisible(int(channels * self.width_multiplier))
    
    def build_model(self, training: bool = False) -> tf.keras.Model:
        """Build the complete MobileNetV3-Small model.
        
        Args:
            training: Whether in training mode
            
        Returns:
            Complete TensorFlow model
        """
        inputs = tf.keras.layers.Input(
            shape=self.config.input_shape,
            name='input_image'
        )
        
        # Initial convolution: 96×96×1 -> 48×48×16
        initial_channels = self._apply_width_multiplier(16)
        
        # Create config with hard_swish activation for initial conv
        initial_conv_config = ComponentConfig(
            activation='hard_swish',
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_epsilon=self.config.batch_norm_epsilon,
            use_batch_norm=True,
            use_bias=False
        )
        
        x = StandardConvBlock(
            config=initial_conv_config,
            filters=initial_channels,
            kernel_size=3,
            strides=2
        ).call(inputs, training)
        
        # Build inverted residual blocks
        for i, (input_ch, exp_ratio, output_ch, use_se, activation, stride, kernel_size) in enumerate(self.LAYER_SPECS):
            # Apply width multiplier
            output_channels = self._apply_width_multiplier(output_ch)
            
            # For very small width multipliers, adjust expansion ratios
            if self.width_multiplier <= 0.25:
                # Reduce expansion ratios for tiny models
                exp_ratio = max(1, exp_ratio * 0.5)
            elif self.width_multiplier <= 0.5:
                # Slightly reduce expansion ratios for small models
                exp_ratio = max(1, exp_ratio * 0.75)
            
            # Create inverted residual block
            block = InvertedResidualBlock(
                config=self.config,
                output_channels=output_channels,
                expansion_ratio=int(exp_ratio),
                stride=stride,
                use_se=use_se,
                se_reduction_ratio=4,
                activation=activation,
                kernel_size=kernel_size
            )
            
            x = block.call(x, training)
        
        # Final convolution before pooling
        final_channels = self._apply_width_multiplier(576)  # As per paper
        x = tf.keras.layers.Conv2D(
            final_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name='final_conv'
        )(x)
        
        x = tf.keras.layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum,
            epsilon=self.config.batch_norm_epsilon,
            name='final_conv_bn'
        )(x, training=training)
        
        x = apply_activation(x, 'hard_swish', training)
        
        # Global Average Pooling
        x = GlobalAveragePoolingBlock(self.config).call(x, training)
        
        # Final classification head
        # Add an intermediate dense layer for better feature learning
        intermediate_features = self._apply_width_multiplier(1024)
        x = tf.keras.layers.Dense(
            intermediate_features,
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            name='features'
        )(x)
        
        x = apply_activation(x, 'hard_swish', training)
        
        # Dropout for regularization
        if self.config.dropout_rate > 0:
            x = tf.keras.layers.Dropout(
                rate=self.config.dropout_rate,
                name='dropout'
            )(x, training=training)
        
        # Final classification layer
        classification_config = ComponentConfig(
            use_bias=True,  # Classification heads typically use bias
            use_batch_norm=False,  # No batch norm in final layer
            activation='relu'  # Use a valid activation (won't be used anyway)
        )
        
        outputs = DenseClassificationHead(
            num_classes=self.config.num_classes,
            dropout_rate=0.0,  # Already applied dropout above
            config=classification_config
        ).call(x, training)
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=self.name
        )
        
        return model
    
    def get_model_summary(self) -> dict:
        """Get summary information about the model architecture.
        
        Returns:
            Dictionary with model information
        """
        # Estimate parameter count based on width multiplier
        base_params = 2_500_000  # Approximate base parameters for width=1.0
        estimated_params = int(base_params * (self.width_multiplier ** 2))
        
        return {
            'architecture': 'MobileNetV3-Small',
            'width_multiplier': self.width_multiplier,
            'input_shape': self.config.input_shape,
            'num_classes': self.config.num_classes,
            'estimated_parameters': estimated_params,
            'num_layers': len(self.LAYER_SPECS) + 3,  # +3 for initial conv, final conv, classifier
            'features': [
                'Inverted residual blocks',
                'Squeeze-and-Excitation attention',
                'Hard-swish activation',
                'Efficient network architecture search',
                f'Width multiplier: {self.width_multiplier}'
            ]
        }
    
    def get_config(self) -> dict:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'width_multiplier': self.width_multiplier,
            'layer_specs': self.LAYER_SPECS
        })
        return config


def create_mobilenet_v3_small_configs() -> List[Tuple[str, dict]]:
    """Create default configurations for all MobileNetV3-Small variants.
    
    Returns:
        List of (architecture_name, default_config) tuples
    """
    configs = []
    
    for width_mult in [0.25, 0.5, 0.75, 1.0]:
        arch_name = f"mobilenet_v3_small_{str(width_mult).replace('.', '_')}"
        
        default_config = {
            'arch_params': {
                'width_multiplier': width_mult,
                'architecture_type': 'mobilenet_v3_small'
            },
            'optimization_params': {
                'quantization_friendly': True,
                'mobile_optimized': True,
                'se_blocks': True,
                'hard_activations': True
            }
        }
        
        configs.append((arch_name, default_config))
    
    return configs 