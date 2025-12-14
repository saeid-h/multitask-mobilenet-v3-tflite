"""
MobileNetV1 architecture implementation using the factory pattern.

This module implements MobileNetV1 with different width multipliers,
following the abstract base class interface while maintaining backward
compatibility with the existing implementation.
"""

from typing import Dict, Any, Tuple, List
import tensorflow as tf
from tensorflow.keras import layers, Model

from ..base import MobileNetArchitecture, ModelConfig
from ..factory import ModelArchitectureFactory


class MobileNetV1Architecture(MobileNetArchitecture):
    """MobileNetV1 architecture implementation.
    
    This class implements MobileNetV1 with configurable width multiplier,
    supporting the same functionality as the original create_mobilenet_v1_025
    function while providing a modular, extensible interface.
    
    Supports width multipliers: 0.25, 0.5, 0.75, 1.0
    """
    
    @property
    def name(self) -> str:
        """Return the architecture name."""
        width_mult = self.config.arch_params.get('width_multiplier', 0.25)
        return f"mobilenet_v1_{str(width_mult).replace('.', '_')}"
    
    @property
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        """Return list of supported input shapes for MobileNetV1."""
        # MobileNetV1 supports various input resolutions
        return [
            (96, 96, 1),    # Current default
            (128, 128, 1),  # Alternative resolution
            (160, 160, 1),  # Higher resolution
            (224, 224, 1),  # Standard ImageNet resolution
            (256, 256, 1),  # High resolution grayscale
            (96, 96, 3),    # RGB version
            (128, 128, 3),
            (160, 160, 3),
            (224, 224, 3),
            (256, 256, 3),  # High resolution RGB
        ]
    
    @property
    def parameter_count_range(self) -> Tuple[int, int]:
        """Return expected parameter count range for MobileNetV1."""
        width_mult = self.config.arch_params.get('width_multiplier', 0.25)
        
        # Approximate parameter counts based on width multiplier
        base_params = 4_200_000  # Full MobileNetV1 has ~4.2M parameters
        
        # Parameters scale roughly with width_multiplier^2
        min_params = int(base_params * (width_mult ** 2) * 0.8)  # 20% tolerance
        max_params = int(base_params * (width_mult ** 2) * 1.2)  # 20% tolerance
        
        return (min_params, max_params)
    
    def validate_config(self) -> None:
        """Validate MobileNetV1-specific configuration parameters."""
        # Validate width multiplier
        width_mult = self.config.arch_params.get('width_multiplier', 0.25)
        valid_width_mults = [0.25, 0.5, 0.75, 1.0]
        
        if width_mult not in valid_width_mults:
            raise ValueError(
                f"width_multiplier must be one of {valid_width_mults}, "
                f"got {width_mult}"
            )
        
        # Validate input shape
        self.validate_input_shape(self.config.input_shape)
        
        # Validate minimum image size for MobileNetV1
        height, width, channels = self.config.input_shape
        if height < 32 or width < 32:
            raise ValueError(
                f"MobileNetV1 requires minimum input size of 32x32, "
                f"got {height}x{width}"
            )
        
        # Validate channels
        if channels not in [1, 3]:
            raise ValueError(
                f"MobileNetV1 supports 1 (grayscale) or 3 (RGB) channels, "
                f"got {channels}"
            )
    
    def build_model(self) -> tf.keras.Model:
        """Build and return the MobileNetV1 TensorFlow model using modular components."""
        width_mult = self.config.arch_params.get('width_multiplier', 0.25)
        use_bias = self.config.arch_params.get('use_bias', False)
        
        # Get TensorFlow version for API compatibility
        tf_version = tf.__version__.split('.')
        is_tf219_or_higher = int(tf_version[0]) > 2 or (int(tf_version[0]) == 2 and int(tf_version[1]) >= 19)
        
        # Create regularizer based on weight decay
        regularizer = None
        if self.config.weight_decay > 0:
            regularizer = tf.keras.regularizers.l2(self.config.weight_decay)
        
        # Create component configuration
        from .components import ComponentConfig, StandardConvBlock, DepthwiseSeparableConvBlock, MobileNetClassificationHead
        
        comp_config = ComponentConfig(
            use_bias=use_bias,
            kernel_regularizer=regularizer if not is_tf219_or_higher else None,
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_epsilon=self.config.batch_norm_epsilon,
            activation=self.config.activation
        )
        
        # Input layer
        height, width, channels = self.config.input_shape
        inputs = layers.Input(shape=(height, width, channels))
        
        # Initial convolution using StandardConvBlock
        initial_conv = StandardConvBlock(
            filters=int(32 * width_mult),
            kernel_size=3,
            strides=2,
            config=comp_config
        )
        initial_conv.set_name_scope("initial_conv")
        x = initial_conv.call(inputs, training=None)
        
        # MobileNetV1 depthwise separable convolutions using DepthwiseSeparableConvBlock
        def create_dw_sep_block(filters: int, strides: int, block_id: int):
            """Create a depthwise separable convolution block."""
            block = DepthwiseSeparableConvBlock(
                filters=int(filters * width_mult),
                kernel_size=3,
                strides=strides,
                config=comp_config
            )
            block.set_name_scope(f"dw_sep_block_{block_id}")
            return block
        
        # MobileNetV1 architecture following the standard structure
        block_id = 1
        x = create_dw_sep_block(64, 1, block_id).call(x, training=None); block_id += 1
        x = create_dw_sep_block(128, 2, block_id).call(x, training=None); block_id += 1
        x = create_dw_sep_block(128, 1, block_id).call(x, training=None); block_id += 1
        x = create_dw_sep_block(256, 2, block_id).call(x, training=None); block_id += 1
        x = create_dw_sep_block(256, 1, block_id).call(x, training=None); block_id += 1
        x = create_dw_sep_block(512, 2, block_id).call(x, training=None); block_id += 1
        
        # 5x 512 filters with stride 1
        for i in range(5):
            x = create_dw_sep_block(512, 1, block_id).call(x, training=None)
            block_id += 1
        
        x = create_dw_sep_block(1024, 2, block_id).call(x, training=None); block_id += 1
        x = create_dw_sep_block(1024, 1, block_id).call(x, training=None)
        
        # Classification head using MobileNetClassificationHead
        classification_head = MobileNetClassificationHead(
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            config=comp_config
        )
        classification_head.set_name_scope("classification_head")
        outputs = classification_head.call(x, training=None)
        
        # Create and return the model
        model = Model(inputs=inputs, outputs=outputs, name=self.name)
        return model
    
    def _get_activation_layer(self):
        """Get activation layer based on configuration."""
        
        def hard_swish_activation(x):
            """Hard Swish activation function for TF 2.15 compatibility."""
            return x * tf.nn.relu6(x + 3.0) / 6.0
        
        activation_map = {
            'relu': layers.ReLU(),
            'relu6': layers.ReLU(6.0),
            'swish': layers.Activation('swish'),
            'hard_swish': layers.Activation(hard_swish_activation),
            'gelu': layers.Activation('gelu'),
        }
        
        activation = self.config.activation.lower()
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
        
        return activation_map[activation]


# Register MobileNetV1 architectures with different width multipliers
def _register_mobilenet_v1_variants():
    """Register MobileNetV1 variants with different width multipliers."""
    
    # Standard width multipliers
    width_multipliers = [0.25, 0.5, 0.75, 1.0]
    
    for width_mult in width_multipliers:
        arch_name = f"mobilenet_v1_{str(width_mult).replace('.', '_')}"
        
        default_config = {
            'arch_params': {
                'width_multiplier': width_mult,
                'use_bias': False,
            }
        }
        
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MobileNetV1Architecture,
            default_config=default_config
        )


# Don't auto-register at import time to avoid circular imports
# _register_mobilenet_v1_variants()


# Backward compatibility function
def create_mobilenet_v1_025() -> tf.keras.Model:
    """Create MobileNetV1 with 0.25 width multiplier for backward compatibility.
    
    This function maintains compatibility with existing code while using
    the new factory pattern internally.
    
    Returns:
        TensorFlow Keras model identical to the original implementation
    """
    # Ensure MobileNetV1 variants are registered
    if not ModelArchitectureFactory.is_architecture_available('mobilenet_v1_0_25'):
        _register_mobilenet_v1_variants()
    
    # Create architecture using factory
    arch = ModelArchitectureFactory.create_model('mobilenet_v1_0_25')
    
    # Return the built model
    return arch.get_model() 