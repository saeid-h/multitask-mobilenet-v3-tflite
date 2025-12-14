"""
Squeeze-and-Excitation (SE) Blocks for MobileNetV3

This module implements SE blocks as used in MobileNetV3 architectures.
SE blocks provide channel-wise attention to improve feature representation
while maintaining computational efficiency.
"""

import tensorflow as tf
from typing import Optional, Tuple, Dict
from .base import MobileNetComponent, ComponentConfig
from .activations import apply_activation

# Global counter for unique SE block naming
_se_block_counter = 0

def _get_unique_se_id() -> int:
    """Get unique identifier for SE block layer naming."""
    global _se_block_counter
    _se_block_counter += 1
    return _se_block_counter

class SqueezeExciteBlock(MobileNetComponent):
    """Squeeze-and-Excitation block for channel attention.
    
    The SE block performs the following operations:
    1. Global Average Pooling (Squeeze)
    2. Fully Connected layer with reduction
    3. ReLU activation
    4. Fully Connected layer with expansion
    5. Hard-Sigmoid activation (for MobileNetV3)
    6. Element-wise multiplication with input (Excitation)
    
    This implementation is optimized for mobile deployment with
    hard-sigmoid activation instead of sigmoid.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None, 
                 reduction_ratio: int = 4, 
                 activation: str = 'hard_sigmoid'):
        """Initialize SE block.
        
        Args:
            config: Component configuration
            reduction_ratio: Reduction ratio for bottleneck (default: 4)
            activation: Activation function for final gate (default: 'hard_sigmoid')
        """
        super().__init__(config or ComponentConfig())
        self.reduction_ratio = reduction_ratio
        self.activation = activation
        self._name = f"se_block_r{reduction_ratio}"
    
    @property
    def name(self) -> str:
        """Return component name."""
        return self._name
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply SE block (MobileNetComponent interface)."""
        return self.build_component(inputs, training or False)
        
    def build_component(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Build SE block component.
        
        Args:
            inputs: Input tensor [batch, height, width, channels]
            training: Whether in training mode
            
        Returns:
            Output tensor with SE attention applied
        """
        input_channels = inputs.shape[-1]
        reduced_channels = max(1, input_channels // self.reduction_ratio)
        
        # Get unique identifier for this SE block instance
        unique_id = _get_unique_se_id()
        
        # Squeeze: Global Average Pooling
        squeeze_layer = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        squeezed = squeeze_layer(inputs)
        
        # Excitation: Two FC layers with bottleneck
        # First FC layer (reduction)
        excitation = tf.keras.layers.Dense(
            reduced_channels,
            activation='relu',
            use_bias=True,
            kernel_initializer='he_normal',
            name=f'{self._name}_reduce_{unique_id}'
        )(squeezed)
        
        # Second FC layer (expansion)
        excitation = tf.keras.layers.Dense(
            input_channels,
            activation=None,
            use_bias=True,
            kernel_initializer='he_normal',
            name=f'{self._name}_expand_{unique_id}'
        )(excitation)
        
        # Apply activation (hard_sigmoid for MobileNetV3)
        excitation = apply_activation(excitation, self.activation, training)
        
        # Apply attention weights
        return inputs * excitation
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape (same as input for SE blocks)."""
        return input_shape
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for this component."""
        # This is an approximation - actual count depends on input channels
        # SE block parameters: reduction FC + expansion FC
        base_params = 100  # Rough estimate
        return {
            'total': base_params,
            'trainable': base_params,
            'non_trainable': 0
        }
    
    def get_config(self) -> dict:
        """Get configuration for serialization."""
        config = super().get_config_dict()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'activation': self.activation
        })
        return config


class EfficientSEBlock(MobileNetComponent):
    """Efficient SE block variant for very small models.
    
    This variant uses a single FC layer instead of two to reduce
    parameter count for very small networks (α < 0.5).
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None,
                 activation: str = 'hard_sigmoid'):
        """Initialize efficient SE block.
        
        Args:
            config: Component configuration
            activation: Activation function for gate (default: 'hard_sigmoid')
        """
        super().__init__(config or ComponentConfig())
        self.activation = activation
        self._name = "efficient_se_block"
    
    @property
    def name(self) -> str:
        """Return component name."""
        return self._name
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply efficient SE block (MobileNetComponent interface)."""
        return self.build_component(inputs, training or False)
        
    def build_component(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Build efficient SE block component.
        
        Args:
            inputs: Input tensor [batch, height, width, channels]
            training: Whether in training mode
            
        Returns:
            Output tensor with SE attention applied
        """
        input_channels = inputs.shape[-1]
        
        # Get unique identifier for this SE block instance
        unique_id = _get_unique_se_id()
        
        # Squeeze: Global Average Pooling
        squeeze_layer = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        squeezed = squeeze_layer(inputs)
        
        # Single FC layer (no bottleneck)
        excitation = tf.keras.layers.Dense(
            input_channels,
            activation=None,
            use_bias=True,
            kernel_initializer='he_normal',
            name=f'{self._name}_gate_{unique_id}'
        )(squeezed)
        
        # Apply activation
        excitation = apply_activation(excitation, self.activation, training)
        
        # Apply attention weights
        return inputs * excitation
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape (same as input for SE blocks)."""
        return input_shape
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for this component."""
        # Single FC layer
        base_params = 50  # Rough estimate
        return {
            'total': base_params,
            'trainable': base_params,
            'non_trainable': 0
        }
    
    def get_config(self) -> dict:
        """Get configuration for serialization."""
        config = super().get_config_dict()
        config.update({
            'activation': self.activation
        })
        return config


class AdaptiveSEBlock(MobileNetComponent):
    """Adaptive SE block that chooses between standard and efficient variants.
    
    Automatically selects the appropriate SE block variant based on
    the number of input channels to optimize for efficiency.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None,
                 reduction_ratio: int = 4,
                 efficient_threshold: int = 16,
                 activation: str = 'hard_sigmoid'):
        """Initialize adaptive SE block.
        
        Args:
            config: Component configuration
            reduction_ratio: Reduction ratio for standard SE block
            efficient_threshold: Channel threshold for using efficient variant
            activation: Activation function for gate
        """
        super().__init__(config or ComponentConfig())
        self.reduction_ratio = reduction_ratio
        self.efficient_threshold = efficient_threshold
        self.activation = activation
        self._name = "adaptive_se_block"
    
    @property
    def name(self) -> str:
        """Return component name."""
        return self._name
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply adaptive SE block (MobileNetComponent interface)."""
        return self.build_component(inputs, training or False)
        
    def build_component(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Build adaptive SE block component.
        
        Args:
            inputs: Input tensor [batch, height, width, channels]
            training: Whether in training mode
            
        Returns:
            Output tensor with SE attention applied
        """
        input_channels = inputs.shape[-1]
        
        if input_channels <= self.efficient_threshold:
            # Use efficient variant for small channel counts
            se_block = EfficientSEBlock(self.config, self.activation)
        else:
            # Use standard variant for larger channel counts
            se_block = SqueezeExciteBlock(
                self.config, self.reduction_ratio, self.activation
            )
        
        return se_block.build_component(inputs, training)
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape (same as input for SE blocks)."""
        return input_shape
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for this component."""
        # Varies based on adaptive choice
        base_params = 75  # Average estimate
        return {
            'total': base_params,
            'trainable': base_params,
            'non_trainable': 0
        }
    
    def get_config(self) -> dict:
        """Get configuration for serialization."""
        config = super().get_config_dict()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'efficient_threshold': self.efficient_threshold,
            'activation': self.activation
        })
        return config


# TensorFlow layer wrappers
class SEBlock(tf.keras.layers.Layer):
    """TensorFlow layer wrapper for SE block."""
    
    def __init__(self, reduction_ratio: int = 4, 
                 activation: str = 'hard_sigmoid', **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.activation = activation
        self.se_block = SqueezeExciteBlock(
            reduction_ratio=reduction_ratio,
            activation=activation
        )
    
    def call(self, inputs, training=None):
        return self.se_block.build_component(inputs, training or False)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'activation': self.activation
        })
        return config


class EfficientSE(tf.keras.layers.Layer):
    """TensorFlow layer wrapper for efficient SE block."""
    
    def __init__(self, activation: str = 'hard_sigmoid', **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.se_block = EfficientSEBlock(activation=activation)
    
    def call(self, inputs, training=None):
        return self.se_block.build_component(inputs, training or False)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'activation': self.activation
        })
        return config


def create_se_block(input_channels: int, 
                   reduction_ratio: int = 4,
                   efficient_threshold: int = 16,
                   activation: str = 'hard_sigmoid') -> MobileNetComponent:
    """Factory function to create appropriate SE block variant.
    
    Args:
        input_channels: Number of input channels
        reduction_ratio: Reduction ratio for standard SE block
        efficient_threshold: Channel threshold for using efficient variant
        activation: Activation function for gate
        
    Returns:
        Appropriate SE block component
    """
    if input_channels <= efficient_threshold:
        return EfficientSEBlock(activation=activation)
    else:
        return SqueezeExciteBlock(
            reduction_ratio=reduction_ratio,
            activation=activation
        ) 