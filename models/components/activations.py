"""
Mobile-Optimized Activation Functions

This module provides activation functions optimized for mobile deployment,
specifically designed for MobileNetV3 and later architectures.
"""

import tensorflow as tf
from typing import Optional, Union, Tuple, Dict
from .base import MobileNetComponent, ComponentConfig


class HardSwishActivation(MobileNetComponent):
    """Hard-Swish activation function for MobileNetV3.
    
    Hard-Swish is defined as:
    hard_swish(x) = x * hard_sigmoid(x)
    where hard_sigmoid(x) = ReLU6(x + 3) / 6
    
    This provides a more efficient approximation to Swish activation
    while maintaining similar performance characteristics.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize Hard-Swish activation.
        
        Args:
            config: Component configuration (not used for activations)
        """
        super().__init__(config or ComponentConfig())
        self._name = "hard_swish"
    
    @property
    def name(self) -> str:
        """Return component name."""
        return self._name
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply Hard-Swish activation (MobileNetComponent interface).
        
        Args:
            inputs: Input tensor
            training: Whether in training mode (not used for activations)
            
        Returns:
            Tensor with Hard-Swish activation applied
        """
        return self.build_component(inputs, training or False)
    
    def build_component(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply Hard-Swish activation to inputs.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode (not used for activations)
            
        Returns:
            Tensor with Hard-Swish activation applied
        """
        # Use Keras ReLU6 layer instead of tf.nn.relu6 for functional model compatibility
        relu6_layer = tf.keras.layers.ReLU(max_value=6.0)
        return inputs * relu6_layer(inputs + 3.0) / 6.0
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape (same as input for activations)."""
        return input_shape
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count (none for activations)."""
        return {
            'total': 0,
            'trainable': 0,
            'non_trainable': 0
        }
    
    def get_config(self) -> dict:
        """Get configuration for serialization."""
        config = super().get_config_dict()
        config.update({
            'activation_type': 'hard_swish'
        })
        return config


class HardSigmoidActivation(MobileNetComponent):
    """Hard-Sigmoid activation function for efficient computation.
    
    Hard-Sigmoid is defined as:
    hard_sigmoid(x) = ReLU6(x + 3) / 6
    
    This provides a computationally efficient approximation to the sigmoid
    function, particularly useful in mobile applications.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize Hard-Sigmoid activation.
        
        Args:
            config: Component configuration (not used for activations)
        """
        super().__init__(config or ComponentConfig())
        self._name = "hard_sigmoid"
    
    @property
    def name(self) -> str:
        """Return component name."""
        return self._name
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply Hard-Sigmoid activation (MobileNetComponent interface)."""
        return self.build_component(inputs, training or False)
    
    def build_component(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply Hard-Sigmoid activation to inputs.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode (not used for activations)
            
        Returns:
            Tensor with Hard-Sigmoid activation applied
        """
        # Use Keras ReLU6 layer instead of tf.nn.relu6 for functional model compatibility
        relu6_layer = tf.keras.layers.ReLU(max_value=6.0)
        return relu6_layer(inputs + 3.0) / 6.0
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape (same as input for activations)."""
        return input_shape
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count (none for activations)."""
        return {
            'total': 0,
            'trainable': 0,
            'non_trainable': 0
        }
    
    def get_config(self) -> dict:
        """Get configuration for serialization."""
        config = super().get_config_dict()
        config.update({
            'activation_type': 'hard_sigmoid'
        })
        return config


class ReLU6Activation(MobileNetComponent):
    """ReLU6 activation function for mobile optimization.
    
    ReLU6 is commonly used in mobile networks as it provides
    bounded activation values that are quantization-friendly.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize ReLU6 activation.
        
        Args:
            config: Component configuration (not used for activations)
        """
        super().__init__(config or ComponentConfig())
        self._name = "relu6"
    
    @property
    def name(self) -> str:
        """Return component name."""
        return self._name
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply ReLU6 activation (MobileNetComponent interface)."""
        return self.build_component(inputs, training or False)
    
    def build_component(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply ReLU6 activation to inputs.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode (not used for activations)
            
        Returns:
            Tensor with ReLU6 activation applied
        """
        # Use Keras ReLU layer instead of tf.nn.relu6 for functional model compatibility
        relu6_layer = tf.keras.layers.ReLU(max_value=6.0)
        return relu6_layer(inputs)
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape (same as input for activations)."""
        return input_shape
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count (none for activations)."""
        return {
            'total': 0,
            'trainable': 0,
            'non_trainable': 0
        }
    
    def get_config(self) -> dict:
        """Get configuration for serialization."""
        config = super().get_config_dict()
        config.update({
            'activation_type': 'relu6'
        })
        return config


def get_activation(activation_name: str) -> MobileNetComponent:
    """Factory function to get activation by name.
    
    Args:
        activation_name: Name of the activation function
        
    Returns:
        Activation component instance
        
    Raises:
        ValueError: If activation name is not recognized
    """
    activation_map = {
        'hard_swish': HardSwishActivation,
        'hard_sigmoid': HardSigmoidActivation,
        'relu6': ReLU6Activation,
    }
    
    if activation_name not in activation_map:
        raise ValueError(
            f"Activation '{activation_name}' not found. "
            f"Available activations: {list(activation_map.keys())}"
        )
    
    return activation_map[activation_name]()


def apply_activation(
    inputs: tf.Tensor, 
    activation: Union[str, MobileNetComponent, None],
    training: bool = False
) -> tf.Tensor:
    """Apply activation function to inputs.
    
    Args:
        inputs: Input tensor
        activation: Activation function (string name, component, or None)
        training: Whether in training mode
        
    Returns:
        Tensor with activation applied
    """
    if activation is None:
        return inputs
    elif isinstance(activation, str):
        activation_component = get_activation(activation)
        return activation_component.build_component(inputs, training)
    elif isinstance(activation, MobileNetComponent):
        return activation.build_component(inputs, training)
    else:
        raise ValueError(f"Invalid activation type: {type(activation)}")


# TensorFlow layer wrappers for easier integration
class HardSwish(tf.keras.layers.Layer):
    """TensorFlow layer wrapper for Hard-Swish activation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation = HardSwishActivation()
    
    def call(self, inputs, training=None):
        return self.activation.build_component(inputs, training or False)
    
    def get_config(self):
        return super().get_config()


class HardSigmoid(tf.keras.layers.Layer):
    """TensorFlow layer wrapper for Hard-Sigmoid activation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation = HardSigmoidActivation()
    
    def call(self, inputs, training=None):
        return self.activation.build_component(inputs, training or False)
    
    def get_config(self):
        return super().get_config()


# Standalone functions for direct use
def hard_swish(x: tf.Tensor) -> tf.Tensor:
    """Apply Hard-Swish activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with Hard-Swish activation applied
    """
    # Use Keras ReLU6 layer for functional compatibility
    relu6_layer = tf.keras.layers.ReLU(max_value=6.0)
    return x * relu6_layer(x + 3.0) / 6.0


def hard_sigmoid(x: tf.Tensor) -> tf.Tensor:
    """Apply Hard-Sigmoid activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with Hard-Sigmoid activation applied
    """
    # Use Keras ReLU6 layer for functional compatibility
    relu6_layer = tf.keras.layers.ReLU(max_value=6.0)
    return relu6_layer(x + 3.0) / 6.0 