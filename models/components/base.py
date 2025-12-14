"""
Base classes and interfaces for MobileNet building blocks.

This module defines abstract base classes that all MobileNet components
must implement, ensuring consistent interfaces and configuration patterns
across different architectural components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import tensorflow as tf


# Global counter for unique layer naming
_layer_counter = 0

def _get_unique_layer_id() -> int:
    """Get unique layer ID for naming."""
    global _layer_counter
    _layer_counter += 1
    return _layer_counter


@dataclass
class ComponentConfig:
    """Configuration for building block components."""
    
    # Regularization parameters
    use_bias: bool = False
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
    
    # Batch normalization parameters
    batch_norm_momentum: float = 0.99
    batch_norm_epsilon: float = 1e-3
    use_batch_norm: bool = True
    
    # Activation parameters
    activation: str = 'relu6'
    
    # Component-specific parameters
    component_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize component_params if None."""
        if self.component_params is None:
            self.component_params = {}
    
    def validate(self) -> None:
        """Validate component configuration."""
        # Validate activation
        valid_activations = ['relu', 'relu6', 'swish', 'hard_swish', 'gelu']
        if self.activation.lower() not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, got {self.activation}")
        
        # Validate batch norm parameters
        if not 0 < self.batch_norm_momentum < 1:
            raise ValueError(f"batch_norm_momentum must be in (0, 1), got {self.batch_norm_momentum}")
        
        if self.batch_norm_epsilon <= 0:
            raise ValueError(f"batch_norm_epsilon must be positive, got {self.batch_norm_epsilon}")


class MobileNetComponent(ABC):
    """Abstract base class for all MobileNet building blocks."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize the component with configuration.
        
        Args:
            config: Component configuration object
        """
        self.config = config
        self.config.validate()
        self._name_scope: Optional[str] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the component name for TensorFlow naming."""
        pass
    
    @property
    def name_scope(self) -> str:
        """Return the name scope for this component."""
        return self._name_scope or self.name
    
    def set_name_scope(self, scope: str) -> None:
        """Set the name scope for this component."""
        self._name_scope = scope
    
    @abstractmethod
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply the component to input tensor.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode (for batch norm, dropout)
            
        Returns:
            Output tensor after applying the component
        """
        pass
    
    @abstractmethod
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape given input shape.
        
        Args:
            input_shape: Input tensor shape (excluding batch dimension)
            
        Returns:
            Output tensor shape (excluding batch dimension)
        """
        pass
    
    @abstractmethod
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count information for this component.
        
        Returns:
            Dictionary with 'total', 'trainable', 'non_trainable' counts
        """
        pass
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary for serialization."""
        return {
            'name': self.name,
            'use_bias': self.config.use_bias,
            'batch_norm_momentum': self.config.batch_norm_momentum,
            'batch_norm_epsilon': self.config.batch_norm_epsilon,
            'use_batch_norm': self.config.use_batch_norm,
            'activation': self.config.activation,
            'component_params': self.config.component_params,
        }
    
    def _get_activation_layer(self) -> tf.keras.layers.Layer:
        """Get activation layer based on configuration."""
        
        def hard_swish_activation(x):
            """Hard Swish activation function for TF version compatibility."""
            return x * tf.nn.relu6(x + 3.0) / 6.0
        
        activation_map = {
            'relu': tf.keras.layers.ReLU(),
            'relu6': tf.keras.layers.ReLU(6.0),
            'swish': tf.keras.layers.Activation('swish'),
            'hard_swish': tf.keras.layers.Activation(hard_swish_activation),
            'gelu': tf.keras.layers.Activation('gelu'),
        }
        
        activation = self.config.activation.lower()
        return activation_map[activation]
    
    def _apply_batch_norm(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply batch normalization if enabled.
        
        Args:
            x: Input tensor
            training: Training mode for batch normalization
            
        Returns:
            Tensor after optional batch normalization
        """
        if not self.config.use_batch_norm:
            return x
        
        # Use unique names to avoid conflicts
        unique_id = _get_unique_layer_id()
        bn_name = f"{self.name_scope}_bn_{unique_id}"
        
        return tf.keras.layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum,
            epsilon=self.config.batch_norm_epsilon,
            name=bn_name
        )(x, training=training)
    
    def _apply_activation(self, x: tf.Tensor) -> tf.Tensor:
        """Apply activation function.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor after activation
        """
        activation_layer = self._get_activation_layer()
        unique_id = _get_unique_layer_id()
        activation_layer._name = f"{self.name_scope}_activation_{unique_id}"
        return activation_layer(x)


class ConvolutionComponent(MobileNetComponent):
    """Base class for convolution-based components."""
    
    @abstractmethod
    def get_kernel_size(self) -> Tuple[int, int]:
        """Get kernel size for this convolution component."""
        pass
    
    @abstractmethod
    def get_strides(self) -> Tuple[int, int]:
        """Get strides for this convolution component."""
        pass
    
    @abstractmethod
    def get_filters(self) -> int:
        """Get number of output filters for this convolution component."""
        pass
    
    def get_padding(self) -> str:
        """Get padding type for this convolution component."""
        return 'same'  # Default to 'same' padding for MobileNet


class PoolingComponent(MobileNetComponent):
    """Base class for pooling-based components."""
    
    @abstractmethod
    def get_pool_size(self) -> Optional[Tuple[int, int]]:
        """Get pooling size. Return None for global pooling."""
        pass


class ClassificationComponent(MobileNetComponent):
    """Base class for classification head components."""
    
    @abstractmethod
    def get_num_classes(self) -> int:
        """Get number of output classes."""
        pass
    
    @abstractmethod
    def supports_dropout(self) -> bool:
        """Whether this component supports dropout."""
        pass 