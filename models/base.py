"""
Abstract base class and configuration schema for MobileNet architectures.

This module defines the common interface that all MobileNet variants must
implement, along with the configuration schema for model parameters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List
import tensorflow as tf


@dataclass
class ModelConfig:
    """Configuration schema for MobileNet architectures.
    
    This dataclass defines the common parameters that all MobileNet variants
    can use, along with architecture-specific parameters stored in the
    `arch_params` dictionary.
    
    Attributes:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        weight_decay: L2 regularization weight decay
        batch_norm_momentum: Batch normalization momentum
        batch_norm_epsilon: Batch normalization epsilon
        activation: Activation function name ('relu', 'relu6', 'swish', etc.)
        arch_params: Architecture-specific parameters
        optimization_params: Optimization-specific parameters
    """
    
    # Common parameters for all architectures
    input_shape: Tuple[int, int, int] = (96, 96, 1)
    num_classes: int = 2
    dropout_rate: float = 0.2
    weight_decay: float = 1e-4
    
    # Batch normalization parameters
    batch_norm_momentum: float = 0.99
    batch_norm_epsilon: float = 1e-3
    
    # Activation function
    activation: str = 'relu6'
    
    # Architecture-specific parameters
    arch_params: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization parameters (quantization, pruning, etc.)
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    
    # Derived input configuration properties (computed on demand)
    @property
    def input_resolution(self) -> Tuple[int, int]:
        """Get input resolution (height, width)."""
        return (self.input_shape[0], self.input_shape[1])
    
    @property
    def input_channels(self) -> int:
        """Get number of input channels."""
        return self.input_shape[2]
    
    @property
    def is_grayscale(self) -> bool:
        """Check if input is grayscale (1 channel)."""
        return self.input_channels == 1
    
    @property
    def is_rgb(self) -> bool:
        """Check if input is RGB (3 channels)."""
        return self.input_channels == 3
    
    @property
    def input_config_name(self) -> str:
        """Get input configuration name (e.g., '96x96x1', '224x224x3')."""
        from src.utils.constants import get_input_config_name
        return get_input_config_name(self.input_shape)
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate input shape
        self.validate_input_config()
        
        if self.num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")
        
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        
        if not 0.0 < self.batch_norm_momentum < 1.0:
            raise ValueError(f"batch_norm_momentum must be in (0, 1), got {self.batch_norm_momentum}")
        
        if self.batch_norm_epsilon <= 0.0:
            raise ValueError(f"batch_norm_epsilon must be positive, got {self.batch_norm_epsilon}")
        
        valid_activations = {'relu', 'relu6', 'swish', 'hard_swish', 'gelu'}
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, got {self.activation}")
    
    def validate_input_config(self) -> None:
        """Validate input configuration parameters.
        
        Raises:
            ValueError: If input configuration is invalid
        """
        if len(self.input_shape) != 3:
            raise ValueError(f"input_shape must have 3 dimensions, got {len(self.input_shape)}")
        
        height, width, channels = self.input_shape
        
        # Validate resolution (must be square)
        if height != width:
            raise ValueError(f"Input must be square, got {height}x{width}")
        
        # Validate supported resolutions
        from src.utils.constants import SUPPORTED_RESOLUTIONS
        if height not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Unsupported input resolution: {height}. Supported: {SUPPORTED_RESOLUTIONS}")
        
        # Validate channels
        from src.utils.constants import SUPPORTED_CHANNELS
        if channels not in SUPPORTED_CHANNELS:
            raise ValueError(f"Unsupported channel count: {channels}. Supported: {SUPPORTED_CHANNELS}")
        
        # Validate against supported configurations
        from src.utils.constants import validate_input_config
        if not validate_input_config(self.input_shape):
            raise ValueError(f"Unsupported input configuration: {self.input_shape}")


class MobileNetArchitecture(ABC):
    """Abstract base class for MobileNet architecture implementations.
    
    This class defines the common interface that all MobileNet variants
    (V1, V3-Small, V4-Conv-S) must implement. It provides a consistent
    API for model creation, configuration, and introspection.
    
    Attributes:
        config: Model configuration parameters
        name: Architecture name (e.g., 'mobilenet_v1', 'mobilenet_v3_small')
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize the architecture with configuration.
        
        Args:
            config: Model configuration parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        config.validate()
        self.config = config
        self._model: Optional[tf.keras.Model] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the architecture name."""
        pass
    
    @property
    @abstractmethod
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        """Return list of supported input shapes for this architecture."""
        pass
    
    @property
    @abstractmethod
    def parameter_count_range(self) -> Tuple[int, int]:
        """Return expected parameter count range (min, max) for this architecture."""
        pass
    
    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        """Build and return the TensorFlow model.
        
        This method should create the complete model architecture according
        to the configuration parameters.
        
        Returns:
            Compiled TensorFlow Keras model
            
        Raises:
            ValueError: If model cannot be built with current configuration
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> None:
        """Validate architecture-specific configuration parameters.
        
        This method should check that all architecture-specific parameters
        in config.arch_params are valid for this architecture.
        
        Raises:
            ValueError: If any architecture-specific parameter is invalid
        """
        pass
    
    def get_model(self) -> tf.keras.Model:
        """Get or create the model instance.
        
        This method implements lazy loading - the model is only built
        when first requested.
        
        Returns:
            TensorFlow Keras model instance
        """
        if self._model is None:
            self.validate_config()
            self._model = self.build_model()
        return self._model
    
    def get_model_config(self) -> Dict[str, Any]:
        """Return the complete model configuration as a dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'architecture': self.name,
            'input_shape': self.config.input_shape,
            'num_classes': self.config.num_classes,
            'dropout_rate': self.config.dropout_rate,
            'weight_decay': self.config.weight_decay,
            'batch_norm_momentum': self.config.batch_norm_momentum,
            'batch_norm_epsilon': self.config.batch_norm_epsilon,
            'activation': self.config.activation,
            'arch_params': self.config.arch_params.copy(),
            'optimization_params': self.config.optimization_params.copy(),
        }
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Return model parameter count information.
        
        Returns:
            Dictionary with parameter count details:
            - total: Total number of parameters
            - trainable: Number of trainable parameters  
            - non_trainable: Number of non-trainable parameters
        """
        model = self.get_model()
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    def get_model_summary(self) -> str:
        """Return a string summary of the model architecture.
        
        Returns:
            Model summary string
        """
        model = self.get_model()
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
    
    def validate_input_shape(self, input_shape: Tuple[int, int, int]) -> None:
        """Validate that the input shape is supported by this architecture.
        
        Args:
            input_shape: Input shape to validate
            
        Raises:
            ValueError: If input shape is not supported
        """
        if input_shape not in self.supported_input_shapes:
            raise ValueError(
                f"Input shape {input_shape} not supported by {self.name}. "
                f"Supported shapes: {self.supported_input_shapes}"
            )
    
    def __str__(self) -> str:
        """Return string representation of the architecture."""
        param_info = self.get_parameter_count()
        return (
            f"{self.name}("
            f"input_shape={self.config.input_shape}, "
            f"num_classes={self.config.num_classes}, "
            f"total_params={param_info['total']:,})"
        )
    
    def __repr__(self) -> str:
        """Return detailed string representation of the architecture."""
        return f"{self.__class__.__name__}(config={self.config})" 