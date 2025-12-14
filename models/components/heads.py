"""
Classification head components for MobileNet architectures.

This module implements the final components of MobileNet models,
including global pooling and dense classification layers.
"""

from typing import Tuple, Dict, Any, Optional
import tensorflow as tf

from .base import PoolingComponent, ClassificationComponent, ComponentConfig, _get_unique_layer_id


class GlobalAveragePoolingBlock(PoolingComponent):
    """Global average pooling component.
    
    Reduces spatial dimensions to 1x1 by taking the average
    across spatial dimensions for each channel.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize global average pooling block.
        
        Args:
            config: Component configuration
        """
        if config is None:
            config = ComponentConfig()
        
        super().__init__(config)
        self._pool_layer = None
        self._built = False
    
    @property
    def name(self) -> str:
        """Return component name."""
        return "global_avg_pool"
    
    def get_pool_size(self) -> Optional[Tuple[int, int]]:
        """Return None for global pooling."""
        return None
    
    def _build_layers(self):
        """Build the pooling layer."""
        if self._built:
            return
        
        unique_id = _get_unique_layer_id()
        self._pool_layer = tf.keras.layers.GlobalAveragePooling2D(
            name=f"{self.name_scope}_gap_{unique_id}"
        )
        self._built = True
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply global average pooling.
        
        Args:
            inputs: Input tensor of shape (batch, height, width, channels)
            training: Training mode flag (unused for pooling)
            
        Returns:
            Output tensor of shape (batch, channels)
        """
        self._build_layers()
        return self._pool_layer(inputs)
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape."""
        height, width, channels = input_shape
        return (channels,)  # Spatial dimensions are collapsed
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count (none for pooling)."""
        return {
            'total': 0,
            'trainable': 0,
            'non_trainable': 0
        }


class DenseClassificationHead(ClassificationComponent):
    """Dense classification head with optional dropout.
    
    Final classification layer that maps from feature vector
    to class logits.
    """
    
    def __init__(self,
                 num_classes: int,
                 dropout_rate: float = 0.0,
                 config: Optional[ComponentConfig] = None):
        """Initialize dense classification head.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate (0.0 to disable)
            config: Component configuration
        """
        if config is None:
            config = ComponentConfig()
        
        super().__init__(config)
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Build layers
        self._dropout_layer = None
        self._dense_layer = None
        self._built = False
    
    @property
    def name(self) -> str:
        """Return component name."""
        dropout_str = f"_drop{self.dropout_rate}" if self.dropout_rate > 0 else ""
        return f"dense_head_{self.num_classes}{dropout_str}"
    
    def get_num_classes(self) -> int:
        """Get number of output classes."""
        return self.num_classes
    
    def supports_dropout(self) -> bool:
        """Whether this component supports dropout."""
        return True
    
    def _build_layers(self):
        """Build the classification layers."""
        if self._built:
            return
        
        dropout_id = _get_unique_layer_id()
        dense_id = _get_unique_layer_id()
        
        # Dropout layer (if enabled)
        if self.dropout_rate > 0:
            self._dropout_layer = tf.keras.layers.Dropout(
                rate=self.dropout_rate,
                name=f"{self.name_scope}_dropout_{dropout_id}"
            )
        
        # Dense classification layer
        dense_args = {
            'units': self.num_classes,
            'use_bias': True,  # Classification head typically uses bias
            'name': f"{self.name_scope}_dense_{dense_id}"
        }
        
        # Add regularizer if specified
        if self.config.kernel_regularizer is not None:
            dense_args['kernel_regularizer'] = self.config.kernel_regularizer
        
        self._dense_layer = tf.keras.layers.Dense(**dense_args)
        self._built = True
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply the classification head.
        
        Args:
            inputs: Input tensor (typically from global pooling)
            training: Training mode flag
            
        Returns:
            Output logits tensor
        """
        self._build_layers()
        
        x = inputs
        
        # Apply dropout if enabled
        if self.dropout_rate > 0 and self._dropout_layer is not None:
            x = self._dropout_layer(x, training=training)
        
        # Apply dense layer
        x = self._dense_layer(x)
        
        return x
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape."""
        return (self.num_classes,)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for this component."""
        if not self._built:
            self._build_layers()
        
        # This is an approximation - actual count depends on input features
        # Dense layer: input_features * num_classes + num_classes (bias)
        
        # Conservative estimate (will be updated when built with actual input)
        bias_params = self.num_classes  # Always have bias in classification head
        
        return {
            'total': bias_params,  # Weights calculated at build time
            'trainable': bias_params,
            'non_trainable': 0
        }


class MobileNetClassificationHead(ClassificationComponent):
    """Complete MobileNet classification head.
    
    Combines global average pooling with dense classification,
    matching the standard MobileNet head structure.
    """
    
    def __init__(self,
                 num_classes: int,
                 dropout_rate: float = 0.0,
                 config: Optional[ComponentConfig] = None):
        """Initialize complete classification head.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate (0.0 to disable)
            config: Component configuration
        """
        if config is None:
            config = ComponentConfig()
        
        super().__init__(config)
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Create sub-components
        self.global_pool = GlobalAveragePoolingBlock(config)
        self.global_pool.set_name_scope(f"{self.name_scope}_pool")
        
        self.dense_head = DenseClassificationHead(num_classes, dropout_rate, config)
        self.dense_head.set_name_scope(f"{self.name_scope}_classifier")
    
    @property
    def name(self) -> str:
        """Return component name."""
        dropout_str = f"_drop{self.dropout_rate}" if self.dropout_rate > 0 else ""
        return f"mobilenet_head_{self.num_classes}{dropout_str}"
    
    def get_num_classes(self) -> int:
        """Get number of output classes."""
        return self.num_classes
    
    def supports_dropout(self) -> bool:
        """Whether this component supports dropout."""
        return True
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply the complete classification head.
        
        Args:
            inputs: Input tensor from backbone (height, width, channels)
            training: Training mode flag
            
        Returns:
            Output logits tensor
        """
        # Global average pooling
        x = self.global_pool.call(inputs, training=training)
        
        # Classification layer
        x = self.dense_head.call(x, training=training)
        
        return x
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape."""
        return (self.num_classes,)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for this component."""
        pool_params = self.global_pool.get_parameter_count()
        dense_params = self.dense_head.get_parameter_count()
        
        return {
            'total': pool_params['total'] + dense_params['total'],
            'trainable': pool_params['trainable'] + dense_params['trainable'],
            'non_trainable': pool_params['non_trainable'] + dense_params['non_trainable']
        } 