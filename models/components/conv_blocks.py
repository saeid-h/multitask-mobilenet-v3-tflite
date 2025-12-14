"""
Convolution building blocks for MobileNet architectures.

This module implements the core convolution components used across
different MobileNet variants, including standard convolution blocks
and depthwise separable convolution blocks.
"""

from typing import Tuple, Dict, Any, Optional
import tensorflow as tf

from .base import ConvolutionComponent, ComponentConfig, _get_unique_layer_id


class StandardConvBlock(ConvolutionComponent):
    """Standard convolution block: Conv2D + BatchNorm + Activation.
    
    This is the basic building block used for initial convolutions
    and pointwise convolutions in MobileNet architectures.
    """
    
    def __init__(self, 
                 filters: int,
                 kernel_size: int = 3,
                 strides: int = 1,
                 config: Optional[ComponentConfig] = None):
        """Initialize standard convolution block.
        
        Args:
            filters: Number of output filters
            kernel_size: Convolution kernel size
            strides: Convolution strides
            config: Component configuration
        """
        if config is None:
            config = ComponentConfig()
        
        super().__init__(config)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        
        # Build layers
        self._conv_layer = None
        self._built = False
    
    @property
    def name(self) -> str:
        """Return component name."""
        return f"conv_block_{self.filters}_{self.kernel_size}x{self.kernel_size}_s{self.strides}"
    
    def get_kernel_size(self) -> Tuple[int, int]:
        """Get kernel size."""
        return (self.kernel_size, self.kernel_size)
    
    def get_strides(self) -> Tuple[int, int]:
        """Get strides."""
        return (self.strides, self.strides)
    
    def get_filters(self) -> int:
        """Get number of output filters."""
        return self.filters
    
    def _build_layers(self):
        """Build the convolution layer."""
        if self._built:
            return
        
        unique_id = _get_unique_layer_id()
        conv_args = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.get_padding(),
            'use_bias': self.config.use_bias,
            'name': f"{self.name_scope}_conv_{unique_id}"
        }
        
        # Add regularizer if specified
        if self.config.kernel_regularizer is not None:
            conv_args['kernel_regularizer'] = self.config.kernel_regularizer
        
        self._conv_layer = tf.keras.layers.Conv2D(**conv_args)
        self._built = True
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply the standard convolution block.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            
        Returns:
            Output tensor
        """
        self._build_layers()
        
        # Apply convolution
        x = self._conv_layer(inputs)
        
        # Apply batch normalization
        x = self._apply_batch_norm(x, training=training)
        
        # Apply activation
        x = self._apply_activation(x)
        
        return x
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape."""
        height, width, channels = input_shape
        
        # Calculate output spatial dimensions
        out_height = (height + self.strides - 1) // self.strides
        out_width = (width + self.strides - 1) // self.strides
        
        return (out_height, out_width, self.filters)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for this component."""
        if not self._built:
            self._build_layers()
        
        # Conv2D parameters
        conv_params = self.kernel_size * self.kernel_size * self.filters
        if hasattr(self._conv_layer, 'input_spec') and self._conv_layer.input_spec:
            input_channels = self._conv_layer.input_spec.axes.get(-1)
            if input_channels:
                conv_params *= input_channels
        
        # Bias parameters (if enabled)
        bias_params = self.filters if self.config.use_bias else 0
        
        # BatchNorm parameters (if enabled)
        bn_params = 4 * self.filters if self.config.use_batch_norm else 0  # gamma, beta, mean, var
        
        total_params = conv_params + bias_params
        trainable_params = conv_params + bias_params + (2 * self.filters if self.config.use_batch_norm else 0)  # gamma, beta
        non_trainable_params = (2 * self.filters if self.config.use_batch_norm else 0)  # mean, var
        
        return {
            'total': total_params + non_trainable_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }


class DepthwiseSeparableConvBlock(ConvolutionComponent):
    """Depthwise separable convolution block.
    
    This is the core building block of MobileNet architectures, consisting of:
    1. Depthwise convolution (spatial filtering)
    2. Pointwise convolution (channel mixing)
    Each followed by BatchNorm + Activation
    """
    
    def __init__(self,
                 filters: int,
                 kernel_size: int = 3,
                 strides: int = 1,
                 config: Optional[ComponentConfig] = None):
        """Initialize depthwise separable convolution block.
        
        Args:
            filters: Number of output filters (pointwise conv)
            kernel_size: Depthwise convolution kernel size
            strides: Depthwise convolution strides
            config: Component configuration
        """
        if config is None:
            config = ComponentConfig()
        
        super().__init__(config)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        
        # Build layers
        self._depthwise_layer = None
        self._pointwise_layer = None
        self._built = False
    
    @property
    def name(self) -> str:
        """Return component name."""
        return f"dw_sep_conv_{self.filters}_{self.kernel_size}x{self.kernel_size}_s{self.strides}"
    
    def get_kernel_size(self) -> Tuple[int, int]:
        """Get depthwise kernel size."""
        return (self.kernel_size, self.kernel_size)
    
    def get_strides(self) -> Tuple[int, int]:
        """Get depthwise strides."""
        return (self.strides, self.strides)
    
    def get_filters(self) -> int:
        """Get number of output filters."""
        return self.filters
    
    def _build_layers(self):
        """Build the depthwise and pointwise layers."""
        if self._built:
            return
        
        dw_id = _get_unique_layer_id()
        pw_id = _get_unique_layer_id()
        
        # Depthwise convolution layer
        dw_args = {
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.get_padding(),
            'use_bias': self.config.use_bias,
            'name': f"{self.name_scope}_dw_conv_{dw_id}"
        }
        
        if self.config.kernel_regularizer is not None:
            dw_args['kernel_regularizer'] = self.config.kernel_regularizer
        
        self._depthwise_layer = tf.keras.layers.DepthwiseConv2D(**dw_args)
        
        # Pointwise convolution layer (1x1 conv)
        pw_args = {
            'filters': self.filters,
            'kernel_size': 1,
            'strides': 1,
            'padding': 'same',
            'use_bias': self.config.use_bias,
            'name': f"{self.name_scope}_pw_conv_{pw_id}"
        }
        
        if self.config.kernel_regularizer is not None:
            pw_args['kernel_regularizer'] = self.config.kernel_regularizer
        
        self._pointwise_layer = tf.keras.layers.Conv2D(**pw_args)
        self._built = True
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply the depthwise separable convolution block.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            
        Returns:
            Output tensor
        """
        self._build_layers()
        
        # Depthwise convolution
        x = self._depthwise_layer(inputs)
        x = self._apply_batch_norm(x, training=training)
        x = self._apply_activation(x)
        
        # Pointwise convolution
        x = self._pointwise_layer(x)
        x = self._apply_batch_norm(x, training=training)
        x = self._apply_activation(x)
        
        return x
    
    def call_depthwise_only(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply only the depthwise convolution part.
        
        Useful for more complex architectures that need separate control
        over depthwise and pointwise operations.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            
        Returns:
            Output tensor after depthwise conv + BN + activation
        """
        self._build_layers()
        
        x = self._depthwise_layer(inputs)
        x = self._apply_batch_norm(x, training=training)
        x = self._apply_activation(x)
        
        return x
    
    def call_pointwise_only(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply only the pointwise convolution part.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            
        Returns:
            Output tensor after pointwise conv + BN + activation
        """
        self._build_layers()
        
        x = self._pointwise_layer(inputs)
        x = self._apply_batch_norm(x, training=training)
        x = self._apply_activation(x)
        
        return x
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape."""
        height, width, channels = input_shape
        
        # Calculate output spatial dimensions (affected by depthwise conv strides)
        out_height = (height + self.strides - 1) // self.strides
        out_width = (width + self.strides - 1) // self.strides
        
        # Output channels determined by pointwise convolution
        return (out_height, out_width, self.filters)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for this component."""
        if not self._built:
            self._build_layers()
        
        # This is an approximation - actual count depends on input channels
        # Depthwise conv: kernel_size^2 * input_channels
        # Pointwise conv: 1 * 1 * input_channels * output_filters
        
        # Conservative estimate
        depthwise_params = self.kernel_size * self.kernel_size  # per input channel
        pointwise_params = self.filters  # per input channel
        
        # Bias parameters (if enabled)
        bias_params = 0
        if self.config.use_bias:
            bias_params += 1  # depthwise bias per input channel
            bias_params += self.filters  # pointwise bias
        
        # BatchNorm parameters (2 sets if enabled)
        bn_params = 0
        if self.config.use_batch_norm:
            bn_params += 4  # depthwise BN per input channel (gamma, beta, mean, var)
            bn_params += 4 * self.filters  # pointwise BN
        
        # Note: These are per-channel estimates - actual count computed during build
        return {
            'total': depthwise_params + pointwise_params + bias_params + bn_params,
            'trainable': depthwise_params + pointwise_params + bias_params + (bn_params // 2 if self.config.use_batch_norm else 0),
            'non_trainable': bn_params // 2 if self.config.use_batch_norm else 0
        } 