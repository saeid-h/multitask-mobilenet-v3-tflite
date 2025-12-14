"""
Inverted Residual Blocks for MobileNetV3

This module implements inverted residual blocks as used in MobileNetV3
architectures. These blocks extend the MobileNetV2 inverted residuals
with Squeeze-and-Excitation blocks and hard-swish activations.
"""

import tensorflow as tf
from typing import Optional, Union, Tuple, Dict
from .base import MobileNetComponent, ComponentConfig
from .activations import apply_activation
from .se_blocks import SqueezeExciteBlock, create_se_block

# Global counter for unique inverted block naming
_inverted_block_counter = 0

def _get_unique_inv_id() -> int:
    """Get unique identifier for inverted block layer naming."""
    global _inverted_block_counter
    _inverted_block_counter += 1
    return _inverted_block_counter

class InvertedResidualBlock(MobileNetComponent):
    """MobileNetV3 Inverted Residual Block with SE support.
    
    The block structure:
    1. 1x1 expansion convolution (if expansion_ratio > 1)
    2. Depthwise convolution with specified stride
    3. Squeeze-and-Excitation block (optional)
    4. 1x1 projection convolution
    5. Residual connection (if input/output shapes match)
    
    Supports both ReLU and Hard-Swish activations as used in MobileNetV3.
    """
    
    def __init__(self, 
                 config: Optional[ComponentConfig] = None,
                 output_channels: int = 32,
                 expansion_ratio: int = 6,
                 stride: int = 1,
                 use_se: bool = False,
                 se_reduction_ratio: int = 4,
                 activation: str = 'relu6',
                 kernel_size: int = 3):
        """Initialize inverted residual block.
        
        Args:
            config: Component configuration
            output_channels: Number of output channels
            expansion_ratio: Expansion ratio for bottleneck
            stride: Stride for depthwise convolution
            use_se: Whether to use Squeeze-and-Excitation
            se_reduction_ratio: SE block reduction ratio
            activation: Activation function ('relu6' or 'hard_swish')
            kernel_size: Depthwise convolution kernel size
        """
        super().__init__(config)
        self.output_channels = output_channels
        self.expansion_ratio = expansion_ratio
        self.stride = stride
        self.use_se = use_se
        self.se_reduction_ratio = se_reduction_ratio
        self.activation = activation
        self.kernel_size = kernel_size
        self._name = f"inv_res_exp{expansion_ratio}_out{output_channels}_s{stride}"
        
        if use_se:
            self._name += f"_se{se_reduction_ratio}"
    
    @property
    def name(self) -> str:
        """Return component name."""
        return self._name
    
    def build_component(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Build inverted residual block.
        
        Args:
            inputs: Input tensor [batch, height, width, input_channels]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch, height/stride, width/stride, output_channels]
        """
        input_channels = inputs.shape[-1]
        expanded_channels = input_channels * self.expansion_ratio
        
        # Get unique identifier for this block instance
        unique_id = _get_unique_inv_id()
        
        x = inputs
        
        # 1. Expansion phase (1x1 conv) - only if expansion_ratio > 1
        if self.expansion_ratio > 1:
            x = tf.keras.layers.Conv2D(
                expanded_channels,
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                name=f'{self._name}_expand_{unique_id}'
            )(x)
            
            x = tf.keras.layers.BatchNormalization(
                momentum=self.config.batch_norm_momentum if self.config else 0.999,
                epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
                name=f'{self._name}_expand_bn_{unique_id}'
            )(x, training=training)
            
            x = apply_activation(x, self.activation, training)
        
        # 2. Depthwise convolution phase
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='same',
            use_bias=False,
            depthwise_initializer='he_normal',
            name=f'{self._name}_depthwise_{unique_id}'
        )(x)
        
        x = tf.keras.layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum if self.config else 0.999,
            epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
            name=f'{self._name}_depthwise_bn_{unique_id}'
        )(x, training=training)
        
        x = apply_activation(x, self.activation, training)
        
        # 3. Squeeze-and-Excitation block (optional)
        if self.use_se:
            se_block = create_se_block(
                input_channels=expanded_channels,
                reduction_ratio=self.se_reduction_ratio,
                activation='hard_sigmoid'
            )
            x = se_block.build_component(x, training)
        
        # 4. Projection phase (1x1 conv)
        x = tf.keras.layers.Conv2D(
            self.output_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name=f'{self._name}_project_{unique_id}'
        )(x)
        
        x = tf.keras.layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum if self.config else 0.999,
            epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
            name=f'{self._name}_project_bn_{unique_id}'
        )(x, training=training)
        
        # No activation after projection (linear)
        
        # 5. Residual connection
        if (self.stride == 1 and 
            input_channels == self.output_channels and 
            inputs.shape[1:] == x.shape[1:]):
            x = tf.keras.layers.Add(name=f'{self._name}_add_{unique_id}')([inputs, x])
        
        return x
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply inverted residual block (MobileNetComponent interface).
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        return self.build_component(inputs, training or False)
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape given input shape.
        
        Args:
            input_shape: Input tensor shape (height, width, channels)
            
        Returns:
            Output tensor shape (height/stride, width/stride, output_channels)
        """
        height, width, channels = input_shape
        
        # Calculate output spatial dimensions
        out_height = (height + self.stride - 1) // self.stride
        out_width = (width + self.stride - 1) // self.stride
        
        return (out_height, out_width, self.output_channels)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for this component.
        
        Returns:
            Dictionary with parameter count details
        """
        # This is an approximation - actual count depends on input channels
        # For inverted residual blocks, parameters come from:
        # 1. Expansion conv (if expansion_ratio > 1)
        # 2. Depthwise conv
        # 3. SE block (if used)
        # 4. Projection conv
        # 5. Batch norm layers
        
        # Rough estimate based on typical configurations
        # This would be more accurate if calculated during model building
        base_params = self.output_channels * 50  # Rough estimate
        
        if self.use_se:
            base_params += self.output_channels // self.se_reduction_ratio * 2
        
        return {
            'total': base_params,
            'trainable': base_params,
            'non_trainable': 0
        }
    
    def get_config(self) -> dict:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'expansion_ratio': self.expansion_ratio,
            'stride': self.stride,
            'use_se': self.use_se,
            'se_reduction_ratio': self.se_reduction_ratio,
            'activation': self.activation,
            'kernel_size': self.kernel_size
        })
        return config


class MobileNetV3Block(MobileNetComponent):
    """Convenience wrapper for MobileNetV3-style inverted residual blocks.
    
    This class provides a simplified interface for creating blocks with
    MobileNetV3-specific defaults and naming conventions.
    """
    
    def __init__(self,
                 config: Optional[ComponentConfig] = None,
                 output_channels: int = 32,
                 expansion_ratio: int = 6,
                 stride: int = 1,
                 use_se: bool = False,
                 activation: str = 'relu6',
                 kernel_size: int = 3):
        """Initialize MobileNetV3 block.
        
        Args:
            config: Component configuration
            output_channels: Number of output channels
            expansion_ratio: Expansion ratio for bottleneck
            stride: Stride for depthwise convolution
            use_se: Whether to use Squeeze-and-Excitation
            activation: Activation function ('relu6' or 'hard_swish')
            kernel_size: Depthwise convolution kernel size
        """
        super().__init__(config)
        
        # Set SE reduction ratio based on MobileNetV3 paper guidelines
        se_reduction_ratio = 4  # Standard ratio for MobileNetV3
        
        self.inv_res_block = InvertedResidualBlock(
            config=config,
            output_channels=output_channels,
            expansion_ratio=expansion_ratio,
            stride=stride,
            use_se=use_se,
            se_reduction_ratio=se_reduction_ratio,
            activation=activation,
            kernel_size=kernel_size
        )
        
        self.name = f"mobilenetv3_block_out{output_channels}"
    
    def build_component(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Build MobileNetV3 block component."""
        return self.inv_res_block.build_component(inputs, training)
    
    def get_config(self) -> dict:
        """Get configuration for serialization."""
        return self.inv_res_block.get_config()


class LinearBottleneck(MobileNetComponent):
    """Linear bottleneck block for very efficient models.
    
    A simplified version of the inverted residual block that omits
    the expansion phase for very small models (α ≤ 0.25).
    """
    
    def __init__(self,
                 config: Optional[ComponentConfig] = None,
                 output_channels: int = 32,
                 stride: int = 1,
                 use_se: bool = False,
                 activation: str = 'relu6',
                 kernel_size: int = 3):
        """Initialize linear bottleneck block.
        
        Args:
            config: Component configuration
            output_channels: Number of output channels
            stride: Stride for depthwise convolution
            use_se: Whether to use Squeeze-and-Excitation
            activation: Activation function
            kernel_size: Depthwise convolution kernel size
        """
        super().__init__(config)
        self.output_channels = output_channels
        self.stride = stride
        self.use_se = use_se
        self.activation = activation
        self.kernel_size = kernel_size
        self.name = f"linear_bottleneck_out{output_channels}_s{stride}"
    
    def build_component(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Build linear bottleneck block.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        input_channels = inputs.shape[-1]
        x = inputs
        
        # Depthwise convolution
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='same',
            use_bias=False,
            depthwise_initializer='he_normal',
            name=f'{self.name}_depthwise'
        )(x)
        
        x = tf.keras.layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum if self.config else 0.999,
            epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
            name=f'{self.name}_depthwise_bn'
        )(x, training=training)
        
        x = apply_activation(x, self.activation, training)
        
        # SE block (optional)
        if self.use_se:
            se_block = create_se_block(
                input_channels=input_channels,
                reduction_ratio=4,
                activation='hard_sigmoid'
            )
            x = se_block.build_component(x, training)
        
        # Projection
        x = tf.keras.layers.Conv2D(
            self.output_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name=f'{self.name}_project'
        )(x)
        
        x = tf.keras.layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum if self.config else 0.999,
            epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
            name=f'{self.name}_project_bn'
        )(x, training=training)
        
        # Residual connection
        if (self.stride == 1 and 
            input_channels == self.output_channels and 
            inputs.shape[1:] == x.shape[1:]):
            x = tf.keras.layers.Add(name=f'{self.name}_add')([inputs, x])
        
        return x
    
    def get_config(self) -> dict:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'stride': self.stride,
            'use_se': self.use_se,
            'activation': self.activation,
            'kernel_size': self.kernel_size
        })
        return config


# TensorFlow layer wrappers
class InvertedResidual(tf.keras.layers.Layer):
    """TensorFlow layer wrapper for inverted residual block."""
    
    def __init__(self, 
                 output_channels: int,
                 expansion_ratio: int = 6,
                 stride: int = 1,
                 use_se: bool = False,
                 activation: str = 'relu6',
                 **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels
        self.expansion_ratio = expansion_ratio
        self.stride = stride
        self.use_se = use_se
        self.activation = activation
        
        self.block = InvertedResidualBlock(
            output_channels=output_channels,
            expansion_ratio=expansion_ratio,
            stride=stride,
            use_se=use_se,
            activation=activation
        )
    
    def call(self, inputs, training=None):
        return self.block.build_component(inputs, training or False)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'expansion_ratio': self.expansion_ratio,
            'stride': self.stride,
            'use_se': self.use_se,
            'activation': self.activation
        })
        return config 


class MobileNetV2InvertedResidualBlock(MobileNetComponent):
    """MobileNetV2 Inverted Residual Block with Linear Bottleneck.
    
    This is the core building block of MobileNetV2, featuring:
    1. 1x1 expansion convolution with ReLU6 (if expansion_ratio > 1)
    2. 3x3 depthwise convolution with ReLU6
    3. 1x1 linear projection (no activation - linear bottleneck)
    4. Residual connection when input/output dimensions match and stride=1
    
    Key differences from MobileNetV3:
    - No Squeeze-and-Excitation blocks
    - Always uses ReLU6 activation (except linear projection)
    - Fixed 3x3 kernel size for depthwise conv
    - Simpler, more streamlined design
    """
    
    def __init__(self, 
                 config: Optional[ComponentConfig] = None,
                 output_channels: int = 32,
                 expansion_ratio: int = 6,
                 stride: int = 1):
        """Initialize MobileNetV2 inverted residual block.
        
        Args:
            config: Component configuration
            output_channels: Number of output channels
            expansion_ratio: Expansion ratio for bottleneck (typically 6)
            stride: Stride for depthwise convolution (1 or 2)
        """
        # Provide default config if None
        if config is None:
            config = ComponentConfig()
        
        super().__init__(config)
        self.output_channels = output_channels
        self.expansion_ratio = expansion_ratio
        self.stride = stride
        self._name = f"v2_inv_res_t{expansion_ratio}_{output_channels}_s{stride}"
    
    @property
    def name(self) -> str:
        """Return component name."""
        return self._name
    
    def build_component(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Build MobileNetV2 inverted residual block.
        
        Args:
            inputs: Input tensor [batch, height, width, input_channels]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch, height/stride, width/stride, output_channels]
        """
        input_channels = inputs.shape[-1]
        expanded_channels = input_channels * self.expansion_ratio
        
        # Get unique identifier for this block instance
        unique_id = _get_unique_inv_id()
        
        x = inputs
        
        # 1. Expansion phase (1x1 conv with ReLU6) - only if expansion_ratio > 1
        if self.expansion_ratio > 1:
            x = tf.keras.layers.Conv2D(
                expanded_channels,
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                name=f'{self._name}_expand_{unique_id}'
            )(x)
            
            x = tf.keras.layers.BatchNormalization(
                momentum=self.config.batch_norm_momentum if self.config else 0.999,
                epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
                name=f'{self._name}_expand_bn_{unique_id}'
            )(x, training=training)
            
            # Always use ReLU6 for expansion in MobileNetV2
            x = tf.keras.layers.ReLU(max_value=6.0, name=f'{self._name}_expand_relu_{unique_id}')(x)
        
        # 2. Depthwise convolution phase (3x3 kernel with ReLU6)
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,  # Fixed 3x3 for MobileNetV2
            strides=self.stride,
            padding='same',
            use_bias=False,
            depthwise_initializer='he_normal',
            name=f'{self._name}_depthwise_{unique_id}'
        )(x)
        
        x = tf.keras.layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum if self.config else 0.999,
            epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
            name=f'{self._name}_depthwise_bn_{unique_id}'
        )(x, training=training)
        
        # Always use ReLU6 for depthwise in MobileNetV2
        x = tf.keras.layers.ReLU(max_value=6.0, name=f'{self._name}_depthwise_relu_{unique_id}')(x)
        
        # 3. Linear projection phase (1x1 conv with NO activation - linear bottleneck)
        x = tf.keras.layers.Conv2D(
            self.output_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name=f'{self._name}_project_{unique_id}'
        )(x)
        
        x = tf.keras.layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum if self.config else 0.999,
            epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
            name=f'{self._name}_project_bn_{unique_id}'
        )(x, training=training)
        
        # NO activation after projection - this is the "linear bottleneck"
        
        # 4. Residual connection (only when stride=1 and input/output channels match)
        if (self.stride == 1 and input_channels == self.output_channels):
            x = tf.keras.layers.Add(name=f'{self._name}_add_{unique_id}')([inputs, x])
        
        return x
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply MobileNetV2 inverted residual block.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        return self.build_component(inputs, training or False)
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape given input shape.
        
        Args:
            input_shape: Input tensor shape (height, width, channels)
            
        Returns:
            Output tensor shape (height/stride, width/stride, output_channels)
        """
        height, width, _ = input_shape
        
        # Calculate spatial dimensions after stride
        output_height = height // self.stride if height is not None else None
        output_width = width // self.stride if width is not None else None
        
        return (output_height, output_width, self.output_channels)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for this MobileNetV2 component.
        
        Returns:
            Dictionary with parameter count details
        """
        # For MobileNetV2 inverted residual blocks, parameters come from:
        # 1. Expansion conv (if expansion_ratio > 1): expansion_channels filters
        # 2. Depthwise conv: output_channels (3x3 kernels)
        # 3. Projection conv: output_channels filters
        # 4. Batch norm layers (4 parameters per channel for each BN)
        
        # This is a rough estimate - actual count depends on input channels
        # For a more accurate count, this would need to be calculated during model building
        
        # Base parameters from convolutions (rough estimate)
        base_params = self.output_channels * 40  # Rough estimate for MobileNetV2 block
        
        # Additional parameters for expansion if used
        if self.expansion_ratio > 1:
            base_params += self.output_channels * self.expansion_ratio * 5
        
        return {
            'total': base_params,
            'trainable': base_params,
            'non_trainable': 0
        }
    
    def get_config(self) -> Dict:
        """Get component configuration for serialization."""
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'expansion_ratio': self.expansion_ratio,
            'stride': self.stride
        })
        return config


def create_mobilenet_v2_inverted_block(output_channels: int,
                                     expansion_ratio: int = 6,
                                     stride: int = 1,
                                     config: Optional[ComponentConfig] = None) -> MobileNetV2InvertedResidualBlock:
    """Create a MobileNetV2 inverted residual block.
    
    Args:
        output_channels: Number of output channels
        expansion_ratio: Expansion ratio for bottleneck (typically 6)
        stride: Stride for depthwise convolution (1 or 2)
        config: Component configuration
        
    Returns:
        MobileNetV2InvertedResidualBlock instance
    """
    return MobileNetV2InvertedResidualBlock(
        config=config,
        output_channels=output_channels,
        expansion_ratio=expansion_ratio,
        stride=stride
    ) 