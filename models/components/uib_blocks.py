"""
Universal Inverted Bottleneck (UIB) Blocks for MobileNetV4

This module implements the Universal Inverted Bottleneck blocks as introduced
in MobileNetV4. UIB extends the traditional inverted bottleneck by making
depthwise convolutions configurable at different positions, enabling 4 distinct
variants: Inverted Bottleneck (IB), ConvNext, ExtraDW, and FFN.

The UIB block structure:
1. Optional start depthwise convolution (ConvNext, ExtraDW variants)
2. 1x1 expansion convolution
3. Optional middle depthwise convolution (IB, ExtraDW variants)
4. Optional end depthwise convolution (not used in current variants)
5. 1x1 projection convolution
6. Residual connection (if input/output shapes match)

This provides maximum flexibility for Neural Architecture Search (NAS) while
maintaining efficiency and hardware compatibility.
"""

import tensorflow as tf
from typing import Optional, Union, Tuple, Dict
from .base import MobileNetComponent, ComponentConfig
from .activations import apply_activation

# Global counter for unique UIB block naming
_uib_block_counter = 0

def _get_unique_uib_id() -> int:
    """Get unique identifier for UIB block layer naming."""
    global _uib_block_counter
    _uib_block_counter += 1
    return _uib_block_counter

class UniversalInvertedBottleneck(MobileNetComponent):
    """Universal Inverted Bottleneck Block for MobileNetV4.
    
    A unified and flexible structure that can represent multiple block types:
    - IB (Inverted Bottleneck): Standard MobileNetV2-style with middle DW
    - ConvNext: Start DW + no middle DW, spatial mixing before expansion
    - ExtraDW: Both start DW and middle DW for increased depth/receptive field
    - FFN: No DW convolutions, pure feed-forward network
    
    The block allows configurable depthwise convolutions at three positions:
    - start_dw: Before expansion (ConvNext, ExtraDW)
    - middle_dw: After expansion (IB, ExtraDW)
    - end_dw: Before projection (not used in current variants)
    """
    
    def __init__(self,
                 config: Optional[ComponentConfig] = None,
                 output_channels: int = 32,
                 expansion_ratio: int = 4,
                 stride: int = 1,
                 start_dw_kernel_size: Optional[int] = None,
                 middle_dw_kernel_size: Optional[int] = None,
                 end_dw_kernel_size: Optional[int] = None,
                 activation: str = 'relu6',
                 use_bias: bool = False):
        """Initialize Universal Inverted Bottleneck block.
        
        Args:
            config: Component configuration (if None, uses defaults)
            output_channels: Number of output channels
            expansion_ratio: Expansion ratio for bottleneck (typically 4 for V4)
            stride: Stride for depthwise convolutions
            start_dw_kernel_size: Kernel size for start DW (None = no start DW)
            middle_dw_kernel_size: Kernel size for middle DW (None = no middle DW)
            end_dw_kernel_size: Kernel size for end DW (None = no end DW)
            activation: Activation function ('relu', 'hard_swish', etc.)
            use_bias: Whether to use bias in convolutions
        """
        # Create default config if none provided
        if config is None:
            config = ComponentConfig(
                use_bias=use_bias,
                activation=activation,
                batch_norm_momentum=0.999,
                batch_norm_epsilon=1e-3,
                use_batch_norm=True
            )
        
        super().__init__(config)
        self.output_channels = output_channels
        self.expansion_ratio = expansion_ratio
        self.stride = stride
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size
        self.end_dw_kernel_size = end_dw_kernel_size
        self.activation = activation
        self.use_bias = use_bias
        
        # Determine block variant for naming
        variant = self._determine_variant()
        self._name = f"uib_{variant}_exp{expansion_ratio}_out{output_channels}_s{stride}"
    
    def _determine_variant(self) -> str:
        """Determine UIB variant based on DW configuration."""
        has_start = self.start_dw_kernel_size is not None
        has_middle = self.middle_dw_kernel_size is not None
        has_end = self.end_dw_kernel_size is not None
        
        if has_start and has_middle:
            return "extradw"  # ExtraDW: both start and middle DW
        elif has_start and not has_middle:
            return "convnext"  # ConvNext: only start DW
        elif not has_start and has_middle:
            return "ib"  # Inverted Bottleneck: only middle DW
        elif not has_start and not has_middle and not has_end:
            return "ffn"  # FFN: no DW convolutions
        else:
            return "custom"  # Custom configuration
    
    @property
    def name(self) -> str:
        """Return component name."""
        return self._name
    
    def build_component(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Build Universal Inverted Bottleneck block.
        
        Args:
            inputs: Input tensor [batch, height, width, input_channels]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch, height/stride, width/stride, output_channels]
        """
        input_channels = inputs.shape[-1]
        expanded_channels = input_channels * self.expansion_ratio
        
        # Get unique identifier for this block instance
        unique_id = _get_unique_uib_id()
        
        x = inputs
        
        # 1. Optional start depthwise convolution (ConvNext, ExtraDW variants)
        if self.start_dw_kernel_size is not None:
            x = tf.keras.layers.DepthwiseConv2D(
                kernel_size=self.start_dw_kernel_size,
                strides=1,  # Start DW typically doesn't change spatial dimensions
                padding='same',
                use_bias=self.use_bias,
                depthwise_initializer='he_normal',
                name=f'{self._name}_start_dw_{unique_id}'
            )(x)
            
            x = tf.keras.layers.BatchNormalization(
                momentum=self.config.batch_norm_momentum if self.config else 0.999,
                epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
                name=f'{self._name}_start_dw_bn_{unique_id}'
            )(x, training=training)
            
            x = apply_activation(x, self.activation, training)
        
        # 2. Expansion phase (1x1 conv) - always present unless expansion_ratio == 1
        if self.expansion_ratio > 1:
            x = tf.keras.layers.Conv2D(
                expanded_channels,
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=self.use_bias,
                kernel_initializer='he_normal',
                name=f'{self._name}_expand_{unique_id}'
            )(x)
            
            x = tf.keras.layers.BatchNormalization(
                momentum=self.config.batch_norm_momentum if self.config else 0.999,
                epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
                name=f'{self._name}_expand_bn_{unique_id}'
            )(x, training=training)
            
            x = apply_activation(x, self.activation, training)
        
        # 3. Optional middle depthwise convolution (IB, ExtraDW variants)
        if self.middle_dw_kernel_size is not None:
            x = tf.keras.layers.DepthwiseConv2D(
                kernel_size=self.middle_dw_kernel_size,
                strides=self.stride,  # Middle DW applies the stride
                padding='same',
                use_bias=self.use_bias,
                depthwise_initializer='he_normal',
                name=f'{self._name}_middle_dw_{unique_id}'
            )(x)
            
            x = tf.keras.layers.BatchNormalization(
                momentum=self.config.batch_norm_momentum if self.config else 0.999,
                epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
                name=f'{self._name}_middle_dw_bn_{unique_id}'
            )(x, training=training)
            
            x = apply_activation(x, self.activation, training)
        
        # 4. Optional end depthwise convolution (not used in current variants)
        if self.end_dw_kernel_size is not None:
            x = tf.keras.layers.DepthwiseConv2D(
                kernel_size=self.end_dw_kernel_size,
                strides=1,  # End DW typically doesn't change spatial dimensions
                padding='same',
                use_bias=self.use_bias,
                depthwise_initializer='he_normal',
                name=f'{self._name}_end_dw_{unique_id}'
            )(x)
            
            x = tf.keras.layers.BatchNormalization(
                momentum=self.config.batch_norm_momentum if self.config else 0.999,
                epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
                name=f'{self._name}_end_dw_bn_{unique_id}'
            )(x, training=training)
            
            x = apply_activation(x, self.activation, training)
        
        # 5. Projection phase (1x1 conv) - always present
        x = tf.keras.layers.Conv2D(
            self.output_channels,
            kernel_size=1,
            strides=1 if self.middle_dw_kernel_size is not None else self.stride,  # Apply stride if no middle DW
            padding='same',
            use_bias=self.use_bias,
            kernel_initializer='he_normal',
            name=f'{self._name}_project_{unique_id}'
        )(x)
        
        x = tf.keras.layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum if self.config else 0.999,
            epsilon=self.config.batch_norm_epsilon if self.config else 1e-3,
            name=f'{self._name}_project_bn_{unique_id}'
        )(x, training=training)
        
        # No activation after projection (linear output)
        
        # 6. Residual connection (if input/output shapes match)
        if (self.stride == 1 and 
            input_channels == self.output_channels and 
            inputs.shape[1:] == x.shape[1:]):
            x = tf.keras.layers.Add(name=f'{self._name}_add_{unique_id}')([inputs, x])
        
        return x
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply UIB block (MobileNetComponent interface).
        
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
        
        # Calculate output spatial dimensions based on stride
        out_height = (height + self.stride - 1) // self.stride
        out_width = (width + self.stride - 1) // self.stride
        
        return (out_height, out_width, self.output_channels)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for this component.
        
        Returns:
            Dictionary with parameter counts for each layer type
        """
        params = {}
        input_channels = None  # Will be determined at build time
        expanded_channels = input_channels * self.expansion_ratio if input_channels else 0
        
        # Start DW parameters
        if self.start_dw_kernel_size is not None:
            start_dw_params = self.start_dw_kernel_size * self.start_dw_kernel_size * input_channels
            params['start_dw'] = start_dw_params
        
        # Expansion parameters
        if self.expansion_ratio > 1:
            expand_params = input_channels * expanded_channels  # 1x1 conv
            params['expansion'] = expand_params
        
        # Middle DW parameters
        if self.middle_dw_kernel_size is not None:
            middle_dw_params = self.middle_dw_kernel_size * self.middle_dw_kernel_size * expanded_channels
            params['middle_dw'] = middle_dw_params
        
        # End DW parameters
        if self.end_dw_kernel_size is not None:
            end_dw_params = self.end_dw_kernel_size * self.end_dw_kernel_size * expanded_channels
            params['end_dw'] = end_dw_params
        
        # Projection parameters
        projection_channels = expanded_channels if self.expansion_ratio > 1 else input_channels
        params['projection'] = projection_channels * self.output_channels  # 1x1 conv
        
        return params
    
    def get_config(self) -> dict:
        """Get configuration dictionary for serialization."""
        return {
            'output_channels': self.output_channels,
            'expansion_ratio': self.expansion_ratio,
            'stride': self.stride,
            'start_dw_kernel_size': self.start_dw_kernel_size,
            'middle_dw_kernel_size': self.middle_dw_kernel_size,
            'end_dw_kernel_size': self.end_dw_kernel_size,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'config': self.config.to_dict() if self.config else None
        }

# Convenience factory functions for common UIB variants

def create_ib_block(output_channels: int,
                   expansion_ratio: int = 4,
                   stride: int = 1,
                   kernel_size: int = 3,
                   activation: str = 'relu6',
                   config: Optional[ComponentConfig] = None) -> UniversalInvertedBottleneck:
    """Create Inverted Bottleneck (IB) variant of UIB.
    
    Traditional MobileNetV2-style inverted residual with only middle DW.
    
    Args:
        output_channels: Number of output channels
        expansion_ratio: Expansion ratio (typically 4 for V4)
        stride: Stride for depthwise convolution
        kernel_size: Kernel size for depthwise convolution
        activation: Activation function
        config: Component configuration
        
    Returns:
        UniversalInvertedBottleneck configured as IB variant
    """
    return UniversalInvertedBottleneck(
        config=config,
        output_channels=output_channels,
        expansion_ratio=expansion_ratio,
        stride=stride,
        start_dw_kernel_size=None,
        middle_dw_kernel_size=kernel_size,
        end_dw_kernel_size=None,
        activation=activation
    )

def create_convnext_block(output_channels: int,
                         expansion_ratio: int = 4,
                         stride: int = 1,
                         kernel_size: int = 3,
                         activation: str = 'relu6',
                         config: Optional[ComponentConfig] = None) -> UniversalInvertedBottleneck:
    """Create ConvNext variant of UIB.
    
    Spatial mixing before expansion with larger kernel size, no middle DW.
    
    Args:
        output_channels: Number of output channels
        expansion_ratio: Expansion ratio (typically 4 for V4)
        stride: Stride for depthwise convolution
        kernel_size: Kernel size for depthwise convolution (3x3 or 5x5)
        activation: Activation function
        config: Component configuration
        
    Returns:
        UniversalInvertedBottleneck configured as ConvNext variant
    """
    return UniversalInvertedBottleneck(
        config=config,
        output_channels=output_channels,
        expansion_ratio=expansion_ratio,
        stride=stride,
        start_dw_kernel_size=kernel_size,
        middle_dw_kernel_size=None,
        end_dw_kernel_size=None,
        activation=activation
    )

def create_extradw_block(output_channels: int,
                        expansion_ratio: int = 4,
                        stride: int = 1,
                        start_kernel_size: int = 3,
                        middle_kernel_size: int = 3,
                        activation: str = 'relu6',
                        config: Optional[ComponentConfig] = None) -> UniversalInvertedBottleneck:
    """Create ExtraDW variant of UIB.
    
    Novel variant with both start and middle DW for increased depth and receptive field.
    Combines benefits of ConvNext and IB variants.
    
    Args:
        output_channels: Number of output channels
        expansion_ratio: Expansion ratio (typically 4 for V4)
        stride: Stride for depthwise convolution
        start_kernel_size: Kernel size for start depthwise convolution
        middle_kernel_size: Kernel size for middle depthwise convolution
        activation: Activation function
        config: Component configuration
        
    Returns:
        UniversalInvertedBottleneck configured as ExtraDW variant
    """
    return UniversalInvertedBottleneck(
        config=config,
        output_channels=output_channels,
        expansion_ratio=expansion_ratio,
        stride=stride,
        start_dw_kernel_size=start_kernel_size,
        middle_dw_kernel_size=middle_kernel_size,
        end_dw_kernel_size=None,
        activation=activation
    )

def create_ffn_block(output_channels: int,
                    expansion_ratio: int = 4,
                    stride: int = 1,
                    activation: str = 'relu6',
                    config: Optional[ComponentConfig] = None) -> UniversalInvertedBottleneck:
    """Create FFN (Feed Forward Network) variant of UIB.
    
    Pure pointwise convolutions with no depthwise convolutions.
    Most accelerator-friendly but works best combined with other block types.
    
    Args:
        output_channels: Number of output channels
        expansion_ratio: Expansion ratio (typically 4 for V4)
        stride: Stride (handled by projection layer since no DW)
        activation: Activation function
        config: Component configuration
        
    Returns:
        UniversalInvertedBottleneck configured as FFN variant
    """
    return UniversalInvertedBottleneck(
        config=config,
        output_channels=output_channels,
        expansion_ratio=expansion_ratio,
        stride=stride,
        start_dw_kernel_size=None,
        middle_dw_kernel_size=None,
        end_dw_kernel_size=None,
        activation=activation
    ) 