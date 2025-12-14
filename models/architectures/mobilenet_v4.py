"""
MobileNetV4-Conv Architecture Implementation

This module implements the MobileNetV4-Conv architectures as described in:
"MobileNetV4 -- Universal Models for the Mobile Ecosystem" (Qin et al., 2024)
https://arxiv.org/abs/2404.10518

The architecture uses Universal Inverted Bottleneck (UIB) blocks with configurable
depthwise convolutions to create efficient mobile-optimized networks. This implementation
focuses on the convolutional variants (Conv-S, Conv-M, Conv-L) without Mobile MQA.

Key features:
- Universal Inverted Bottleneck (UIB) blocks with 4 variants: IB, ConvNext, ExtraDW, FFN
- Hardware-optimized design for balanced MAC/memory bandwidth usage
- Pareto optimal across CPUs, GPUs, DSPs, and accelerators
- Improved ridge point optimization compared to V3
"""

import tensorflow as tf
from typing import Optional, List, Tuple, Dict, Any
from ..base import MobileNetArchitecture, ModelConfig
from ..components import (
    UniversalInvertedBottleneck,
    create_ib_block,
    create_convnext_block,
    create_extradw_block,
    create_ffn_block,
    StandardConvBlock,
    GlobalAveragePoolingBlock,
    DenseClassificationHead,
    apply_activation,
    ComponentConfig
)


class MobileNetV4ConvS(MobileNetArchitecture):
    """MobileNetV4-Conv-S architecture implementation.
    
    This implements the Small convolutional variant of MobileNetV4 with approximately
    3.8M parameters. The architecture uses UIB blocks in different configurations
    to optimize for mobile hardware while maintaining accuracy.
    
    Architecture stages:
    1. Initial conv: 224×224×3 -> 112×112×32
    2. Stage 1: UIB blocks with expansion, light computation
    3. Stage 2: More complex UIB blocks with SE attention
    4. Stage 3: Heavy computation stage with larger expansion
    5. Stage 4: Final feature extraction with max channels
    6. Final conv and classification head
    """
    
    # MobileNetV4-Conv-S layer specification
    # Format: (block_type, input_channels, expansion_ratio, output_channels, stride, kernel_size, se_ratio)
    # block_type: 'ib' = Inverted Bottleneck, 'cn' = ConvNext, 'ed' = ExtraDW, 'ffn' = FFN
    LAYER_SPECS = [
        # Stage 1: Initial UIB blocks (112×112 -> 56×56)
        ('cn',  32,  1,  32, 1, 3, None),    # ConvNext: spatial mixing first
        ('ib',  32,  2,  32, 1, 3, None),    # Standard IB
        ('ib',  32,  2,  64, 2, 3, None),    # Downsample to 56×56
        
        # Stage 2: Intermediate blocks (56×56 -> 28×28)  
        ('ib',  64,  4,  64, 1, 3, None),    # Standard expansion=4
        ('ib',  64,  4,  96, 1, 3, None),    # Channel expansion
        ('ib',  96,  4,  96, 1, 3, None),    # Maintain channels
        ('ib',  96,  4,  96, 2, 3, None),    # Downsample to 28×28
        
        # Stage 3: Heavy computation stage (28×28 -> 14×14)
        ('cn',  96,  4, 128, 1, 5, None),    # ConvNext with larger kernel
        ('ib', 128,  6, 128, 1, 3, None),    # Higher expansion ratio
        ('ib', 128,  6, 128, 1, 3, None),    # Maintain complexity
        ('ib', 128,  6, 128, 1, 3, None),    # Multiple blocks for depth
        ('ib', 128,  6, 128, 1, 3, None),    # Continue heavy computation
        ('ib', 128,  6, 128, 2, 3, None),    # Downsample to 14×14
        
        # Stage 4: Final feature extraction (14×14 -> 7×7)
        ('ed', 128,  6, 160, 1, 3, None),    # ExtraDW for rich features
        ('ib', 160,  6, 160, 1, 3, None),    # Standard processing
        ('ib', 160,  6, 160, 1, 3, None),    # Maintain channels
        ('cn', 160,  6, 192, 1, 5, None),    # ConvNext with larger kernel
        ('ib', 192,  6, 192, 1, 3, None),    # Final processing
        ('ib', 192,  6, 192, 2, 3, None),    # Downsample to 7×7
    ]
    
    def __init__(self, config: ModelConfig):
        """Initialize MobileNetV4-Conv-S architecture.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self._name = "mobilenet_v4_conv_s"
        
        # MobileNetV4 specific parameters
        self.width_multiplier = config.arch_params.get('width_multiplier', 1.0)
        self.final_conv_channels = self._make_divisible(960 * self.width_multiplier)  # Apply width multiplier
        self.dropout_rate = config.arch_params.get('dropout_rate', 0.2)
        
    @property
    def name(self) -> str:
        """Return the architecture name."""
        return self._name
    
    @property 
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        """Return list of supported input shapes for this architecture."""
        return [
            (96, 96, 1),     # Person detection optimized
            (96, 96, 3),     # Person detection RGB
            (128, 128, 1),   # Medium resolution grayscale
            (128, 128, 3),   # Medium resolution RGB
            (192, 192, 3),   # Smaller input for efficiency
            (224, 224, 1),   # Grayscale variant
            (224, 224, 3),   # Standard ImageNet input
            (256, 256, 1),   # High resolution grayscale
            (256, 256, 3),   # High resolution RGB
        ]
    
    @property
    def parameter_count_range(self) -> Tuple[int, int]:
        """Return expected parameter count range (min, max) for this architecture."""
        # Base parameter count for width_multiplier=1.0: ~3.8M parameters
        base_params = 3_800_000
        
        # Scale by width multiplier squared (roughly, since both input and output channels scale)
        scaled_params = int(base_params * (self.width_multiplier ** 2))
        
        # Apply tolerance of ±20%
        tolerance = 0.2
        
        return (
            int(scaled_params * (1 - tolerance)),
            int(scaled_params * (1 + tolerance))
        )
    
    def validate_config(self) -> None:
        """Validate architecture-specific configuration parameters."""
        # Validate input shape
        if self.config.input_shape not in self.supported_input_shapes:
            raise ValueError(
                f"Input shape {self.config.input_shape} not supported. "
                f"Supported shapes: {self.supported_input_shapes}"
            )
        
        # Validate width multiplier
        width_multiplier = self.config.arch_params.get('width_multiplier', 1.0)
        if width_multiplier <= 0 or width_multiplier > 2.0:
            raise ValueError(f"Width multiplier must be between 0.0 and 2.0, got {width_multiplier}")
        
        # Validate dropout rate
        dropout = self.config.arch_params.get('dropout_rate', 0.2)
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"Dropout rate must be between 0.0 and 1.0, got {dropout}")
        
        # Validate architecture type if specified
        arch_type = self.config.arch_params.get('architecture_type', 'mobilenet_v4_conv_s')
        if arch_type not in ['mobilenet_v4_conv_s', 'mobilenet_v4']:
            raise ValueError(
                f"Architecture type '{arch_type}' not supported by MobileNetV4ConvS. "
                f"Expected: 'mobilenet_v4_conv_s' or 'mobilenet_v4'"
            )
    
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
    
    def _create_uib_block(self, 
                         block_type: str,
                         output_channels: int,
                         expansion_ratio: int, 
                         stride: int,
                         kernel_size: int,
                         se_ratio: Optional[float] = None) -> UniversalInvertedBottleneck:
        """Create UIB block based on type specification.
        
        Args:
            block_type: Type of UIB block ('ib', 'cn', 'ed', 'ffn')
            output_channels: Number of output channels (before width multiplier)
            expansion_ratio: Expansion ratio for bottleneck
            stride: Stride for convolutions
            kernel_size: Kernel size for depthwise convolutions
            se_ratio: SE ratio (not used in Conv-S variant)
            
        Returns:
            Configured UIB block
        """
        # Apply width multiplier to output channels
        scaled_output_channels = self._make_divisible(output_channels * self.width_multiplier)
        
        # Common configuration
        config = ComponentConfig(
            activation='relu6',  # MobileNetV4 primarily uses ReLU6
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_epsilon=self.config.batch_norm_epsilon,
            use_batch_norm=True,
            use_bias=False
        )
        
        if block_type == 'ib':
            # Inverted Bottleneck: standard MobileNetV2-style
            return create_ib_block(
                output_channels=scaled_output_channels,
                expansion_ratio=expansion_ratio,
                stride=stride,
                kernel_size=kernel_size,
                activation='relu6',
                config=config
            )
        elif block_type == 'cn':
            # ConvNext: spatial mixing before expansion
            return create_convnext_block(
                output_channels=scaled_output_channels,
                expansion_ratio=expansion_ratio,
                stride=stride,
                kernel_size=kernel_size,
                activation='relu6',
                config=config
            )
        elif block_type == 'ed':
            # ExtraDW: both start and middle DW for extra depth
            return create_extradw_block(
                output_channels=scaled_output_channels,
                expansion_ratio=expansion_ratio,
                stride=stride,
                start_kernel_size=kernel_size,
                middle_kernel_size=kernel_size,
                activation='relu6',
                config=config
            )
        elif block_type == 'ffn':
            # FFN: pure pointwise convolutions
            return create_ffn_block(
                output_channels=scaled_output_channels,
                expansion_ratio=expansion_ratio,
                stride=stride,
                activation='relu6',
                config=config
            )
        else:
            raise ValueError(f"Unknown UIB block type: {block_type}")
    
    def build_model(self, training: bool = False) -> tf.keras.Model:
        """Build the complete MobileNetV4-Conv-S model.
        
        Args:
            training: Whether in training mode
            
        Returns:
            Complete TensorFlow model
        """
        inputs = tf.keras.layers.Input(
            shape=self.config.input_shape,
            name='input_image'
        )
        
        # Initial convolution: 224×224×3 -> 112×112×32
        initial_conv_config = ComponentConfig(
            activation='relu6',
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_epsilon=self.config.batch_norm_epsilon,
            use_batch_norm=True,
            use_bias=False
        )
        
        # Apply width multiplier to initial conv channels
        initial_channels = self._make_divisible(32 * self.width_multiplier)
        
        x = StandardConvBlock(
            config=initial_conv_config,
            filters=initial_channels,
            kernel_size=3,
            strides=2
        ).call(inputs, training)
        
        # Build UIB blocks according to specification
        for i, (block_type, input_ch, exp_ratio, output_ch, stride, kernel, se_ratio) in enumerate(self.LAYER_SPECS):
            uib_block = self._create_uib_block(
                block_type=block_type,
                output_channels=output_ch,
                expansion_ratio=exp_ratio,
                stride=stride,
                kernel_size=kernel,
                se_ratio=se_ratio
            )
            
            x = uib_block.call(x, training)
        
        # Final convolution before classification
        final_conv_config = ComponentConfig(
            activation='relu6',
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_epsilon=self.config.batch_norm_epsilon,
            use_batch_norm=True,
            use_bias=False
        )
        
        x = StandardConvBlock(
            config=final_conv_config,
            filters=self.final_conv_channels,
            kernel_size=1,
            strides=1
        ).call(x, training)
        
        # Global average pooling
        pooling_config = ComponentConfig()
        x = GlobalAveragePoolingBlock(config=pooling_config).call(x, training)
        
        # Classification head with dropout
        if self.dropout_rate > 0:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')(x, training=training)
        
        head_config = ComponentConfig()
        outputs = DenseClassificationHead(
            num_classes=self.config.num_classes,
            config=head_config
        ).call(x, training)
        
        # Create model
        model = tf.keras.Model(inputs, outputs, name=self.name)
        
        return model
    
    def get_model_summary(self) -> dict:
        """Get detailed model summary information.
        
        Returns:
            Dictionary with model architecture details
        """
        return {
            'architecture': 'MobileNetV4-Conv-S',
            'family': 'MobileNetV4',
            'variant': f'Conv-S (Convolutional Small, α={self.width_multiplier})',
            'width_multiplier': self.width_multiplier,
            'total_stages': 4,
            'uib_blocks': len(self.LAYER_SPECS),
            'input_shape': self.config.input_shape,
            'num_classes': self.config.num_classes,
            'final_conv_channels': self.final_conv_channels,
            'dropout_rate': self.dropout_rate,
            'expected_params': f'~{3.8 * (self.width_multiplier ** 2):.1f}M',
            'block_types': {
                'ib': sum(1 for spec in self.LAYER_SPECS if spec[0] == 'ib'),
                'cn': sum(1 for spec in self.LAYER_SPECS if spec[0] == 'cn'),
                'ed': sum(1 for spec in self.LAYER_SPECS if spec[0] == 'ed'),
                'ffn': sum(1 for spec in self.LAYER_SPECS if spec[0] == 'ffn'),
            },
            'optimizations': [
                'Universal Inverted Bottleneck blocks',
                'Hardware-optimized design',
                'Balanced MAC/memory bandwidth',
                'Ridge point optimization'
            ]
        }
    
    def get_config(self) -> dict:
        """Get architecture configuration dictionary."""
        return {
            'architecture_name': self.name,
            'architecture_type': 'mobilenet_v4_conv_s',
            'input_shape': self.config.input_shape,
            'num_classes': self.config.num_classes,
            'final_conv_channels': self.final_conv_channels,
            'dropout_rate': self.dropout_rate,
            'layer_specs': self.LAYER_SPECS,
            'supported_input_shapes': self.supported_input_shapes,
            'parameter_count_range': self.parameter_count_range
        }


def create_mobilenet_v4_conv_s_configs() -> List[Tuple[str, dict]]:
    """Create configurations for all MobileNetV4-Conv-S width multiplier variants.
    
    Returns:
        List of (architecture_name, config_dict) tuples for α = 0.25, 0.5, 0.75, 1.0
    """
    configs = []
    
    # Width multiplier variants
    width_multipliers = [0.25, 0.5, 0.75, 1.0]
    
    for alpha in width_multipliers:
        # Format alpha for naming (0.25 -> 0_25)
        alpha_str = str(alpha).replace('.', '_')
        arch_name = f'mobilenet_v4_conv_s_{alpha_str}'
        
        config_dict = {
            'architecture_type': 'mobilenet_v4_conv_s',
            'input_shape': (224, 224, 3),
            'num_classes': 1000,  # Default ImageNet classes
            'arch_params': {
                'width_multiplier': alpha,
                'dropout_rate': 0.2,
                'architecture_type': 'mobilenet_v4_conv_s'
            },
            'optimization_params': {
                'use_mixed_precision': True,
                'gradient_clipping': True,
                'use_ema': True
            }
        }
        
        configs.append((arch_name, config_dict))
    
    return configs


# Backward compatibility: keep the original function for single variant
def create_mobilenet_v4_conv_s_config() -> Tuple[str, dict]:
    """Create default configuration for MobileNetV4-Conv-S (width_multiplier=1.0).
    
    Returns:
        Tuple of (architecture_name, config_dict)
    """
    configs = create_mobilenet_v4_conv_s_configs()
    # Return the 1.0 variant (last in list)
    return configs[-1] 