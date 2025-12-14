"""
MobileNetV2 architecture implementation.

This module implements the MobileNetV2 architecture with inverted residuals
and linear bottlenecks. MobileNetV2 introduces the concept of inverted
residuals where the shortcut connections are between thin bottleneck layers.

Key features:
- Inverted residual blocks with linear bottlenecks
- ReLU6 activations throughout
- Width multiplier support (α = 0.25, 0.5, 0.75, 1.0)
- Configurable input resolutions
- Optimized for mobile and embedded deployment

Reference: 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
https://arxiv.org/abs/1801.04381
"""

from typing import Dict, Any, Tuple, List, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model

from ..base import MobileNetArchitecture, ModelConfig
from ..factory import ModelArchitectureFactory
from ..components import (
    MobileNetV2InvertedResidualBlock,
    create_mobilenet_v2_inverted_block,
    ComponentConfig,
    MobileNetClassificationHead
)


class MobileNetV2Architecture(MobileNetArchitecture):
    """MobileNetV2 architecture implementation.
    
    This class implements MobileNetV2 with configurable width multiplier,
    supporting standard MobileNetV2 configurations optimized for mobile
    and embedded deployment scenarios.
    
    The architecture follows the standard MobileNetV2 design:
    1. Initial 3x3 convolution with stride 2
    2. 17 inverted residual blocks with varying expansion ratios
    3. Final 1x1 convolution to 1280 channels
    4. Global average pooling and classification head
    
    Supports width multipliers: 0.25, 0.5, 0.75, 1.0
    """
    
    @property
    def name(self) -> str:
        """Return the architecture name."""
        width_mult = self.config.arch_params.get('width_multiplier', 1.0)
        return f"mobilenet_v2_{str(width_mult).replace('.', '_')}"
    
    @property
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        """Return list of supported input shapes for MobileNetV2."""
        return [
            (96, 96, 1),     # Person detection optimized
            (96, 96, 3),     # Person detection RGB
            (128, 128, 1),   # Medium resolution grayscale
            (128, 128, 3),   # Medium resolution RGB
            (160, 160, 1),   # Higher resolution grayscale
            (160, 160, 3),   # Higher resolution RGB
            (192, 192, 3),   # Efficient high-res
            (224, 224, 1),   # Standard resolution grayscale
            (224, 224, 3),   # Standard ImageNet input
            (256, 256, 1),   # High resolution grayscale
            (256, 256, 3),   # High resolution RGB
        ]
    
    @property
    def parameter_count_range(self) -> Tuple[int, int]:
        """Return expected parameter count range for MobileNetV2."""
        width_mult = self.config.arch_params.get('width_multiplier', 1.0)
        expansion_factor = self.config.arch_params.get('expansion_factor', 6)
        
        # Calculate theoretical parameter count
        theoretical_params = self.calculate_theoretical_parameters(width_mult, expansion_factor)
        
        # Allow 15% tolerance for custom configurations
        min_params = int(theoretical_params * 0.85)
        max_params = int(theoretical_params * 1.15)
        
        return (min_params, max_params)
    
    def calculate_theoretical_parameters(self, width_mult: float, expansion_factor: int) -> int:
        """Calculate theoretical parameter count for given configuration.
        
        Args:
            width_mult: Width multiplier
            expansion_factor: Expansion factor for inverted residuals
            
        Returns:
            Estimated parameter count
        """
        # Base calculations for standard MobileNetV2 structure
        
        # Initial conv: 3x3x3x(32*width_mult) + BN
        initial_conv = 3 * 3 * 3 * self._make_divisible(32 * width_mult)
        
        # Inverted residual blocks parameters
        block_params = 0
        input_channels = self._make_divisible(32 * width_mult)
        
        # Block configurations (same as in build_architecture)
        block_configs = [
            (1, 16, 1, 1),   # Block 1 - always expansion=1
            (expansion_factor, 24, 2, 2),
            (expansion_factor, 32, 3, 2),
            (expansion_factor, 64, 4, 2),
            (expansion_factor, 96, 3, 1),
            (expansion_factor, 160, 3, 2),
            (expansion_factor, 320, 1, 1),
        ]
        
        for exp_ratio, output_ch, num_blocks, stride in block_configs:
            output_channels = self._make_divisible(output_ch * width_mult)
            
            for _ in range(num_blocks):
                # Expansion conv (if exp_ratio > 1): 1x1 conv
                if exp_ratio > 1:
                    expanded_ch = input_channels * exp_ratio
                    block_params += input_channels * expanded_ch
                    
                    # Depthwise conv: kernel_size^2 * expanded_channels
                    block_params += 3 * 3 * expanded_ch
                    
                    # Projection conv: 1x1 conv
                    block_params += expanded_ch * output_channels
                else:
                    # First block with no expansion - direct depthwise + projection
                    block_params += 3 * 3 * input_channels  # Depthwise
                    block_params += input_channels * output_channels  # Projection
                
                input_channels = output_channels
        
        # Final conv: 1x1x(320*width_mult)x1280
        final_conv = self._make_divisible(320 * width_mult) * 1280
        
        # Classification head (if num_classes set)
        classifier = 1280 * self.config.num_classes
        
        total_params = initial_conv + block_params + final_conv + classifier
        
        return int(total_params)
    
    def validate_config(self) -> None:
        """Validate MobileNetV2-specific configuration parameters."""
        # Validate width multiplier (now supports custom values)
        width_mult = self.config.arch_params.get('width_multiplier', 1.0)
        
        # Allow custom width multipliers with reasonable bounds
        if not isinstance(width_mult, (int, float)) or width_mult <= 0:
            raise ValueError(
                f"width_multiplier must be a positive number, got {width_mult}"
            )
        
        if width_mult < 0.1 or width_mult > 2.0:
            raise ValueError(
                f"width_multiplier must be between 0.1 and 2.0 for practical use, "
                f"got {width_mult}"
            )
            
        # Validate expansion factor (new configuration option)
        expansion_factor = self.config.arch_params.get('expansion_factor', 6)
        
        if not isinstance(expansion_factor, int) or expansion_factor < 1:
            raise ValueError(
                f"expansion_factor must be a positive integer, got {expansion_factor}"
            )
            
        if expansion_factor > 12:
            raise ValueError(
                f"expansion_factor must be ≤ 12 for practical use, got {expansion_factor}"
            )
        
        # Validate input shape (enhanced for non-square support)
        self.validate_input_shape(self.config.input_shape)
        
        # Validate minimum image size for MobileNetV2 (based on architecture depth)
        height, width, channels = self.config.input_shape
        min_size = 32  # Minimum size due to 5 stride-2 operations (224→7)
        
        if height < min_size or width < min_size:
            raise ValueError(
                f"MobileNetV2 requires minimum input size of {min_size}x{min_size} "
                f"due to architectural constraints, got {height}x{width}"
            )
            
        # Support for non-square input shapes, but warn about potential issues
        if height != width:
            import warnings
            warnings.warn(
                f"Non-square input shapes ({height}x{width}) may impact performance. "
                f"Square inputs are recommended for optimal efficiency.",
                UserWarning
            )
        
        # Validate channels
        if channels not in [1, 3]:
            raise ValueError(
                f"MobileNetV2 supports 1 (grayscale) or 3 (RGB) channels, "
                f"got {channels}"
            )
    
    def _make_divisible(self, value: int, divisor: int = 8) -> int:
        """Make value divisible by divisor.
        
        This ensures the number of channels is divisible by 8,
        which is optimal for many hardware accelerators.
        
        Args:
            value: Original value
            divisor: Divisor (typically 8)
            
        Returns:
            Value divisible by divisor
        """
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value
    
    def build_architecture(self) -> Model:
        """Build the complete MobileNetV2 architecture.
        
        Returns:
            Compiled Keras model
        """
        self.validate_config()
        
        # Configuration
        width_mult = self.config.arch_params.get('width_multiplier', 1.0)
        expansion_factor = self.config.arch_params.get('expansion_factor', 6)
        dropout_rate = self.config.dropout_rate
        
        # Component configuration
        component_config = ComponentConfig(
            use_bias=False,
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_epsilon=self.config.batch_norm_epsilon,
            activation='relu6'
        )
        
        # Input layer
        inputs = tf.keras.Input(
            shape=self.config.input_shape,
            name='mobilenet_v2_input'
        )
        
        x = inputs
        
        # 1. Initial convolution layer
        initial_channels = self._make_divisible(32 * width_mult)
        x = tf.keras.layers.Conv2D(
            initial_channels,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name='initial_conv'
        )(x)
        
        x = tf.keras.layers.BatchNormalization(
            momentum=component_config.batch_norm_momentum,
            epsilon=component_config.batch_norm_epsilon,
            name='initial_bn'
        )(x)
        
        x = tf.keras.layers.ReLU(max_value=6.0, name='initial_relu')(x)
        
        # 2. Inverted residual blocks
        # MobileNetV2 standard configuration with configurable expansion
        block_configs = [
            # (expansion_ratio, output_channels, num_blocks, stride)
            (1, 16, 1, 1),   # Block 1 - always uses expansion=1 (no expansion)
            (expansion_factor, 24, 2, 2),   # Block 2-3
            (expansion_factor, 32, 3, 2),   # Block 4-6
            (expansion_factor, 64, 4, 2),   # Block 7-10
            (expansion_factor, 96, 3, 1),   # Block 11-13
            (expansion_factor, 160, 3, 2),  # Block 14-16
            (expansion_factor, 320, 1, 1),  # Block 17
        ]
        
        block_id = 1
        for expansion_ratio, output_channels, num_blocks, stride in block_configs:
            # Apply width multiplier to output channels
            output_channels = self._make_divisible(output_channels * width_mult)
            
            for i in range(num_blocks):
                # First block in each stage uses the specified stride
                block_stride = stride if i == 0 else 1
                
                # Create inverted residual block
                inv_block = create_mobilenet_v2_inverted_block(
                    output_channels=output_channels,
                    expansion_ratio=expansion_ratio,
                    stride=block_stride,
                    config=component_config
                )
                
                x = inv_block.build_component(x, training=False)
                block_id += 1
        
        # 3. Final convolution layer
        final_channels = self._make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280
        x = tf.keras.layers.Conv2D(
            final_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name='final_conv'
        )(x)
        
        x = tf.keras.layers.BatchNormalization(
            momentum=component_config.batch_norm_momentum,
            epsilon=component_config.batch_norm_epsilon,
            name='final_bn'
        )(x)
        
        x = tf.keras.layers.ReLU(max_value=6.0, name='final_relu')(x)
        
        # 4. Classification head
        classification_head = MobileNetClassificationHead(
            num_classes=self.config.num_classes,
            dropout_rate=dropout_rate,
            config=component_config
        )
        
        outputs = classification_head.call(x, training=False)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        return model
    
    def build_model(self) -> tf.keras.Model:
        """Build and return the TensorFlow model.
        
        This method implements the abstract method from MobileNetArchitecture.
        
        Returns:
            Compiled TensorFlow Keras model
        """
        return self.build_architecture()
    
    def get_flops_estimate(self) -> int:
        """Estimate FLOPs for the architecture.
        
        Returns:
            Estimated FLOPs count
        """
        width_mult = self.config.arch_params.get('width_multiplier', 1.0)
        height, width, channels = self.config.input_shape
        
        # Base FLOPs for MobileNetV2-1.0 with 224x224 input is ~300M
        base_flops = 300_000_000
        
        # Scale by width multiplier (roughly quadratic)
        flops = base_flops * (width_mult ** 2)
        
        # Scale by input resolution (quadratic)
        resolution_scale = (height * width) / (224 * 224)
        flops *= resolution_scale
        
        return int(flops)


def create_mobilenet_v2_configs() -> List[Tuple[str, dict]]:
    """Create configurations for all MobileNetV2 width multiplier variants.
    
    Returns:
        List of (architecture_name, config_dict) tuples for α = 0.25, 0.5, 0.75, 1.0
    """
    configs = []
    
    # Standard width multiplier variants
    width_multipliers = [0.25, 0.5, 0.75, 1.0]
    
    for alpha in width_multipliers:
        # Format alpha for naming (0.25 -> 0_25)
        alpha_str = str(alpha).replace('.', '_')
        arch_name = f'mobilenet_v2_{alpha_str}'
        
        config_dict = {
            'architecture_type': 'mobilenet_v2',
            'input_shape': (224, 224, 3),  # Default ImageNet size
            'num_classes': 1000,  # Default ImageNet classes
            'arch_params': {
                'width_multiplier': alpha,
                'expansion_factor': 6,  # Standard MobileNetV2 expansion
                'architecture_type': 'mobilenet_v2'
            },
            'optimization_params': {
                'use_mixed_precision': False,  # Keep simple for MobileNetV2
                'gradient_clipping': True,
                'use_ema': False
            }
        }
        
        configs.append((arch_name, config_dict))
    
    return configs


def create_mobilenet_v2_variant_configs() -> List[Tuple[str, dict]]:
    """Create configurations for MobileNetV2 architectural variants.
    
    Returns:
        List of (architecture_name, config_dict) tuples for Lite, Plus, and person detection variants
    """
    configs = []
    
    # MobileNetV2-Lite: Reduced expansion for ultra-efficiency
    lite_config = {
        'architecture_type': 'mobilenet_v2',
        'input_shape': (224, 224, 3),
        'num_classes': 1000,
        'arch_params': {
            'width_multiplier': 1.0,
            'expansion_factor': 3,  # Reduced expansion for efficiency
            'architecture_type': 'mobilenet_v2'
        },
        'optimization_params': {
            'use_mixed_precision': False,
            'gradient_clipping': True,
            'use_ema': False
        }
    }
    configs.append(('mobilenet_v2_lite', lite_config))
    
    # MobileNetV2-Plus: Enhanced expansion for better accuracy
    plus_config = {
        'architecture_type': 'mobilenet_v2',
        'input_shape': (224, 224, 3),
        'num_classes': 1000,
        'arch_params': {
            'width_multiplier': 1.0,
            'expansion_factor': 8,  # Enhanced expansion for accuracy
            'architecture_type': 'mobilenet_v2'
        },
        'optimization_params': {
            'use_mixed_precision': False,
            'gradient_clipping': True,
            'use_ema': False
        }
    }
    configs.append(('mobilenet_v2_plus', plus_config))
    
    # Person Detection Optimized variants
    person_detection_configs = [
        # Person detection with optimized input sizes
        {
            'name': 'mobilenet_v2_person_96',
            'input_shape': (96, 96, 3),
            'num_classes': 2,  # Person/No-person
            'width_multiplier': 0.5,
            'expansion_factor': 4,
        },
        {
            'name': 'mobilenet_v2_person_128',
            'input_shape': (128, 128, 3),
            'num_classes': 2,
            'width_multiplier': 0.75,
            'expansion_factor': 6,
        },
        {
            'name': 'mobilenet_v2_person_224',
            'input_shape': (224, 224, 3),
            'num_classes': 2,
            'width_multiplier': 1.0,
            'expansion_factor': 6,
        }
    ]
    
    for pd_config in person_detection_configs:
        config_dict = {
            'architecture_type': 'mobilenet_v2',
            'input_shape': pd_config['input_shape'],
            'num_classes': pd_config['num_classes'],
            'arch_params': {
                'width_multiplier': pd_config['width_multiplier'],
                'expansion_factor': pd_config['expansion_factor'],
                'architecture_type': 'mobilenet_v2'
            },
            'optimization_params': {
                'use_mixed_precision': False,
                'gradient_clipping': True,
                'use_ema': False
            }
        }
        configs.append((pd_config['name'], config_dict))
    
    return configs


def create_custom_mobilenet_v2_config(
    width_multiplier: float,
    expansion_factor: int = 6,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1000
) -> dict:
    """Create a custom MobileNetV2 configuration.
    
    Args:
        width_multiplier: Width multiplier (0.1 to 2.0)
        expansion_factor: Expansion factor for inverted residuals (1 to 12)
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Configuration dictionary
    """
    return {
        'architecture_type': 'mobilenet_v2',
        'input_shape': input_shape,
        'num_classes': num_classes,
        'arch_params': {
            'width_multiplier': width_multiplier,
            'expansion_factor': expansion_factor,
            'architecture_type': 'mobilenet_v2'
        },
        'optimization_params': {
            'use_mixed_precision': False,
            'gradient_clipping': True,
            'use_ema': False
        }
    }


# Register MobileNetV2 architectures with different width multipliers and variants
def _register_mobilenet_v2_variants():
    """Register MobileNetV2 variants with different width multipliers and architectural variants."""
    
    # Register standard width multiplier variants
    standard_configs = create_mobilenet_v2_configs()
    for arch_name, config_dict in standard_configs:
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MobileNetV2Architecture,
            default_config=config_dict
        )
    
    # Register architectural variants (Lite, Plus, Person Detection)
    variant_configs = create_mobilenet_v2_variant_configs()
    for arch_name, config_dict in variant_configs:
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MobileNetV2Architecture,
            default_config=config_dict
        )


# Auto-register when module is imported
_register_mobilenet_v2_variants() 