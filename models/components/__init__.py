"""
MobileNet Components Package

This package provides modular building blocks for constructing MobileNet
architectures. Components are designed to be reusable across MobileNetV1,
V3, and V4 implementations.

The package includes:
- Base classes and interfaces
- Convolution building blocks (standard and depthwise separable)
- Classification heads and pooling components
- Advanced blocks for V3/V4 architectures (SE blocks, inverted residuals)
- Mobile-optimized activation functions
"""

# Base classes and configuration
from .base import (
    MobileNetComponent,
    ConvolutionComponent, 
    PoolingComponent,
    ClassificationComponent,
    ComponentConfig
)

# Basic convolution blocks
from .conv_blocks import (
    StandardConvBlock,
    DepthwiseSeparableConvBlock
)

# Classification head components
from .heads import (
    GlobalAveragePoolingBlock,
    DenseClassificationHead,
    MobileNetClassificationHead
)

# Mobile-optimized activation functions
from .activations import (
    HardSwishActivation,
    HardSigmoidActivation,
    ReLU6Activation,
    get_activation,
    apply_activation,
    HardSwish,
    HardSigmoid,
    hard_swish,
    hard_sigmoid
)

# Squeeze-and-Excitation blocks
from .se_blocks import (
    SqueezeExciteBlock,
    EfficientSEBlock,
    AdaptiveSEBlock,
    SEBlock,
    EfficientSE,
    create_se_block
)

# Inverted residual blocks for MobileNetV3
from .inverted_blocks import (
    InvertedResidualBlock,
    MobileNetV3Block,
    LinearBottleneck,
    InvertedResidual,
    # MobileNetV2 specific blocks
    MobileNetV2InvertedResidualBlock,
    create_mobilenet_v2_inverted_block
)

# Universal Inverted Bottleneck blocks for MobileNetV4
from .uib_blocks import (
    UniversalInvertedBottleneck,
    create_ib_block,
    create_convnext_block,
    create_extradw_block,
    create_ffn_block
)

__all__ = [
    # Base classes
    'MobileNetComponent',
    'ConvolutionComponent',
    'PoolingComponent', 
    'ClassificationComponent',
    'ComponentConfig',
    
    # Convolution blocks
    'StandardConvBlock',
    'DepthwiseSeparableConvBlock',
    
    # Classification components
    'GlobalAveragePoolingBlock',
    'DenseClassificationHead',
    'MobileNetClassificationHead',
    
    # Activation functions
    'HardSwishActivation',
    'HardSigmoidActivation',
    'ReLU6Activation',
    'get_activation',
    'apply_activation',
    'HardSwish',
    'HardSigmoid',
    'hard_swish',
    'hard_sigmoid',
    
    # SE blocks
    'SqueezeExciteBlock',
    'EfficientSEBlock',
    'AdaptiveSEBlock',
    'SEBlock',
    'EfficientSE',
    'create_se_block',
    
    # Inverted residual blocks
    'InvertedResidualBlock',
    'MobileNetV3Block',
    'LinearBottleneck',
    'InvertedResidual',
    # MobileNetV2 specific blocks
    'MobileNetV2InvertedResidualBlock',
    'create_mobilenet_v2_inverted_block',
    
    # Universal Inverted Bottleneck blocks
    'UniversalInvertedBottleneck',
    'create_ib_block',
    'create_convnext_block',
    'create_extradw_block',
    'create_ffn_block',
]

__version__ = '0.3.0' 