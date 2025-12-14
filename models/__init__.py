"""
Models package for multi-architecture MobileNet implementation.

This package provides a modular factory pattern for creating different
MobileNet architectures (V1, V3-Small, V4-Conv-S) with consistent
configuration and initialization patterns.
"""

from .base import MobileNetArchitecture, ModelConfig
from .factory import ModelArchitectureFactory

# Import MobileNetV1 after factory to avoid circular imports
from .architectures.mobilenet_v1 import MobileNetV1Architecture, create_mobilenet_v1_025, _register_mobilenet_v1_variants

# Import MobileNetV3-Small
from .architectures.mobilenet_v3_small import MobileNetV3Small, create_mobilenet_v3_small_configs

# Import MobileNetV4-Conv-S
from .architectures.mobilenet_v4 import MobileNetV4ConvS, create_mobilenet_v4_conv_s_configs

def _register_mobilenet_v3_small_variants():
    """Register all MobileNetV3-Small variants with the factory."""
    configs = create_mobilenet_v3_small_configs()
    
    for arch_name, default_config in configs:
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MobileNetV3Small,
            default_config=default_config
        )

def _register_mobilenet_v4_variants():
    """Register all MobileNetV4 variants with the factory."""
    configs = create_mobilenet_v4_conv_s_configs()
    
    for arch_name, default_config in configs:
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MobileNetV4ConvS,
            default_config=default_config
        )

# Register all architecture variants after imports are complete
_register_mobilenet_v1_variants()
_register_mobilenet_v3_small_variants()
_register_mobilenet_v4_variants()

__all__ = [
    'MobileNetArchitecture',
    'ModelConfig', 
    'ModelArchitectureFactory',
    'MobileNetV1Architecture',
    'create_mobilenet_v1_025',
    'MobileNetV3Small',
    'MobileNetV4ConvS',
]

__version__ = '0.2.0' 