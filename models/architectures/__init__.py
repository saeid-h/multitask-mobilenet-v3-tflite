"""
MobileNet Architectures Package

This package contains complete architecture implementations for different
MobileNet variants. Each architecture is built using the modular components
system and can be instantiated through the model factory.

Available architectures:
- MobileNetV1 (various width multipliers)
- MobileNetV2 (various width multipliers)
- MobileNetV2 QAT (various alpha values for quantization-aware training)
- MobileNetV3-Small (various width multipliers)
- MobileNetV4-Conv-S (various width multipliers)
- Multi-head architectures (shared backbone with multiple classification heads)
"""

# Import architectures to register them with the factory
from . import mobilenet_v1  # This will auto-register MobileNetV1 variants
from . import mobilenet_v1_qat  # This will auto-register MobileNetV1 QAT variants
from . import mobilenet_v2  # This will auto-register MobileNetV2 variants
from . import mobilenet_v2_qat  # This will auto-register MobileNetV2 QAT variants
from . import mobilenet_v3_small  # This will auto-register MobileNetV3-Small variants
from . import mobilenet_v3_qat  # This will auto-register MobileNetV3 QAT variants
from . import mobilenet_v4  # This will auto-register MobileNetV4 variants
from . import mobilenet_v4_qat  # This will auto-register MobileNetV4 QAT variants

# Import multi-head base architecture
from .multi_head_base import MultiHeadMobileNetArchitecture

# Import multi-head MobileNetV3 QAT architecture
from . import mobilenet_v3_qat_multi

__version__ = '0.2.0' 