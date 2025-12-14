"""
Constants for input configuration support.

This module defines supported input configurations and provides utilities
for input configuration validation and management.
"""

from typing import Dict, Tuple, List, Optional

# Supported input configurations
SUPPORTED_INPUT_CONFIGS = {
    '96x96x1': (96, 96, 1),     # Grayscale 96x96
    '96x96x3': (96, 96, 3),     # RGB 96x96
    '128x128x1': (128, 128, 1), # Grayscale 128x128
    '128x128x3': (128, 128, 3), # RGB 128x128
    '224x224x1': (224, 224, 1), # Grayscale 224x224
    '224x224x3': (224, 224, 3), # RGB 224x224
    '256x256x1': (256, 256, 1), # Grayscale 256x256
    '256x256x3': (256, 256, 3), # RGB 256x256
}

# Default configuration (backward compatibility)
DEFAULT_INPUT_CONFIG = '96x96x1'
DEFAULT_INPUT_SHAPE = SUPPORTED_INPUT_CONFIGS[DEFAULT_INPUT_CONFIG]

# Supported resolutions
SUPPORTED_RESOLUTIONS = [96, 128, 224, 256]

# Supported channel counts
SUPPORTED_CHANNELS = [1, 3]  # Grayscale, RGB

# QAT-specific constants
QAT_ALPHA_VALUES = {
    'mobilenet_v1': [0.25, 0.5, 0.75, 1.0],
    'mobilenet_v2': [0.25, 0.35, 0.5, 0.75, 1.0],  # Reference: 0.25~400KB, 0.35~600KB
    'mobilenet_v3': [0.25, 0.35, 0.5, 0.75, 1.0],
    'mobilenet_v4': [0.25, 0.35, 0.5, 0.75, 1.0],
}

# QAT model size targets (reference from successful project)
QAT_MODEL_SIZE_TARGETS = {
    'mobilenet_v2_0_25': 400,  # KB
    'mobilenet_v2_0_35': 600,  # KB
}

# QAT architecture naming patterns
QAT_ARCHITECTURE_PATTERNS = {
    'mobilenet_v1': 'mobilenet_v1_qat_{alpha}',
    'mobilenet_v2': 'mobilenet_v2_qat_{alpha}',
    'mobilenet_v3': 'mobilenet_v3_qat_{alpha}',
    'mobilenet_v4': 'mobilenet_v4_qat_{alpha}',
}

# QAT training parameters
QAT_TRAINING_PARAMS = {
    'representative_dataset_size': 200,  # Number of samples for quantization calibration
    'focal_loss_alpha': 0.25,  # Focal loss alpha parameter
    'focal_loss_gamma': 2.0,   # Focal loss gamma parameter
    'stage1_epochs': 10,       # Stage 1: Freeze backbone
    'stage2_epochs': 20,       # Stage 2: Unfreeze backbone
    'finetune_lr': 0.0001,     # Learning rate for fine-tuning
}

# Multi-head architecture constants
MULTI_HEAD_TRAINING_MODES = {
    'joint': 'joint',           # Train all heads simultaneously
    'sequential': 'sequential', # Train heads sequentially
    'hybrid': 'hybrid',         # Hybrid training approach
}

MULTI_HEAD_INFERENCE_MODES = {
    'all_active': 'all_active',     # Run all heads during inference
    'selective': 'selective',       # Run only selected heads
}

# Multi-head naming patterns
MULTI_HEAD_ARCHITECTURE_PATTERNS = {
    'mobilenet_v3': 'mobilenet_v3_qat_multi_{head_config}',
}

# Multi-head training parameters
MULTI_HEAD_TRAINING_PARAMS = {
    'default_loss_weight': 1.0,     # Default loss weight for heads
    'equal_loss_weighting': True,   # Use equal loss weights by default
    'head_specific_lr': False,      # Use head-specific learning rates
    'backbone_freeze_epochs': 5,    # Epochs to freeze backbone initially
}

# Input configuration validation
def validate_input_config(input_shape: Tuple[int, int, int]) -> bool:
    """Validate if input shape is supported.
    
    Args:
        input_shape: Tuple of (height, width, channels)
        
    Returns:
        True if input shape is supported, False otherwise
    """
    return input_shape in SUPPORTED_INPUT_CONFIGS.values()

def get_input_config_name(input_shape: Tuple[int, int, int]) -> str:
    """Get configuration name from input shape.
    
    Args:
        input_shape: Tuple of (height, width, channels)
        
    Returns:
        Configuration name (e.g., '96x96x1', '224x224x3')
        
    Raises:
        ValueError: If input shape is not supported
    """
    for name, shape in SUPPORTED_INPUT_CONFIGS.items():
        if shape == input_shape:
            return name
    raise ValueError(f"Unsupported input shape: {input_shape}")

def get_input_shape_from_name(config_name: str) -> Tuple[int, int, int]:
    """Get input shape from configuration name.
    
    Args:
        config_name: Configuration name (e.g., '96x96x1', '224x224x3')
        
    Returns:
        Input shape tuple (height, width, channels)
        
    Raises:
        ValueError: If configuration name is not supported
    """
    if config_name not in SUPPORTED_INPUT_CONFIGS:
        raise ValueError(f"Unsupported configuration name: {config_name}")
    return SUPPORTED_INPUT_CONFIGS[config_name]

def get_supported_config_names() -> List[str]:
    """Get list of all supported configuration names.
    
    Returns:
        List of supported configuration names
    """
    return list(SUPPORTED_INPUT_CONFIGS.keys())

def get_supported_input_shapes() -> List[Tuple[int, int, int]]:
    """Get list of all supported input shapes.
    
    Returns:
        List of supported input shape tuples
    """
    return list(SUPPORTED_INPUT_CONFIGS.values())

def get_configs_by_resolution(resolution: int) -> List[str]:
    """Get configuration names for a specific resolution.
    
    Args:
        resolution: Input resolution (96, 128, 224, 256)
        
    Returns:
        List of configuration names for the resolution
        
    Raises:
        ValueError: If resolution is not supported
    """
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(f"Unsupported resolution: {resolution}")
    
    return [name for name, shape in SUPPORTED_INPUT_CONFIGS.items() 
            if shape[0] == resolution]

def get_configs_by_channels(channels: int) -> List[str]:
    """Get configuration names for a specific channel count.
    
    Args:
        channels: Number of channels (1 for grayscale, 3 for RGB)
        
    Returns:
        List of configuration names for the channel count
        
    Raises:
        ValueError: If channel count is not supported
    """
    if channels not in SUPPORTED_CHANNELS:
        raise ValueError(f"Unsupported channel count: {channels}")
    
    return [name for name, shape in SUPPORTED_INPUT_CONFIGS.items() 
            if shape[2] == channels]

def parse_input_shape_string(shape_str: str) -> Tuple[int, int, int]:
    """Parse input shape from string format.
    
    Args:
        shape_str: Input shape string (e.g., '96x96x1', '224x224x3')
        
    Returns:
        Input shape tuple (height, width, channels)
        
    Raises:
        ValueError: If string format is invalid or shape is not supported
    """
    try:
        # Handle tuple string format like "(96, 96, 1)"
        if shape_str.startswith('(') and shape_str.endswith(')'):
            shape_str = shape_str[1:-1]  # Remove parentheses
        
        # Split by 'x' or ','
        if 'x' in shape_str:
            parts = shape_str.split('x')
        else:
            parts = shape_str.split(',')
        
        if len(parts) != 3:
            raise ValueError(f"Invalid shape format: {shape_str}")
        
        height = int(parts[0].strip())
        width = int(parts[1].strip())
        channels = int(parts[2].strip())
        
        input_shape = (height, width, channels)
        
        if not validate_input_config(input_shape):
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        return input_shape
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid input shape string '{shape_str}': {e}")

def get_input_config_info(config_name: str) -> Dict[str, any]:
    """Get detailed information about an input configuration.
    
    Args:
        config_name: Configuration name (e.g., '96x96x1', '224x224x3')
        
    Returns:
        Dictionary with configuration information
        
    Raises:
        ValueError: If configuration name is not supported
    """
    if config_name not in SUPPORTED_INPUT_CONFIGS:
        raise ValueError(f"Unsupported configuration name: {config_name}")
    
    height, width, channels = SUPPORTED_INPUT_CONFIGS[config_name]
    
    return {
        'name': config_name,
        'resolution': (height, width),
        'channels': channels,
        'input_shape': (height, width, channels),
        'is_grayscale': channels == 1,
        'is_rgb': channels == 3,
        'pixel_count': height * width,
        'input_size_bytes_fp32': height * width * channels * 4,  # 4 bytes per float32
        'input_size_bytes_int8': height * width * channels,      # 1 byte per int8
        'description': f"{height}x{width} {'RGB' if channels == 3 else 'Grayscale'}",
        'use_case': _get_use_case_description(height, width, channels)
    }

def _get_use_case_description(height: int, width: int, channels: int) -> str:
    """Get use case description for input configuration.
    
    Args:
        height: Input height
        width: Input width
        channels: Number of channels
        
    Returns:
        Use case description
    """
    if height == 96:
        base_case = "MCU deployment"
    elif height == 128:
        base_case = "Medium performance"
    elif height == 224:
        base_case = "Standard deployment"
    elif height == 256:
        base_case = "High performance"
    else:
        base_case = "Custom deployment"
    
    if channels == 1:
        return f"{base_case} (grayscale)"
    else:
        return f"{base_case} (RGB)" 