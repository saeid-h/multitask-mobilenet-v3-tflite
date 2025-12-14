"""Input parsing utilities."""

from typing import Tuple


def parse_input_shape(shape_str: str) -> Tuple[int, int, int]:
    """
    Parse input shape string like '224x224x3' into tuple.
    
    Args:
        shape_str: Input shape string in format HxWxC (e.g., "224x224x3")
        
    Returns:
        Tuple of (height, width, channels)
        
    Raises:
        ValueError: If format is invalid
    """
    try:
        parts = shape_str.split('x')
        if len(parts) != 3:
            raise ValueError("Input shape must be in format HxWxC (e.g., 224x224x3)")
        return tuple(int(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"Invalid input shape format: {e}")

