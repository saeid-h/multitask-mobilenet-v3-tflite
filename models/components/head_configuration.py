"""
Head configuration data structures for multi-head architectures.

This module defines the data structures used to configure individual heads
in multi-head MobileNet architectures.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import tensorflow as tf


@dataclass
class HeadConfiguration:
    """Configuration for a single classification head in a multi-head model.
    
    This dataclass defines the parameters for individual heads in multi-head
    architectures, allowing flexible configuration of each head's properties.
    
    Attributes:
        name: Unique name for the head (e.g., 'person_detection', 'gender_classification')
        num_classes: Number of output classes for this head
        activation: Activation function for the output layer ('softmax', 'sigmoid', etc.)
        dropout_rate: Dropout rate for regularization in the head
        loss_weight: Weight for this head's loss in multi-task training
        head_type: Type of head architecture ('standard', 'custom', etc.)
        custom_params: Additional custom parameters for the head
    """
    
    name: str
    num_classes: int
    activation: str = 'softmax'
    dropout_rate: float = 0.2
    loss_weight: float = 1.0
    head_type: str = 'standard'
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate head configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"name must be a non-empty string, got {self.name}")
        
        if self.num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        
        valid_activations = {'softmax', 'sigmoid', 'linear', 'relu', 'tanh'}
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, got {self.activation}")
        
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")
        
        if self.loss_weight < 0.0:
            raise ValueError(f"loss_weight must be non-negative, got {self.loss_weight}")
        
        valid_head_types = {'standard', 'custom'}
        if self.head_type not in valid_head_types:
            raise ValueError(f"head_type must be one of {valid_head_types}, got {self.head_type}")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'name': self.name,
            'num_classes': self.num_classes,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'loss_weight': self.loss_weight,
            'head_type': self.head_type,
            'custom_params': self.custom_params.copy()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HeadConfiguration':
        """Create HeadConfiguration from dictionary.
        
        Args:
            config_dict: Dictionary containing head configuration
            
        Returns:
            HeadConfiguration instance
            
        Raises:
            ValueError: If dictionary is invalid
        """
        required_keys = {'name', 'num_classes'}
        missing_keys = required_keys - set(config_dict.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        
        return cls(**config_dict)


@dataclass
class MultiHeadConfiguration:
    """Configuration for a complete multi-head model.
    
    This dataclass defines the configuration for a multi-head model,
    including the backbone configuration and all head configurations.
    
    Attributes:
        backbone_config: Configuration for the shared backbone
        heads: List of head configurations
        training_mode: Training mode ('joint', 'sequential', 'hybrid')
        inference_mode: Inference mode ('all_active', 'selective')
    """
    
    backbone_config: Dict[str, Any]
    heads: List[HeadConfiguration]
    training_mode: str = 'joint'
    inference_mode: str = 'all_active'
    
    def validate(self) -> None:
        """Validate multi-head configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.heads:
            raise ValueError("At least one head must be configured")
        
        # Validate each head
        for head in self.heads:
            head.validate()
        
        # Check for duplicate head names
        head_names = [head.name for head in self.heads]
        if len(head_names) != len(set(head_names)):
            raise ValueError(f"Duplicate head names found: {head_names}")
        
        # Validate training mode
        valid_training_modes = {'joint', 'sequential', 'hybrid'}
        if self.training_mode not in valid_training_modes:
            raise ValueError(f"training_mode must be one of {valid_training_modes}, got {self.training_mode}")
        
        # Validate inference mode
        valid_inference_modes = {'all_active', 'selective'}
        if self.inference_mode not in valid_inference_modes:
            raise ValueError(f"inference_mode must be one of {valid_inference_modes}, got {self.inference_mode}")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def get_head_by_name(self, name: str) -> Optional[HeadConfiguration]:
        """Get head configuration by name.
        
        Args:
            name: Name of the head to find
            
        Returns:
            HeadConfiguration if found, None otherwise
        """
        for head in self.heads:
            if head.name == name:
                return head
        return None
    
    def get_head_names(self) -> List[str]:
        """Get list of all head names.
        
        Returns:
            List of head names
        """
        return [head.name for head in self.heads]
    
    def get_total_classes(self) -> int:
        """Get total number of classes across all heads.
        
        Returns:
            Total number of classes
        """
        return sum(head.num_classes for head in self.heads)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'backbone_config': self.backbone_config.copy(),
            'heads': [head.to_dict() for head in self.heads],
            'training_mode': self.training_mode,
            'inference_mode': self.inference_mode
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultiHeadConfiguration':
        """Create MultiHeadConfiguration from dictionary.
        
        Args:
            config_dict: Dictionary containing multi-head configuration
            
        Returns:
            MultiHeadConfiguration instance
            
        Raises:
            ValueError: If dictionary is invalid
        """
        required_keys = {'backbone_config', 'heads'}
        missing_keys = required_keys - set(config_dict.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        
        # Convert head dictionaries to HeadConfiguration objects
        heads = [HeadConfiguration.from_dict(head_dict) for head_dict in config_dict['heads']]
        
        return cls(
            backbone_config=config_dict['backbone_config'],
            heads=heads,
            training_mode=config_dict.get('training_mode', 'joint'),
            inference_mode=config_dict.get('inference_mode', 'all_active')
        )


def create_head_config_from_list(head_config_list: List[int], 
                                head_names: Optional[List[str]] = None) -> List[HeadConfiguration]:
    """Create head configurations from a list of class counts.
    
    This utility function creates HeadConfiguration objects from a simple list
    of class counts, which is useful for quick configuration.
    
    Args:
        head_config_list: List of class counts (e.g., [2, 2, 5])
        head_names: Optional list of head names. If None, auto-generates names
        
    Returns:
        List of HeadConfiguration objects
        
    Raises:
        ValueError: If lists are invalid
    """
    if not head_config_list:
        raise ValueError("head_config_list cannot be empty")
    
    if head_names is not None and len(head_names) != len(head_config_list):
        raise ValueError(f"head_names length ({len(head_names)}) must match head_config_list length ({len(head_config_list)})")
    
    heads = []
    for i, num_classes in enumerate(head_config_list):
        if head_names:
            name = head_names[i]
        else:
            name = f"head_{i+1}"
        
        head_config = HeadConfiguration(
            name=name,
            num_classes=num_classes
        )
        heads.append(head_config)
    
    return heads
