"""
Multi-head model configuration for multi-head MobileNet architectures.

This module extends the base ModelConfig to support multi-head model configurations,
including head-specific parameters and multi-task training settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from ..base import ModelConfig
from .head_configuration import HeadConfiguration, MultiHeadConfiguration, create_head_config_from_list


@dataclass
class MultiHeadModelConfig(ModelConfig):
    """Extended model configuration for multi-head architectures.
    
    This class extends the base ModelConfig to support multi-head model
    configurations, including head-specific parameters and multi-task
    training settings.
    
    Attributes:
        head_configs: List of head configurations
        training_mode: Multi-task training mode ('joint', 'sequential', 'hybrid')
        inference_mode: Inference mode ('all_active', 'selective')
        loss_weights: Optional custom loss weights for each head
        head_specific_params: Head-specific parameters for each head
    """
    
    # Multi-head specific parameters
    head_configs: List[HeadConfiguration] = field(default_factory=list)
    training_mode: str = 'joint'
    inference_mode: str = 'all_active'
    loss_weights: Optional[Dict[str, float]] = None
    head_specific_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate multi-head configuration parameters.
        
        This method extends the base validation to include multi-head
        specific validation.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Call base validation first
        super().validate()
        
        # Validate multi-head specific parameters
        if not self.head_configs:
            raise ValueError("At least one head configuration must be provided")
        
        # Validate each head configuration
        for head_config in self.head_configs:
            head_config.validate()
        
        # Check for duplicate head names
        head_names = [head.name for head in self.head_configs]
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
        
        # Validate loss weights if provided
        if self.loss_weights is not None:
            for head_name, weight in self.loss_weights.items():
                if weight < 0.0:
                    raise ValueError(f"Loss weight for head '{head_name}' must be non-negative, got {weight}")
            
            # Check that all head names in loss_weights exist in head_configs
            config_head_names = set(head_names)
            weight_head_names = set(self.loss_weights.keys())
            if not weight_head_names.issubset(config_head_names):
                missing_heads = weight_head_names - config_head_names
                raise ValueError(f"Loss weights specified for non-existent heads: {missing_heads}")
    
    def get_head_config_by_name(self, name: str) -> Optional[HeadConfiguration]:
        """Get head configuration by name.
        
        Args:
            name: Name of the head to find
            
        Returns:
            HeadConfiguration if found, None otherwise
        """
        for head_config in self.head_configs:
            if head_config.name == name:
                return head_config
        return None
    
    def get_head_names(self) -> List[str]:
        """Get list of all head names.
        
        Returns:
            List of head names
        """
        return [head.name for head in self.head_configs]
    
    def get_total_classes(self) -> int:
        """Get total number of classes across all heads.
        
        Returns:
            Total number of classes
        """
        return sum(head.num_classes for head in self.head_configs)
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights for all heads.
        
        If custom loss weights are provided, use them. Otherwise,
        use the loss weights from individual head configurations.
        
        Returns:
            Dictionary mapping head names to loss weights
        """
        if self.loss_weights is not None:
            return self.loss_weights.copy()
        
        # Use loss weights from head configurations
        weights = {}
        for head_config in self.head_configs:
            weights[head_config.name] = head_config.loss_weight
        
        return weights
    
    def get_head_specific_params(self, head_name: str) -> Dict[str, Any]:
        """Get head-specific parameters for a given head.
        
        Args:
            head_name: Name of the head
            
        Returns:
            Dictionary of head-specific parameters
        """
        return self.head_specific_params.get(head_name, {}).copy()
    
    def to_multi_head_config(self) -> MultiHeadConfiguration:
        """Convert to MultiHeadConfiguration format.
        
        Returns:
            MultiHeadConfiguration instance
        """
        return MultiHeadConfiguration(
            backbone_config=self.arch_params.copy(),
            heads=self.head_configs.copy(),
            training_mode=self.training_mode,
            inference_mode=self.inference_mode
        )
    
    @classmethod
    def from_head_list(cls, 
                      head_config_list: List[int],
                      head_names: Optional[List[str]] = None,
                      **kwargs) -> 'MultiHeadModelConfig':
        """Create MultiHeadModelConfig from a list of class counts.
        
        This is a convenience method for creating multi-head configurations
        from simple lists of class counts.
        
        Args:
            head_config_list: List of class counts (e.g., [2, 2, 5])
            head_names: Optional list of head names
            **kwargs: Additional configuration parameters
            
        Returns:
            MultiHeadModelConfig instance
        """
        # Create head configurations from the list
        head_configs = create_head_config_from_list(head_config_list, head_names)
        
        # Create the multi-head config
        config = cls(head_configs=head_configs, **kwargs)
        
        return config
    
    @classmethod
    def from_multi_head_config(cls, 
                              multi_head_config: MultiHeadConfiguration,
                              **kwargs) -> 'MultiHeadModelConfig':
        """Create MultiHeadModelConfig from MultiHeadConfiguration.
        
        Args:
            multi_head_config: MultiHeadConfiguration instance
            **kwargs: Additional configuration parameters
            
        Returns:
            MultiHeadModelConfig instance
        """
        # Extract backbone config
        arch_params = multi_head_config.backbone_config.copy()
        
        # Create the multi-head config
        config = cls(
            head_configs=multi_head_config.heads.copy(),
            training_mode=multi_head_config.training_mode,
            inference_mode=multi_head_config.inference_mode,
            arch_params=arch_params,
            **kwargs
        )
        
        return config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Return the complete model configuration as a dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        base_config = super().get_model_config()
        
        # Add multi-head specific parameters
        multi_head_config = {
            'head_configs': [head.to_dict() for head in self.head_configs],
            'training_mode': self.training_mode,
            'inference_mode': self.inference_mode,
            'loss_weights': self.loss_weights,
            'head_specific_params': self.head_specific_params.copy(),
            'total_classes': self.get_total_classes(),
            'head_names': self.get_head_names()
        }
        
        base_config.update(multi_head_config)
        return base_config
