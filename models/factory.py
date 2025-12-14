"""
Model Architecture Factory for creating MobileNet variants.

This module implements the factory pattern for creating different MobileNet
architectures with a registration system for extensibility.
"""

from typing import Dict, Type, List, Optional, Any
from .base import MobileNetArchitecture, ModelConfig


class ModelArchitectureFactory:
    """Factory class for creating MobileNet architecture instances.
    
    This factory manages the registration and creation of different MobileNet
    architecture variants. It provides a clean interface for instantiating
    models based on architecture type and configuration.
    
    The factory uses a registration system that allows new architectures
    to be added without modifying existing code.
    """
    
    # Registry of available architectures
    _architectures: Dict[str, Type[MobileNetArchitecture]] = {}
    
    # Default configurations for each architecture type
    _default_configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_architecture(
        cls, 
        name: str, 
        architecture_class: Type[MobileNetArchitecture],
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new architecture type with the factory.
        
        Args:
            name: Architecture identifier (e.g., 'mobilenet_v1')
            architecture_class: Class implementing MobileNetArchitecture
            default_config: Default configuration parameters for this architecture
            
        Raises:
            ValueError: If architecture name is already registered
            TypeError: If architecture_class doesn't inherit from MobileNetArchitecture
        """
        if name in cls._architectures:
            raise ValueError(f"Architecture '{name}' is already registered")
        
        if not issubclass(architecture_class, MobileNetArchitecture):
            raise TypeError(
                f"Architecture class must inherit from MobileNetArchitecture, "
                f"got {architecture_class.__name__}"
            )
        
        cls._architectures[name] = architecture_class
        cls._default_configs[name] = default_config or {}
        
        print(f"Registered architecture: {name}")
    
    @classmethod
    def unregister_architecture(cls, name: str) -> None:
        """Unregister an architecture type.
        
        Args:
            name: Architecture identifier to remove
            
        Raises:
            KeyError: If architecture name is not registered
        """
        if name not in cls._architectures:
            raise KeyError(f"Architecture '{name}' is not registered")
        
        del cls._architectures[name]
        del cls._default_configs[name]
        
        print(f"Unregistered architecture: {name}")
    
    @classmethod
    def get_available_architectures(cls) -> List[str]:
        """Return list of available architecture names.
        
        Returns:
            List of registered architecture identifiers
        """
        return list(cls._architectures.keys())
    
    @classmethod
    def is_architecture_available(cls, name: str) -> bool:
        """Check if an architecture is available.
        
        Args:
            name: Architecture identifier to check
            
        Returns:
            True if architecture is registered, False otherwise
        """
        return name in cls._architectures
    
    @classmethod
    def get_default_config(cls, arch_type: str) -> ModelConfig:
        """Get default configuration for an architecture type.
        
        Args:
            arch_type: Architecture identifier
            
        Returns:
            Default ModelConfig for the architecture
            
        Raises:
            KeyError: If architecture type is not registered
        """
        if arch_type not in cls._architectures:
            raise KeyError(
                f"Architecture '{arch_type}' not found. "
                f"Available architectures: {cls.get_available_architectures()}"
            )
        
        default_params = cls._default_configs[arch_type]
        architecture_class = cls._architectures[arch_type]
        
        # Check if this is a multi-head architecture
        try:
            from models.components.multi_head_model_config import MultiHeadModelConfig
            from models.components.head_configuration import HeadConfiguration
        except ImportError:
            # Fallback for relative imports
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from components.multi_head_model_config import MultiHeadModelConfig
            from components.head_configuration import HeadConfiguration
        
        if hasattr(architecture_class, '__bases__') and any('MultiHeadMobileNetArchitecture' in str(base) for base in architecture_class.__bases__):
            # This is a multi-head architecture, create MultiHeadModelConfig
            config = MultiHeadModelConfig()
            
            # Handle head_configs specially
            if 'head_configs' in default_params:
                head_configs = []
                for head_config_dict in default_params['head_configs']:
                    head_config = HeadConfiguration(**head_config_dict)
                    head_configs.append(head_config)
                config.head_configs = head_configs
                del default_params['head_configs']
            
            # Handle multi-head specific parameters
            for key, value in default_params.items():
                if key in ['training_mode', 'inference_mode', 'loss_weights', 'head_specific_params']:
                    setattr(config, key, value)
                elif key == 'arch_params':
                    config.arch_params.update(value)
                elif key == 'optimization_params':
                    config.optimization_params.update(value)
                elif hasattr(config, key):
                    setattr(config, key, value)
                else:
                    # Store unknown parameters in arch_params
                    config.arch_params[key] = value
        else:
            # Regular architecture, create standard ModelConfig
            config = ModelConfig()
            
            # Update with architecture-specific defaults
            for key, value in default_params.items():
                if key == 'arch_params':
                    config.arch_params.update(value)
                elif key == 'optimization_params':
                    config.optimization_params.update(value)
                elif hasattr(config, key):
                    setattr(config, key, value)
                else:
                    # Store unknown parameters in arch_params
                    config.arch_params[key] = value
        
        return config
    
    @classmethod
    def create_model(
        cls, 
        arch_type: str, 
        config: Optional[ModelConfig] = None,
        **kwargs
    ) -> MobileNetArchitecture:
        """Create a model architecture instance.
        
        Args:
            arch_type: Architecture identifier (e.g., 'mobilenet_v1')
            config: Model configuration. If None, uses default configuration
            **kwargs: Additional configuration parameters to override
            
        Returns:
            Architecture instance ready for model building
            
        Raises:
            KeyError: If architecture type is not registered
            ValueError: If configuration is invalid
        """
        if arch_type not in cls._architectures:
            raise KeyError(
                f"Architecture '{arch_type}' not found. "
                f"Available architectures: {cls.get_available_architectures()}"
            )
        
        # Get default configuration if none provided
        if config is None:
            config = cls.get_default_config(arch_type)
        else:
            # Make a copy to avoid modifying the original
            # Check if this is a multi-head config
            try:
                from models.components.multi_head_model_config import MultiHeadModelConfig
            except ImportError:
                # Fallback for relative imports
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                from components.multi_head_model_config import MultiHeadModelConfig
            
            if isinstance(config, MultiHeadModelConfig):
                # Copy MultiHeadModelConfig
                config = MultiHeadModelConfig(
                    input_shape=config.input_shape,
                    num_classes=config.num_classes,
                    dropout_rate=config.dropout_rate,
                    weight_decay=config.weight_decay,
                    batch_norm_momentum=config.batch_norm_momentum,
                    batch_norm_epsilon=config.batch_norm_epsilon,
                    activation=config.activation,
                    arch_params=config.arch_params.copy(),
                    optimization_params=config.optimization_params.copy(),
                    head_configs=config.head_configs.copy(),
                    training_mode=config.training_mode,
                    inference_mode=config.inference_mode,
                    loss_weights=config.loss_weights,
                    head_specific_params=config.head_specific_params.copy()
                )
            else:
                # Copy regular ModelConfig
                config = ModelConfig(
                    input_shape=config.input_shape,
                    num_classes=config.num_classes,
                    dropout_rate=config.dropout_rate,
                    weight_decay=config.weight_decay,
                    batch_norm_momentum=config.batch_norm_momentum,
                    batch_norm_epsilon=config.batch_norm_epsilon,
                    activation=config.activation,
                    arch_params=config.arch_params.copy(),
                    optimization_params=config.optimization_params.copy()
                )
        
        # Apply any additional parameters from kwargs
        for key, value in kwargs.items():
            if key in ['arch_params', 'optimization_params']:
                if isinstance(value, dict):
                    getattr(config, key).update(value)
                else:
                    getattr(config, key)[key] = value
            elif hasattr(config, key):
                setattr(config, key, value)
            else:
                # Store unknown parameters in arch_params
                config.arch_params[key] = value
        
        # Create and return architecture instance
        architecture_class = cls._architectures[arch_type]
        
        try:
            return architecture_class(config)
        except Exception as e:
            raise ValueError(
                f"Failed to create {arch_type} architecture: {str(e)}"
            ) from e
    
    @classmethod
    def get_architecture_info(cls, arch_type: str) -> Dict[str, Any]:
        """Get information about a registered architecture.
        
        Args:
            arch_type: Architecture identifier
            
        Returns:
            Dictionary with architecture information
            
        Raises:
            KeyError: If architecture type is not registered
        """
        if arch_type not in cls._architectures:
            raise KeyError(
                f"Architecture '{arch_type}' not found. "
                f"Available architectures: {cls.get_available_architectures()}"
            )
        
        architecture_class = cls._architectures[arch_type]
        default_config = cls._default_configs[arch_type]
        
        return {
            'name': arch_type,
            'class': architecture_class.__name__,
            'module': architecture_class.__module__,
            'default_config': default_config,
            'docstring': architecture_class.__doc__,
        }
    
    @classmethod
    def list_architectures(cls, detailed: bool = False) -> Dict[str, Any]:
        """List all registered architectures.
        
        Args:
            detailed: If True, include detailed information for each architecture
            
        Returns:
            Dictionary mapping architecture names to information
        """
        if detailed:
            return {
                name: cls.get_architecture_info(name) 
                for name in cls.get_available_architectures()
            }
        else:
            return {
                name: cls._architectures[name].__name__
                for name in cls.get_available_architectures()
            }
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered architectures.
        
        Warning: This will remove all registered architectures. Use with caution.
        """
        cls._architectures.clear()
        cls._default_configs.clear()
        print("Cleared all registered architectures")


# Convenience functions for common operations
def create_model(arch_type: str, config: Optional[ModelConfig] = None, **kwargs) -> MobileNetArchitecture:
    """Convenience function to create a model architecture.
    
    Args:
        arch_type: Architecture identifier
        config: Model configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Architecture instance
    """
    return ModelArchitectureFactory.create_model(arch_type, config, **kwargs)


def get_available_architectures() -> List[str]:
    """Convenience function to get available architectures.
    
    Returns:
        List of available architecture names
    """
    return ModelArchitectureFactory.get_available_architectures()


def get_default_config(arch_type: str) -> ModelConfig:
    """Convenience function to get default configuration.
    
    Args:
        arch_type: Architecture identifier
        
    Returns:
        Default configuration for the architecture
    """
    return ModelArchitectureFactory.get_default_config(arch_type) 