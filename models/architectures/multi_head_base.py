"""
Base multi-head architecture for MobileNet variants.

This module defines the base class for multi-head MobileNet architectures,
providing a common interface for creating models with multiple classification
heads sharing a single backbone.
"""

from abc import abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model

from ..base import MobileNetArchitecture
from ..components.multi_head_model_config import MultiHeadModelConfig
from ..components.head_configuration import HeadConfiguration


class MultiHeadMobileNetArchitecture(MobileNetArchitecture):
    """Abstract base class for multi-head MobileNet architectures.
    
    This class extends MobileNetArchitecture to support multiple classification
    heads sharing a single backbone. It provides a common interface for all
    multi-head MobileNet variants.
    
    Key features:
    - Shared backbone across multiple heads
    - Flexible head configuration system
    - Support for different training and inference modes
    - QAT compatibility for quantization-aware training
    """
    
    def __init__(self, config: MultiHeadModelConfig):
        """Initialize the multi-head architecture with configuration.
        
        Args:
            config: Multi-head model configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate that config is a MultiHeadModelConfig
        if not isinstance(config, MultiHeadModelConfig):
            raise ValueError(f"config must be MultiHeadModelConfig, got {type(config)}")
        
        # Call parent constructor
        super().__init__(config)
    
    @property
    def multi_head_config(self) -> MultiHeadModelConfig:
        """Get the multi-head configuration.
        
        Returns:
            MultiHeadModelConfig instance
        """
        return self.config
    
    @property
    def head_configs(self) -> List[HeadConfiguration]:
        """Get list of head configurations.
        
        Returns:
            List of HeadConfiguration objects
        """
        return self.multi_head_config.head_configs
    
    @property
    def head_names(self) -> List[str]:
        """Get list of head names.
        
        Returns:
            List of head names
        """
        return self.multi_head_config.get_head_names()
    
    @property
    def total_classes(self) -> int:
        """Get total number of classes across all heads.
        
        Returns:
            Total number of classes
        """
        return self.multi_head_config.get_total_classes()
    
    @property
    def training_mode(self) -> str:
        """Get training mode.
        
        Returns:
            Training mode ('joint', 'sequential', 'hybrid')
        """
        return self.multi_head_config.training_mode
    
    @property
    def inference_mode(self) -> str:
        """Get inference mode.
        
        Returns:
            Inference mode ('all_active', 'selective')
        """
        return self.multi_head_config.inference_mode
    
    def get_head_config_by_name(self, name: str) -> Optional[HeadConfiguration]:
        """Get head configuration by name.
        
        Args:
            name: Name of the head to find
            
        Returns:
            HeadConfiguration if found, None otherwise
        """
        return self.multi_head_config.get_head_config_by_name(name)
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights for all heads.
        
        Returns:
            Dictionary mapping head names to loss weights
        """
        return self.multi_head_config.get_loss_weights()
    
    @abstractmethod
    def build_backbone(self) -> tf.keras.Model:
        """Build the shared backbone model.
        
        This method should create the backbone architecture that will be
        shared across all heads. The backbone should output feature maps
        that can be used by multiple classification heads.
        
        Returns:
            TensorFlow Keras backbone model
            
        Raises:
            ValueError: If backbone cannot be built with current configuration
        """
        pass
    
    def build_head(self, head_config: HeadConfiguration, backbone_output: tf.Tensor) -> tf.Tensor:
        """Build a single classification head.
        
        This method creates a classification head for the given configuration.
        The head takes the backbone output and produces classification predictions.
        
        Args:
            head_config: Configuration for the head
            backbone_output: Output tensor from the backbone
            
        Returns:
            Classification output tensor
        """
        # Standard head architecture: GlobalAvgPool + Dropout + Dense
        x = layers.GlobalAveragePooling2D(name=f"{head_config.name}_global_pool")(backbone_output)
        x = layers.Dropout(head_config.dropout_rate, name=f"{head_config.name}_dropout")(x)
        
        # Output layer with specified activation
        output = layers.Dense(
            head_config.num_classes,
            activation=head_config.activation,
            name=f"{head_config.name}_output"
        )(x)
        
        return output
    
    def build_model(self) -> tf.keras.Model:
        """Build and return the complete multi-head TensorFlow model.
        
        This method creates the complete multi-head model by:
        1. Building the shared backbone
        2. Creating multiple classification heads
        3. Connecting heads to backbone outputs
        4. Creating a model with multiple outputs
        
        Returns:
            Compiled TensorFlow Keras model with multiple outputs
            
        Raises:
            ValueError: If model cannot be built with current configuration
        """
        # Build the shared backbone
        backbone = self.build_backbone()
        
        # Create input layer
        input_layer = layers.Input(shape=self.config.input_shape, name='input')
        
        # Get backbone output
        backbone_output = backbone(input_layer)
        
        # Create multiple heads
        outputs = {}
        for head_config in self.head_configs:
            head_output = self.build_head(head_config, backbone_output)
            outputs[head_config.name] = head_output
        
        # Create the complete model
        model = Model(inputs=input_layer, outputs=outputs, name=self.name)
        
        return model
    
    def validate_config(self) -> None:
        """Validate multi-head architecture-specific configuration parameters.
        
        This method extends the base validation to include multi-head
        specific validation.
        
        Raises:
            ValueError: If any architecture-specific parameter is invalid
        """
        # Call parent validation
        super().validate_config()
        
        # Validate multi-head specific parameters
        self.multi_head_config.validate()
        
        # Additional architecture-specific validation can be added here
        # by subclasses that override this method
    
    def get_model_config(self) -> Dict[str, Any]:
        """Return the complete model configuration as a dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return self.multi_head_config.get_model_config()
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Return model parameter count information.
        
        Returns:
            Dictionary with parameter count details:
            - total: Total number of parameters
            - trainable: Number of trainable parameters  
            - non_trainable: Number of non-trainable parameters
            - backbone: Number of backbone parameters
            - heads: Number of head parameters
        """
        model = self.get_model()
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # Calculate backbone vs head parameters
        backbone = self.build_backbone()
        backbone_params = backbone.count_params()
        head_params = total_params - backbone_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params,
            'backbone': backbone_params,
            'heads': head_params
        }
    
    def __str__(self) -> str:
        """Return string representation of the multi-head architecture."""
        param_info = self.get_parameter_count()
        head_info = f"heads={len(self.head_configs)}"
        return (
            f"{self.name}("
            f"input_shape={self.config.input_shape}, "
            f"{head_info}, "
            f"total_params={param_info['total']:,})"
        )
    
    def __repr__(self) -> str:
        """Return detailed string representation of the multi-head architecture."""
        return f"{self.__class__.__name__}(config={self.multi_head_config})"
