"""
Base multi-head architecture for MobileNet variants.

This module defines the base class for multi-head MobileNet architectures,
providing a common interface for creating models with multiple classification
heads sharing a single backbone.
"""

from abc import abstractmethod
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
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
    - Separable weights: save/load backbone and heads separately
    - Trainable control: freeze backbone or individual heads
    """
    
    def __init__(self, config: MultiHeadModelConfig, unified_output: bool = False):
        """Initialize the multi-head architecture with configuration.
        
        Args:
            config: Multi-head model configuration
            unified_output: If True, add unified concatenated output for multi-head models.
                          Ignored for single-head models.
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate that config is a MultiHeadModelConfig
        if not isinstance(config, MultiHeadModelConfig):
            raise ValueError(f"config must be MultiHeadModelConfig, got {type(config)}")
        
        # Call parent constructor
        super().__init__(config)
        
        # Store unified output flag (only meaningful for multi-head models)
        self._unified_output = unified_output and len(config.head_configs) > 1
        
        # Store separable weights flag
        self._separable_weights = config.separable_weights
        
        # Cache for backbone and head models (for separable weights feature)
        self._backbone_model: Optional[tf.keras.Model] = None
        self._head_models: Dict[str, tf.keras.Model] = {}
    
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
    
    @property
    def is_separable(self) -> bool:
        """Check if model supports separable weight operations.
        
        Returns:
            True if separable_weights is enabled
        """
        return self._separable_weights
    
    @property
    def backbone(self) -> tf.keras.Model:
        """Get the backbone model for direct access.
        
        This allows setting trainable status:
            architecture.backbone.trainable = False
        
        Returns:
            The backbone model
            
        Raises:
            ValueError: If separable_weights is not enabled
        """
        if not self._separable_weights:
            raise ValueError(
                "backbone property requires separable_weights=True. "
                "Recreate model with separable_weights=True in config."
            )
        
        if self._backbone_model is None:
            self._backbone_model = self.build_backbone()
        
        return self._backbone_model
    
    def head(self, name: str) -> tf.keras.Model:
        """Get a specific head model for direct access.
        
        This allows setting trainable status:
            architecture.head('person_detection').trainable = False
        
        Args:
            name: Name of the head
            
        Returns:
            The head model (submodel containing head layers)
            
        Raises:
            ValueError: If separable_weights is not enabled or head not found
        """
        if not self._separable_weights:
            raise ValueError(
                "head() method requires separable_weights=True. "
                "Recreate model with separable_weights=True in config."
            )
        
        if name not in self.head_names:
            raise ValueError(f"Head '{name}' not found. Available heads: {self.head_names}")
        
        # Build head model if not cached
        if name not in self._head_models:
            self._build_head_model(name)
        
        return self._head_models[name]
    
    def _build_head_model(self, head_name: str) -> tf.keras.Model:
        """Build a standalone head model for the given head.
        
        Args:
            head_name: Name of the head
            
        Returns:
            Standalone head model
        """
        model = self.get_model()
        
        # Find layers belonging to this head
        head_layer_names = [
            f"{head_name}_global_pool",
            f"{head_name}_dropout",
            f"{head_name}_output"
        ]
        
        head_layers = []
        for layer_name in head_layer_names:
            try:
                layer = model.get_layer(layer_name)
                head_layers.append(layer)
            except ValueError:
                pass  # Layer not found
        
        if not head_layers:
            raise ValueError(f"No layers found for head '{head_name}'")
        
        # Create a simple container model for the head layers
        # This allows setting trainable on all head layers at once
        head_config = self.get_head_config_by_name(head_name)
        
        # Get feature shape from backbone
        backbone = self.backbone
        dummy_input = tf.zeros((1, *self.config.input_shape))
        backbone_output_shape = backbone(dummy_input).shape[1:]
        
        # Build standalone head model
        feature_input = layers.Input(shape=backbone_output_shape, name=f'{head_name}_feature_input')
        x = layers.GlobalAveragePooling2D(name=f"{head_name}_gap")(feature_input)
        x = layers.Dropout(head_config.dropout_rate, name=f"{head_name}_drop")(x)
        output = layers.Dense(
            head_config.num_classes,
            activation=head_config.activation,
            name=f"{head_name}_dense"
        )(x)
        
        head_model = Model(inputs=feature_input, outputs=output, name=f"{head_name}_head")
        
        # Copy weights from main model
        main_model = self.get_model()
        for src_name, dst_name in [
            (f"{head_name}_global_pool", f"{head_name}_gap"),
            (f"{head_name}_dropout", f"{head_name}_drop"),
            (f"{head_name}_output", f"{head_name}_dense")
        ]:
            try:
                src_layer = main_model.get_layer(src_name)
                dst_layer = head_model.get_layer(dst_name)
                if src_layer.get_weights():
                    dst_layer.set_weights(src_layer.get_weights())
            except (ValueError, Exception):
                pass  # Layer not found or no weights
        
        self._head_models[head_name] = head_model
        return head_model
    
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
        """Build a head using the registered head builder for the specified type.
        
        This method dispatches to the appropriate head builder based on the
        head_config.head_type field. Supports extensible head architectures
        for different task types (classification, regression, embedding, etc.).
        
        Args:
            head_config: Configuration for the head
            backbone_output: Output tensor from the backbone
            
        Returns:
            Head output tensor
            
        Raises:
            ValueError: If head_config.head_type is not registered
        """
        from ..components.head_builders import build_head_for_type
        return build_head_for_type(head_config.head_type, head_config, backbone_output)
    
    def build_model(self) -> tf.keras.Model:
        """Build and return the complete multi-head TensorFlow model.
        
        This method creates the complete multi-head model by:
        1. Building the shared backbone
        2. Creating multiple classification heads
        3. Connecting heads to backbone outputs
        4. Optionally creating a unified concatenated output
        5. Creating a model with multiple outputs
        
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
        head_outputs_list = []
        for head_config in self.head_configs:
            head_output = self.build_head(head_config, backbone_output)
            outputs[head_config.name] = head_output
            head_outputs_list.append(head_output)
        
        # Add unified output if requested and model has multiple heads
        if self._unified_output:
            # Concatenate all head outputs in order
            unified_heads = layers.Concatenate(name='unified_heads')(head_outputs_list)
            outputs['unified_heads'] = unified_heads
        
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
    
    # ==================== Separable Weights Methods ====================
    
    def _check_separable(self, method_name: str) -> None:
        """Check if separable_weights is enabled.
        
        Args:
            method_name: Name of the calling method for error message
            
        Raises:
            ValueError: If separable_weights is not enabled
        """
        if not self._separable_weights:
            raise ValueError(
                f"{method_name}() requires separable_weights=True. "
                "Recreate model with separable_weights=True in config."
            )
    
    def save_backbone_weights(self, filepath: str) -> None:
        """Save backbone weights to a file.
        
        Args:
            filepath: Path to save weights (e.g., 'backbone_weights.h5')
            
        Raises:
            ValueError: If separable_weights is not enabled
        """
        self._check_separable('save_backbone_weights')
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Get or build backbone
        backbone = self.backbone
        backbone.save_weights(filepath)
        print(f"Backbone weights saved to: {filepath}")
    
    def load_backbone_weights(self, filepath: str) -> None:
        """Load backbone weights from a file.
        
        Args:
            filepath: Path to load weights from
            
        Raises:
            ValueError: If separable_weights is not enabled or file not found
        """
        self._check_separable('load_backbone_weights')
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Backbone weights file not found: {filepath}")
        
        # Get or build backbone
        backbone = self.backbone
        backbone.load_weights(filepath)
        
        # Transfer weights to the full model
        model = self.get_model()
        for layer in model.layers:
            # Find the nested backbone model in the full model
            if hasattr(layer, 'layers'):  # This is the backbone functional model
                for backbone_layer in layer.layers:
                    try:
                        src_layer = backbone.get_layer(backbone_layer.name)
                        if src_layer.get_weights():
                            backbone_layer.set_weights(src_layer.get_weights())
                    except (ValueError, Exception):
                        pass
        
        print(f"Backbone weights loaded from: {filepath}")
    
    def save_head_weights(self, head_name: str, filepath: str) -> None:
        """Save weights for a specific head to a file.
        
        Args:
            head_name: Name of the head to save
            filepath: Path to save weights (e.g., 'head_person_weights.h5')
            
        Raises:
            ValueError: If separable_weights is not enabled or head not found
        """
        self._check_separable('save_head_weights')
        
        if head_name not in self.head_names:
            raise ValueError(f"Head '{head_name}' not found. Available heads: {self.head_names}")
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Get or build head model
        head_model = self.head(head_name)
        head_model.save_weights(filepath)
        print(f"Head '{head_name}' weights saved to: {filepath}")
    
    def load_head_weights(self, head_name: str, filepath: str) -> None:
        """Load weights for a specific head from a file.
        
        Args:
            head_name: Name of the head to load
            filepath: Path to load weights from
            
        Raises:
            ValueError: If separable_weights is not enabled, head not found, or file not found
        """
        self._check_separable('load_head_weights')
        
        if head_name not in self.head_names:
            raise ValueError(f"Head '{head_name}' not found. Available heads: {self.head_names}")
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Head weights file not found: {filepath}")
        
        # Get or build head model
        head_model = self.head(head_name)
        head_model.load_weights(filepath)
        
        # Transfer weights to the full model
        model = self.get_model()
        for src_name, dst_name in [
            (f"{head_name}_gap", f"{head_name}_global_pool"),
            (f"{head_name}_drop", f"{head_name}_dropout"),
            (f"{head_name}_dense", f"{head_name}_output")
        ]:
            try:
                src_layer = head_model.get_layer(src_name)
                dst_layer = model.get_layer(dst_name)
                if src_layer.get_weights():
                    dst_layer.set_weights(src_layer.get_weights())
            except (ValueError, Exception):
                pass
        
        print(f"Head '{head_name}' weights loaded from: {filepath}")
    
    def save_all_weights_separately(self, output_dir: str) -> Dict[str, str]:
        """Save backbone and all heads to separate files.
        
        Args:
            output_dir: Directory to save weight files
            
        Returns:
            Dictionary mapping component names to file paths
            
        Raises:
            ValueError: If separable_weights is not enabled
        """
        self._check_separable('save_all_weights_separately')
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save backbone
        backbone_path = str(output_path / "backbone_weights.h5")
        self.save_backbone_weights(backbone_path)
        saved_files['backbone'] = backbone_path
        
        # Save each head
        for head_name in self.head_names:
            head_path = str(output_path / f"{head_name}_weights.h5")
            self.save_head_weights(head_name, head_path)
            saved_files[head_name] = head_path
        
        print(f"All weights saved to: {output_dir}")
        return saved_files
    
    def load_all_weights_separately(self, input_dir: str) -> None:
        """Load backbone and all heads from separate files.
        
        Args:
            input_dir: Directory containing weight files
            
        Raises:
            ValueError: If separable_weights is not enabled
        """
        self._check_separable('load_all_weights_separately')
        
        input_path = Path(input_dir)
        
        # Load backbone
        backbone_path = input_path / "backbone_weights.h5"
        if backbone_path.exists():
            self.load_backbone_weights(str(backbone_path))
        else:
            print(f"Warning: Backbone weights not found at {backbone_path}")
        
        # Load each head
        for head_name in self.head_names:
            head_path = input_path / f"{head_name}_weights.h5"
            if head_path.exists():
                self.load_head_weights(head_name, str(head_path))
            else:
                print(f"Warning: Head '{head_name}' weights not found at {head_path}")
        
        print(f"All weights loaded from: {input_dir}")
    
    def add_head_dynamically(
        self,
        num_classes: int,
        head_name: str,
        activation: str = 'linear',
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False
    ) -> tf.keras.Model:
        """Add a new head to the existing model dynamically.
        
        This method creates a new head and adds it to the model without
        retraining the backbone. Optionally freeze the backbone to train
        only the new head.
        
        Args:
            num_classes: Number of classes for the new head
            head_name: Name for the new head
            activation: Activation function ('linear' for Vela compatibility)
            dropout_rate: Dropout rate for the head
            freeze_backbone: If True, freeze backbone weights
            
        Returns:
            Updated model with the new head
            
        Raises:
            ValueError: If separable_weights is not enabled or head name exists
        """
        self._check_separable('add_head_dynamically')
        
        if head_name in self.head_names:
            raise ValueError(f"Head '{head_name}' already exists. Choose a different name.")
        
        # Create new head configuration
        new_head_config = HeadConfiguration(
            name=head_name,
            num_classes=num_classes,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        # Get current model
        model = self.get_model()
        
        # Get backbone
        backbone = self.backbone
        
        # Freeze backbone if requested
        if freeze_backbone:
            backbone.trainable = False
            print(f"Backbone frozen for training new head '{head_name}'")
        
        # Get backbone output from the current model
        backbone_output = backbone(model.input)
        
        # Build new head
        new_head_output = self.build_head(new_head_config, backbone_output)
        
        # Collect all outputs (existing + new)
        new_outputs = {}
        for output_name in model.output_names:
            if output_name != 'unified_heads':  # Skip unified output, will recreate if needed
                new_outputs[output_name] = model.output[output_name]
        new_outputs[head_name] = new_head_output
        
        # Add unified output if it was enabled
        if self._unified_output:
            head_outputs_list = [new_outputs[name] for name in self.head_names + [head_name]]
            unified_heads = layers.Concatenate(name='unified_heads')(head_outputs_list)
            new_outputs['unified_heads'] = unified_heads
        
        # Create new model
        new_model = Model(inputs=model.input, outputs=new_outputs, name=self.name)
        
        # Update configuration
        self.config.head_configs.append(new_head_config)
        
        # Clear cached model to force rebuild
        self._model = new_model
        
        # Build head model for the new head
        self._build_head_model(head_name)
        
        print(f"New head '{head_name}' added with {num_classes} classes")
        if freeze_backbone:
            print("  Backbone is frozen. Only new head will be trained.")
        
        return new_model
    
    def freeze_backbone(self) -> None:
        """Freeze backbone weights (make non-trainable).
        
        Raises:
            ValueError: If separable_weights is not enabled
        """
        self._check_separable('freeze_backbone')
        self.backbone.trainable = False
        print("Backbone frozen (trainable=False)")
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone weights (make trainable).
        
        Raises:
            ValueError: If separable_weights is not enabled
        """
        self._check_separable('unfreeze_backbone')
        self.backbone.trainable = True
        print("Backbone unfrozen (trainable=True)")
    
    def freeze_head(self, head_name: str) -> None:
        """Freeze a specific head's weights (make non-trainable).
        
        Args:
            head_name: Name of the head to freeze
            
        Raises:
            ValueError: If separable_weights is not enabled or head not found
        """
        self._check_separable('freeze_head')
        self.head(head_name).trainable = False
        print(f"Head '{head_name}' frozen (trainable=False)")
    
    def unfreeze_head(self, head_name: str) -> None:
        """Unfreeze a specific head's weights (make trainable).
        
        Args:
            head_name: Name of the head to unfreeze
            
        Raises:
            ValueError: If separable_weights is not enabled or head not found
        """
        self._check_separable('unfreeze_head')
        self.head(head_name).trainable = True
        print(f"Head '{head_name}' unfrozen (trainable=True)")
    
    def get_trainable_status(self) -> Dict[str, bool]:
        """Get trainable status of backbone and all heads.
        
        Returns:
            Dictionary mapping component names to trainable status
            
        Raises:
            ValueError: If separable_weights is not enabled
        """
        self._check_separable('get_trainable_status')
        
        status = {'backbone': self.backbone.trainable}
        for head_name in self.head_names:
            status[head_name] = self.head(head_name).trainable
        
        return status
    
    # ==================== End Separable Weights Methods ====================
    
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
