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
        
        valid_head_types = {'standard', 'custom', 'multilabel', 'regression', 'embedding', 'ordinal', 
                           'ssd_detection', 'yolo_detection', 'segmentation', 'keypoint_detection',
                           'text_detection', 'text_recognition', 'scene_text'}
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
                                head_names: Optional[List[str]] = None,
                                head_type: str = "standard") -> List[HeadConfiguration]:
    """Create head configurations from a list of class counts.
    
    This utility function creates HeadConfiguration objects from a simple list
    of class counts, which is useful for quick configuration.
    
    Args:
        head_config_list: List of class counts (e.g., [2, 2, 5])
        head_names: Optional list of head names. If None, auto-generates names
        head_type: Head type for all heads (default: "standard")
        
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
            num_classes=num_classes,
            head_type=head_type
        )
        heads.append(head_config)
    
    return heads


def create_classification_head(name: str, num_classes: int, activation: str = "linear", 
                             dropout_rate: float = 0.2) -> HeadConfiguration:
    """Create a standard classification head configuration.
    
    Args:
        name: Head name
        num_classes: Number of output classes
        activation: Output activation ('linear', 'softmax')
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for classification
    """
    return HeadConfiguration(
        name=name,
        num_classes=num_classes,
        activation=activation,
        dropout_rate=dropout_rate,
        head_type="standard"
    )


def create_multilabel_head(name: str, num_labels: int, dropout_rate: float = 0.2) -> HeadConfiguration:
    """Create a multi-label classification head configuration.
    
    Args:
        name: Head name
        num_labels: Number of binary labels
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for multi-label classification
    """
    return HeadConfiguration(
        name=name,
        num_classes=num_labels,
        activation="sigmoid",  # Will be overridden by builder
        dropout_rate=dropout_rate,
        head_type="multilabel"
    )


def create_regression_head(name: str, n_outputs: int = 1, 
                         output_activation: Optional[str] = None,
                         output_scale: float = 1.0, 
                         output_bias: float = 0.0,
                         dropout_rate: float = 0.2) -> HeadConfiguration:
    """Create a regression head configuration.
    
    Args:
        name: Head name
        n_outputs: Number of regression outputs
        output_activation: Output activation ('sigmoid', 'tanh', None)
        output_scale: Scale factor for outputs
        output_bias: Bias added to outputs
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for regression
    """
    custom_params = {
        "n_outputs": n_outputs,
        "output_scale": output_scale,
        "output_bias": output_bias
    }
    if output_activation:
        custom_params["output_activation"] = output_activation
    
    return HeadConfiguration(
        name=name,
        num_classes=n_outputs,  # Used as default if n_outputs not in custom_params
        activation="linear",
        dropout_rate=dropout_rate,
        head_type="regression",
        custom_params=custom_params
    )


def create_embedding_head(name: str, embed_dim: int = 128,
                        projection_layers: Optional[List[int]] = None,
                        use_bn: bool = False,
                        dropout_rate: float = 0.2) -> HeadConfiguration:
    """Create an embedding head configuration.
    
    Args:
        name: Head name
        embed_dim: Embedding dimension
        projection_layers: Hidden layer sizes for MLP projection
        use_bn: Use batch normalization before L2 normalization
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for embedding
    """
    custom_params = {
        "embed_dim": embed_dim,
        "use_bn": use_bn
    }
    if projection_layers:
        custom_params["projection_layers"] = projection_layers
    
    return HeadConfiguration(
        name=name,
        num_classes=embed_dim,  # Placeholder
        activation="linear",
        dropout_rate=dropout_rate,
        head_type="embedding",
        custom_params=custom_params
    )


def create_ordinal_head(name: str, num_classes: int,
                      threshold_init: str = "ascending",
                      dropout_rate: float = 0.2) -> HeadConfiguration:
    """Create an ordinal regression head configuration.
    
    Args:
        name: Head name
        num_classes: Number of ordinal classes (must be >= 2)
        threshold_init: Threshold initialization ('ascending', 'uniform')
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for ordinal regression
    """
    if num_classes < 2:
        raise ValueError(f"Ordinal regression requires num_classes >= 2, got {num_classes}")
    
    return HeadConfiguration(
        name=name,
        num_classes=num_classes,
        activation="sigmoid",
        dropout_rate=dropout_rate,
        head_type="ordinal",
        custom_params={"threshold_init": threshold_init}
    )


def create_ssd_detection_head(name: str, num_classes: int = 1,
                            num_anchors: int = 3,
                            anchor_scales: Optional[List[float]] = None,
                            anchor_ratios: Optional[List[float]] = None,
                            dropout_rate: float = 0.1) -> HeadConfiguration:
    """Create an SSD detection head configuration.
    
    Args:
        name: Head name
        num_classes: Number of classes (1 for face detection, >1 for multi-class)
        num_anchors: Number of anchor boxes per spatial location
        anchor_scales: Scale factors for anchors
        anchor_ratios: Aspect ratios for anchors
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for SSD detection
    """
    anchor_scales = anchor_scales or [0.1, 0.2, 0.37]
    anchor_ratios = anchor_ratios or [0.5, 1.0, 2.0]
    
    custom_params = {
        "num_anchors": num_anchors,
        "anchor_scales": anchor_scales,
        "anchor_ratios": anchor_ratios,
        "box_loss_weight": 1.0,
        "conf_threshold": 0.5
    }
    
    return HeadConfiguration(
        name=name,
        num_classes=num_classes,
        activation="linear",
        dropout_rate=dropout_rate,
        head_type="ssd_detection",
        custom_params=custom_params
    )


def create_yolo_detection_head(name: str, num_classes: int,
                             num_boxes: int = 3,
                             coord_scale: float = 1.0,
                             dropout_rate: float = 0.1) -> HeadConfiguration:
    """Create a YOLO detection head configuration.
    
    Args:
        name: Head name
        num_classes: Number of object classes
        num_boxes: Number of bounding boxes per grid cell
        coord_scale: Scaling factor for coordinates
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for YOLO detection
    """
    custom_params = {
        "num_boxes": num_boxes,
        "coord_scale": coord_scale
    }
    
    return HeadConfiguration(
        name=name,
        num_classes=num_classes,
        activation="linear",
        dropout_rate=dropout_rate,
        head_type="yolo_detection",
        custom_params=custom_params
    )


def create_segmentation_head(name: str, num_classes: int,
                           upsample_factor: int = 8,
                           intermediate_channels: Optional[List[int]] = None,
                           dropout_rate: float = 0.2) -> HeadConfiguration:
    """Create a semantic segmentation head configuration.
    
    Args:
        name: Head name
        num_classes: Number of semantic classes
        upsample_factor: Factor to upsample feature maps to input resolution
        intermediate_channels: Channels for progressive upsampling layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for semantic segmentation
    """
    intermediate_channels = intermediate_channels or [256, 128]
    
    custom_params = {
        "upsample_factor": upsample_factor,
        "intermediate_channels": intermediate_channels,
        "use_skip_connections": False
    }
    
    return HeadConfiguration(
        name=name,
        num_classes=num_classes,
        activation="linear",
        dropout_rate=dropout_rate,
        head_type="segmentation",
        custom_params=custom_params
    )


def create_keypoint_detection_head(name: str, num_keypoints: int,
                                 upsample_factor: int = 4,
                                 heatmap_sigma: float = 1.0,
                                 intermediate_dim: int = 256,
                                 dropout_rate: float = 0.1) -> HeadConfiguration:
    """Create a keypoint detection head configuration.
    
    Args:
        name: Head name
        num_keypoints: Number of keypoints to detect
        upsample_factor: Upsampling factor for heatmaps
        heatmap_sigma: Gaussian sigma for ground truth heatmaps
        intermediate_dim: Intermediate feature dimension
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for keypoint detection
    """
    custom_params = {
        "upsample_factor": upsample_factor,
        "heatmap_sigma": heatmap_sigma,
        "intermediate_dim": intermediate_dim
    }
    
    return HeadConfiguration(
        name=name,
        num_classes=num_keypoints,
        activation="sigmoid",
        dropout_rate=dropout_rate,
        head_type="keypoint_detection",
        custom_params=custom_params
    )


def create_text_detection_head(name: str,
                             detect_orientation: bool = True,
                             min_text_size: int = 8,
                             link_threshold: float = 0.4,
                             text_threshold: float = 0.7,
                             dropout_rate: float = 0.1) -> HeadConfiguration:
    """Create a text detection head configuration.
    
    Args:
        name: Head name
        detect_orientation: Whether to predict text orientation angles
        min_text_size: Minimum text region size in pixels
        link_threshold: Threshold for linking text segments
        text_threshold: Threshold for text confidence
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for text detection
    """
    custom_params = {
        "detect_orientation": detect_orientation,
        "min_text_size": min_text_size,
        "link_threshold": link_threshold,
        "text_threshold": text_threshold
    }
    
    return HeadConfiguration(
        name=name,
        num_classes=1,  # Text/no-text binary classification
        activation="sigmoid",
        dropout_rate=dropout_rate,
        head_type="text_detection",
        custom_params=custom_params
    )


def create_text_recognition_head(name: str, vocab_size: int,
                               max_text_length: int = 32,
                               use_attention: bool = False,
                               rnn_units: int = 256,
                               num_rnn_layers: int = 2,
                               dropout_rate: float = 0.2) -> HeadConfiguration:
    """Create a text recognition head configuration.
    
    Args:
        name: Head name
        vocab_size: Size of character vocabulary (including blank for CTC)
        max_text_length: Maximum text sequence length
        use_attention: Whether to use attention mechanism
        rnn_units: RNN hidden units
        num_rnn_layers: Number of RNN layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for text recognition
    """
    custom_params = {
        "max_text_length": max_text_length,
        "use_attention": use_attention,
        "rnn_units": rnn_units,
        "num_rnn_layers": num_rnn_layers
    }
    
    return HeadConfiguration(
        name=name,
        num_classes=vocab_size,
        activation="linear",  # CTC compatible
        dropout_rate=dropout_rate,
        head_type="text_recognition",
        custom_params=custom_params
    )


def create_scene_text_head(name: str, char_vocab_size: int,
                         max_detections: int = 100,
                         max_chars_per_text: int = 25,
                         detection_threshold: float = 0.5,
                         dropout_rate: float = 0.1) -> HeadConfiguration:
    """Create an end-to-end scene text reading head configuration.
    
    Args:
        name: Head name
        char_vocab_size: Character vocabulary size
        max_detections: Maximum number of text instances
        max_chars_per_text: Maximum characters per text instance
        detection_threshold: Text detection confidence threshold
        dropout_rate: Dropout rate for regularization
        
    Returns:
        HeadConfiguration for scene text reading
    """
    custom_params = {
        "max_detections": max_detections,
        "max_chars_per_text": max_chars_per_text,
        "detection_threshold": detection_threshold,
        "char_vocab_size": char_vocab_size
    }
    
    return HeadConfiguration(
        name=name,
        num_classes=char_vocab_size,
        activation="linear",
        dropout_rate=dropout_rate,
        head_type="scene_text",
        custom_params=custom_params
    )
