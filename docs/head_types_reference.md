# Head Types Reference

This document describes all available head types in the multi-head MobileNet architecture and how to use them for different tasks.

## Overview

The multi-head architecture supports extensible head types through a registry system. Each head type is optimized for specific task categories and automatically handles appropriate loss functions and metrics.

## Available Head Types

### Classification & Recognition Heads

#### 1. Standard Classification (`standard`)

**Purpose**: Traditional multi-class classification where each sample belongs to exactly one class.

**Architecture**: GlobalAveragePooling2D → Dropout → Dense(num_classes, activation)

**Use Cases**:
- Object classification (cat, dog, car, etc.)
- Scene recognition (indoor, outdoor, urban, etc.)
- Quality assessment (good, fair, poor)

**Configuration**:
```python
from models.components.head_configuration import create_classification_head

head = create_classification_head(
    name="scene_type",
    num_classes=5,
    activation="linear",  # or "softmax"
    dropout_rate=0.2
)
```

**Training**:
- **Loss**: `SparseCategoricalCrossentropy(from_logits=True)`
- **Metrics**: `accuracy`, `sparse_top_k_categorical_accuracy`
- **Label format**: Integer class indices (0, 1, 2, ...)

### 2. Multi-Label Classification (`multilabel`)

**Purpose**: Multiple independent binary classifications where samples can have multiple active labels simultaneously.

**Architecture**: GlobalAveragePooling2D → Dropout → Dense(num_labels, activation='sigmoid')

**Use Cases**:
- Image tagging (indoor + person + furniture)
- Medical diagnosis (multiple conditions present)
- Content moderation (multiple policy violations)

**Configuration**:
```python
from models.components.head_configuration import create_multilabel_head

head = create_multilabel_head(
    name="image_tags",
    num_labels=8,
    dropout_rate=0.3
)
```

**Training**:
- **Loss**: `BinaryCrossentropy(from_logits=False)`
- **Metrics**: `binary_accuracy`, `precision`, `recall`
- **Label format**: Binary array (e.g., [1, 0, 1, 0, 1] for 5 labels)

### 3. Regression (`regression`)

**Purpose**: Predicting continuous numerical values (single or multiple outputs).

**Architecture**: GlobalAveragePooling2D → Dropout → Dense(n_outputs, activation='linear') → Optional scaling/bias

**Use Cases**:
- Age estimation (years)
- Pose estimation (yaw, pitch, roll angles)
- Quality scores (0.0 to 1.0)
- Bounding box coordinates

**Configuration**:
```python
from models.components.head_configuration import create_regression_head

# Single output regression
age_head = create_regression_head(
    name="age_years",
    n_outputs=1,
    output_scale=100.0,  # Scale to 0-100 years
    dropout_rate=0.2
)

# Multi-output regression with bounded activation
pose_head = create_regression_head(
    name="head_pose",
    n_outputs=3,  # yaw, pitch, roll
    output_activation="tanh",  # Bound to [-1, 1]
    dropout_rate=0.1
)

# Quality score (0-1 range)
quality_head = create_regression_head(
    name="blur_score",
    n_outputs=1,
    output_activation="sigmoid",  # Bound to [0, 1]
)
```

**Training**:
- **Loss**: `MeanSquaredError`, `MeanAbsoluteError`, or `Huber`
- **Metrics**: `mae`, `mse`
- **Label format**: Float arrays matching n_outputs

### 4. Embedding (`embedding`)

**Purpose**: Learning feature representations for similarity matching, clustering, or retrieval.

**Architecture**: GlobalAveragePooling2D → Dropout → Optional MLP → Dense(embed_dim) → BatchNorm (optional) → L2Normalize

**Use Cases**:
- Face recognition (face embeddings)
- Image retrieval (content-based search)
- Person re-identification
- Product similarity

**Configuration**:
```python
from models.components.head_configuration import create_embedding_head

head = create_embedding_head(
    name="face_embedding",
    embed_dim=256,
    projection_layers=[512, 256],  # Optional MLP projection
    use_bn=True,  # Batch normalization before L2 norm
    dropout_rate=0.1
)
```

**Training**:
- **Loss**: `CosineSimilarity`, `TripletLoss`, or custom metric learning losses
- **Metrics**: `cosine_similarity`
- **Label format**: Identity labels for supervised learning, or pairs/triplets for metric learning

**Output**: L2-normalized vectors suitable for cosine similarity computation.

### 5. Ordinal Regression (`ordinal`)

**Purpose**: Ordered categorical classification where class order matters (rating scales, severity levels).

**Architecture**: GlobalAveragePooling2D → Dropout → Dense(num_classes-1, activation='sigmoid') with ordered bias initialization

**Method**: CORAL (Consistent Rank Logits) - uses K-1 binary threshold classifiers for K ordinal classes.

**Use Cases**:
- Age groups (child, teen, adult, senior)
- Damage severity (none, mild, moderate, severe)
- Quality ratings (1-star to 5-star)
- Disease staging

**Configuration**:
```python
from models.components.head_configuration import create_ordinal_head

head = create_ordinal_head(
    name="age_group",
    num_classes=5,  # 5 ordinal classes
    threshold_init="ascending",  # Initialize thresholds in order
    dropout_rate=0.2
)
```

**Training**:
- **Loss**: `OrdinalCrossEntropy` (custom loss for CORAL)
- **Metrics**: `ordinal_accuracy`, `ordinal_mae`
- **Label format**: Integer ordinal class indices (0, 1, 2, ...)

**Prediction**: Predicted class = sum(sigmoid_outputs > 0.5)

### Detection & Localization Heads

#### 6. SSD Detection (`ssd_detection`)

**Purpose**: Object detection using Single Shot MultiBox Detector architecture for finding and localizing objects.

**Architecture**: Conv2D layers → Classification branch + Localization branch → Multi-scale anchor predictions

**Use Cases**:
- Face detection (find faces in images)
- Object detection (cars, people, etc.)
- Logo detection
- General bounding box regression

**Configuration**:
```python
from models.components.head_configuration import create_ssd_detection_head

# Face detection
face_head = create_ssd_detection_head(
    name="face_detector",
    num_classes=1,  # Binary: face/no-face (+ background)
    num_anchors=3,
    anchor_scales=[0.1, 0.2, 0.37],
    anchor_ratios=[0.5, 1.0, 2.0],
    dropout_rate=0.1
)

# Multi-class object detection
object_head = create_ssd_detection_head(
    name="object_detector", 
    num_classes=20,  # PASCAL VOC classes
    num_anchors=6,
    dropout_rate=0.1
)
```

**Training**:
- **Loss**: `SSDLoss` (combines classification + localization)
- **Metrics**: `binary_accuracy` for classification component
- **Label format**: Dictionary with 'boxes' [N, 4] and 'labels' [N] per image

**Output**: Dictionary with:
- `scores`: [batch, num_anchors_total, num_classes] - Classification probabilities
- `boxes`: [batch, num_anchors_total, 4] - Bounding box predictions (x_center, y_center, width, height)

#### 7. YOLO Detection (`yolo_detection`)

**Purpose**: Alternative object detection using YOLO (You Only Look Once) approach.

**Architecture**: Conv2D layers → Grid-based predictions (objectness + classes + coordinates)

**Use Cases**:
- Real-time object detection
- Multi-object scenes
- Alternative to SSD with different trade-offs

**Configuration**:
```python
from models.components.head_configuration import create_yolo_detection_head

head = create_yolo_detection_head(
    name="yolo_detector",
    num_classes=80,  # COCO classes
    num_boxes=3,
    coord_scale=1.0,
    dropout_rate=0.1
)
```

**Training**:
- **Loss**: `YOLOLoss` (objectness + classification + localization)
- **Metrics**: `binary_accuracy` for objectness component
- **Label format**: Grid-based format [batch, grid_h, grid_w, boxes * (5 + classes)]

**Output**: YOLO prediction tensor with objectness, coordinates, and class probabilities per grid cell.

### Dense Prediction Heads

#### 8. Semantic Segmentation (`segmentation`)

**Purpose**: Pixel-wise classification for semantic understanding of image regions.

**Architecture**: Progressive upsampling with Conv2DTranspose → Final 1x1 classification

**Use Cases**:
- Face parsing (skin, hair, eyes, background)
- Scene segmentation (road, sidewalk, buildings)
- Medical image segmentation
- Background removal

**Configuration**:
```python
from models.components.head_configuration import create_segmentation_head

# Face parsing
face_seg = create_segmentation_head(
    name="face_parsing",
    num_classes=7,  # background, skin, hair, eyes, nose, mouth, other
    upsample_factor=8,
    intermediate_channels=[256, 128],
    dropout_rate=0.2
)

# Scene segmentation  
scene_seg = create_segmentation_head(
    name="scene_segmentation",
    num_classes=21,  # PASCAL VOC segmentation
    upsample_factor=16,
    dropout_rate=0.3
)
```

**Training**:
- **Loss**: `SparseCategoricalCrossentropy` (pixel-wise classification)
- **Metrics**: `accuracy`, `sparse_categorical_accuracy`
- **Label format**: Pixel-wise class labels [batch, height, width] 

**Output**: Segmentation logits [batch, height, width, num_classes] at input resolution.

#### 9. Keypoint Detection (`keypoint_detection`)

**Purpose**: Detecting and localizing specific keypoints using heatmap regression.

**Architecture**: Feature processing → Upsampling → Heatmap generation (one per keypoint)

**Use Cases**:
- Facial landmark detection (68 points)
- Human pose estimation (17 body joints)
- Hand keypoints (21 hand joints)
- Object keypoints (corners, features)

**Configuration**:
```python
from models.components.head_configuration import create_keypoint_detection_head

# Facial landmarks
landmarks = create_keypoint_detection_head(
    name="face_landmarks",
    num_keypoints=68,
    upsample_factor=4,
    heatmap_sigma=1.0,
    intermediate_dim=256,
    dropout_rate=0.1
)

# Human pose
pose = create_keypoint_detection_head(
    name="body_pose",
    num_keypoints=17,  # COCO pose
    upsample_factor=4,
    dropout_rate=0.1
)
```

**Training**:
- **Loss**: `MeanSquaredError` (heatmap regression)
- **Metrics**: `mse`, `mae` 
- **Label format**: Ground truth heatmaps [batch, height, width, num_keypoints]

**Output**: Keypoint heatmaps [batch, height, width, num_keypoints] where each channel represents one keypoint.

## Using Head Types

### Creating Models with Different Head Types

```python
from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import *
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture

# Create different head types
heads = [
    create_classification_head("object_type", num_classes=10),
    create_multilabel_head("attributes", num_labels=5),
    create_regression_head("age", n_outputs=1, output_scale=100),
    create_embedding_head("identity", embed_dim=128),
    create_ordinal_head("quality", num_classes=4),
    create_ssd_detection_head("face_detector", num_classes=1),
    create_segmentation_head("face_parsing", num_classes=7),
    create_keypoint_detection_head("landmarks", num_keypoints=68)
]

config = MultiHeadModelConfig(
    input_shape=(224, 224, 3),
    head_configs=heads,
    arch_params={'alpha': 0.75, 'use_pretrained': True}
)

arch = MultiHeadMobileNetV3QATArchitecture(config)
model = arch.get_model()
```

### Complete Face Analysis Example

```python
# Face detection + recognition + analysis system
face_system_heads = [
    # Detection: Find faces in images
    create_ssd_detection_head("face_detector", num_classes=1),
    
    # Recognition: Identity embeddings for cropped faces
    create_embedding_head("face_identity", embed_dim=512, projection_layers=[1024, 512]),
    
    # Parsing: Segment face regions
    create_segmentation_head("face_parsing", num_classes=7, upsample_factor=8),
    
    # Landmarks: 68 facial keypoints
    create_keypoint_detection_head("landmarks", num_keypoints=68),
    
    # Attributes: Age, gender, emotion
    create_classification_head("age_group", num_classes=5),
    create_classification_head("gender", num_classes=2),
    create_multilabel_head("emotions", num_labels=7)
]

# This creates a complete face analysis pipeline in a single model
```

### Automatic Loss and Metrics Selection

```python
from src.utils import get_losses_for_heads, get_metrics_for_heads

# Get appropriate losses for each head type
losses = get_losses_for_heads(heads, from_logits=True)

# Get suitable metrics for each head type
metrics = get_metrics_for_heads(heads)

# Compile model
model.compile(
    optimizer='adam',
    loss=losses,
    metrics=metrics
)
```

### Custom Loss Overrides

```python
from src.utils import SSDLoss, YOLOLoss

# Override default losses for specific heads
loss_overrides = {
    'age': {'loss_type': 'mae'},  # Use MAE instead of MSE for regression
    'identity': tf.keras.losses.TripletSemiHardLoss(),  # Custom triplet loss
    'face_detector': SSDLoss(alpha=1.0, neg_pos_ratio=3.0),  # Custom SSD loss
    'object_detector': YOLOLoss(lambda_coord=5.0, lambda_noobj=0.5)  # Custom YOLO loss
}

losses = get_losses_for_heads(heads, loss_overrides=loss_overrides)
```

### Detection Head Training Considerations

Detection heads (SSD, YOLO) require special handling:

```python
# Detection heads return dictionaries, need custom training
def train_with_detection_heads(model, dataset):
    for batch_images, batch_labels in dataset:
        with tf.GradientTape() as tape:
            predictions = model(batch_images, training=True)
            
            # Handle different output types
            losses = {}
            for head_name, pred in predictions.items():
                if isinstance(pred, dict):  # Detection head
                    # pred contains 'boxes' and 'scores'
                    losses[head_name] = ssd_loss_fn(batch_labels[head_name], pred)
                else:  # Regular head
                    losses[head_name] = standard_loss_fn(batch_labels[head_name], pred)
            
            total_loss = sum(losses.values())
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

## Extending with Custom Head Types

You can register custom head builders using the `@register_head` decorator:

```python
from models.components.head_builders import register_head
from tensorflow.keras import layers

@register_head("attention_pooling")
def build_attention_pooling_head(head_config, backbone_output):
    # Custom attention-based pooling instead of GAP
    B, H, W, C = backbone_output.shape
    
    # Attention mechanism
    attention = layers.Conv2D(1, 1, activation='sigmoid')(backbone_output)
    attended = backbone_output * attention
    
    # Weighted average pooling
    pooled = layers.GlobalAveragePooling2D()(attended)
    
    if head_config.dropout_rate > 0:
        pooled = layers.Dropout(head_config.dropout_rate)(pooled)
    
    return layers.Dense(
        head_config.num_classes,
        activation=head_config.activation
    )(pooled)
```

Then use it in head configuration:

```python
head = HeadConfiguration(
    name="attended_classifier",
    num_classes=100,
    head_type="attention_pooling",
    dropout_rate=0.2
)
```

## Best Practices

1. **Choose the right head type**: Match the head type to your task requirements
2. **Proper activation**: Use 'linear' for Vela compatibility, apply softmax in post-processing
3. **Loss function alignment**: Use `get_losses_for_heads()` for automatic selection
4. **Data format**: Ensure your training data matches the expected format for each head type
5. **Balanced learning**: Consider loss weights when combining different head types
6. **Regularization**: Adjust dropout rates based on head complexity and data size

## Performance Considerations

### Computational Complexity
- **Standard/Multi-label/Ordinal**: Lightweight, GAP + Dense layers
- **Regression**: Most efficient, linear outputs only
- **Embedding**: L2 normalization adds minimal overhead
- **SSD/YOLO Detection**: Moderate overhead, multiple conv layers + anchors
- **Segmentation**: Higher memory usage, upsampling to full resolution
- **Keypoint Detection**: Similar to segmentation, heatmap generation

### Memory Usage
- **Classification heads**: Minimal memory (GAP reduces spatial dimensions)
- **Detection heads**: Moderate (anchor predictions across feature map)
- **Dense prediction heads**: High (full-resolution outputs)

### TFLite Optimization
- **Most efficient**: Standard, multilabel, regression, embedding, ordinal
- **Moderate**: SSD/YOLO detection (conv layers supported)
- **Requires careful optimization**: Segmentation, keypoint (upsampling layers)

## Quantization Compatibility

All head types are compatible with:
- **Int8 quantization**: Full post-training quantization
- **Mixed precision**: Int8 backbone + FP16/FP32 heads via `--export-separate-tflite`
- **Vela compiler**: Use 'linear' activation for NPU compatibility

## Examples

See the following files for complete examples:
- `examples/12_alternative_heads.sh` - CLI examples for classification head types
- `examples/demo_alternative_heads.py` - Python training examples (basic heads)
- `examples/demo_face_detection_recognition.py` - Face detection + recognition system
- `docs/training_guide.md` - Integration with training workflows

## Summary of All Head Types

| Head Type | Purpose | Output Format | Use Cases |
|-----------|---------|---------------|-----------|
| `standard` | Multi-class classification | Class logits | Object classification, scene recognition |
| `multilabel` | Multi-label classification | Binary probabilities | Image tagging, multi-attribute prediction |
| `regression` | Continuous values | Real numbers | Age estimation, pose angles, quality scores |
| `embedding` | Feature representations | L2-normalized vectors | Face recognition, image retrieval |
| `ordinal` | Ordered categories | Threshold probabilities | Ratings, severity levels, age groups |
| `ssd_detection` | Object detection | Boxes + scores dict | Face detection, object localization |
| `yolo_detection` | Alternative detection | Grid predictions | Real-time object detection |
| `segmentation` | Pixel-wise classification | Full-resolution masks | Face parsing, scene segmentation |
| `keypoint_detection` | Point localization | Heatmaps | Facial landmarks, pose estimation |

The system now supports the complete spectrum of computer vision tasks from simple classification to complex dense prediction problems!