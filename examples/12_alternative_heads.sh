#!/bin/bash
# Example 12: Alternative Head Types
# Demonstrates different head architectures for various task types

echo "========================================="
echo "Example 12: Alternative Head Types"
echo "========================================="
echo "Creating models with different head types..."
echo ""

# Create output directory
mkdir -p ../output/examples/alternative_heads

echo "1. Multi-label classification model..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import create_multilabel_head, create_classification_head
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture
from src.utils import quantize_to_tflite

# Create multi-label head
multilabel_head = create_multilabel_head('image_tags', num_labels=5, dropout_rate=0.3)
binary_head = create_classification_head('person_present', num_classes=2, activation='linear')

config = MultiHeadModelConfig(
    input_shape=(128, 128, 3),
    head_configs=[multilabel_head, binary_head],
    arch_params={'alpha': 0.25}
)

arch = MultiHeadMobileNetV3QATArchitecture(config)
model = arch.get_model()

print(f'Model created with {len(arch.head_configs)} heads:')
for head in arch.head_configs:
    print(f'  - {head.name}: {head.head_type} ({head.num_classes} classes)')

# Export to TFLite
tflite_model, _ = quantize_to_tflite(
    model, (128, 128, 3), calibration_samples=10,
    output_path='../output/examples/alternative_heads/multilabel_model_int8.tflite'
)
print('Multi-label model exported successfully!')
"

echo ""
echo "2. Regression model..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import create_regression_head
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture
from src.utils import quantize_to_tflite

# Create regression heads
age_head = create_regression_head('age_years', n_outputs=1, output_scale=100.0, dropout_rate=0.2)
blur_head = create_regression_head('blur_score', n_outputs=1, output_activation='sigmoid', dropout_rate=0.2)
pose_head = create_regression_head('head_pose', n_outputs=3, dropout_rate=0.3)  # yaw, pitch, roll

config = MultiHeadModelConfig(
    input_shape=(96, 96, 3),
    head_configs=[age_head, blur_head, pose_head],
    arch_params={'alpha': 0.25}
)

arch = MultiHeadMobileNetV3QATArchitecture(config)
model = arch.get_model()

print(f'Regression model created with {len(arch.head_configs)} heads:')
for head in arch.head_configs:
    n_outputs = head.custom_params.get('n_outputs', head.num_classes)
    print(f'  - {head.name}: {head.head_type} ({n_outputs} outputs)')

# Export to TFLite
tflite_model, _ = quantize_to_tflite(
    model, (96, 96, 3), calibration_samples=10,
    output_path='../output/examples/alternative_heads/regression_model_int8.tflite'
)
print('Regression model exported successfully!')
"

echo ""
echo "3. Embedding model..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import create_embedding_head, create_classification_head
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture
from src.utils import quantize_to_tflite

# Create embedding head
face_embed = create_embedding_head(
    'face_embedding', 
    embed_dim=256, 
    projection_layers=[512],  # MLP projection
    use_bn=True,
    dropout_rate=0.1
)
id_head = create_classification_head('identity', num_classes=100, activation='linear')

config = MultiHeadModelConfig(
    input_shape=(112, 112, 3),
    head_configs=[face_embed, id_head],
    arch_params={'alpha': 0.5}
)

arch = MultiHeadMobileNetV3QATArchitecture(config)
model = arch.get_model()

print(f'Embedding model created with {len(arch.head_configs)} heads:')
for head in arch.head_configs:
    if head.head_type == 'embedding':
        embed_dim = head.custom_params.get('embed_dim', 128)
        print(f'  - {head.name}: {head.head_type} ({embed_dim}D embedding)')
    else:
        print(f'  - {head.name}: {head.head_type} ({head.num_classes} classes)')

# Export to TFLite
tflite_model, _ = quantize_to_tflite(
    model, (112, 112, 3), calibration_samples=10,
    output_path='../output/examples/alternative_heads/embedding_model_int8.tflite'
)
print('Embedding model exported successfully!')
"

echo ""
echo "4. Ordinal regression model..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import create_ordinal_head
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture
from src.utils import quantize_to_tflite

# Create ordinal regression heads
age_group = create_ordinal_head('age_group', num_classes=5, threshold_init='ascending')  # child, teen, adult, middle, senior
severity = create_ordinal_head('damage_severity', num_classes=4, threshold_init='ascending')  # none, mild, moderate, severe

config = MultiHeadModelConfig(
    input_shape=(224, 224, 3),
    head_configs=[age_group, severity],
    arch_params={'alpha': 0.25}
)

arch = MultiHeadMobileNetV3QATArchitecture(config)
model = arch.get_model()

print(f'Ordinal regression model created with {len(arch.head_configs)} heads:')
for head in arch.head_configs:
    print(f'  - {head.name}: {head.head_type} ({head.num_classes} ordinal classes)')

# Export to TFLite
tflite_model, _ = quantize_to_tflite(
    model, (224, 224, 3), calibration_samples=10,
    output_path='../output/examples/alternative_heads/ordinal_model_int8.tflite'
)
print('Ordinal regression model exported successfully!')
"

echo ""
echo "All models created successfully!"
echo "Output directory: ../output/examples/alternative_heads/"
echo ""
echo "Models created:"
echo "  1. multilabel_model_int8.tflite - Multi-label + binary classification"
echo "  2. regression_model_int8.tflite - Age, blur, pose regression"
echo "  3. embedding_model_int8.tflite - Face embedding + ID classification"
echo "  4. ordinal_model_int8.tflite - Age group + damage severity (ordinal)"
echo ""
echo "Head types demonstrated:"
echo "  - standard: Traditional multi-class classification"
echo "  - multilabel: Multiple independent binary classifications"
echo "  - regression: Continuous value prediction (bounded and unbounded)"
echo "  - embedding: L2-normalized feature vectors for similarity"
echo "  - ordinal: Ordered categorical classification (CORAL)"
echo ""
echo "Next steps:"
echo "  - Use get_losses_for_heads() to get appropriate loss functions"
echo "  - Use get_metrics_for_heads() to get suitable metrics"
echo "  - Train with task-specific loss functions and data formats"
echo ""