# Examples

Real-world examples of creating quantized MobileNet V3 models for different use cases.

## Example 1: Person Detection (Single Head)

Simple binary classification - detect if a person is present in an image.

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "2" \
    --output-dir ./models/person_detection
```

This creates a small, fast model suitable for edge devices. The output has 2 classes: person present / no person.

## Example 2: Person Detection + Gender Classification

Multi-task learning: detect person and classify gender simultaneously.

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "2,2" \
    --head-names "person_detection,gender" \
    --output-dir ./models/person_gender
```

The model outputs:
- `person_detection`: 2 classes (person/no-person)
- `gender`: 2 classes (male/female)

Both predictions share the same backbone features, making this more efficient than separate models.

## Example 3: Object Classification + Person Detection + Age Group

Three-head model for comprehensive scene understanding.

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.5 \
    --input-shape "224x224x3" \
    --heads "10,2,5" \
    --head-names "object_class,person,age_group" \
    --output-dir ./models/multi_task
```

This model predicts:
- Object class (10 categories)
- Person detection (2 classes)
- Age group (5 ranges: child, teen, adult, middle-aged, senior)

Use alpha 0.5 for better accuracy when handling multiple tasks.

## Example 4: Grayscale Image Processing

For applications using grayscale input (like some security cameras or medical imaging).

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "96x96x1" \
    --heads "2" \
    --output-dir ./models/grayscale
```

Smaller input size (96x96) with grayscale reduces model size further. Good for constrained devices.

## Example 5: Using Pretrained Weights

When you need better accuracy and can use a larger model.

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.75 \
    --input-shape "224x224x3" \
    --heads "100,20" \
    --head-names "object_class,action_class" \
    --use-pretrained \
    --output-dir ./models/pretrained
```

Pretrained ImageNet weights help when:
- You have limited training data
- The task is similar to ImageNet classification
- You can accept larger model size

Only works with alpha 0.75 or 1.0, and requires RGB input.

## Example 6: High-Resolution Input

For tasks needing more detail, use larger input resolution.

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.5 \
    --input-shape "320x320x3" \
    --heads "50,10" \
    --output-dir ./models/high_res
```

Larger inputs improve detail recognition but increase:
- Model size
- Inference time
- Memory usage

## Example 7: Custom Calibration for Better Quantization

If quantization quality is critical, use more calibration samples.

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --heads "5,2,3" \
    --calibration-samples 300 \
    --output-dir ./models/high_quality
```

More calibration samples can improve quantization accuracy, especially for:
- Models with complex activation distributions
- Multi-head models with diverse outputs
- Models where quantization artifacts are noticeable

Trade-off: Takes longer to generate the model.

## Example 8: Minimal Output (TFLite Only)

Skip saving the Keras model to save disk space.

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --heads "2" \
    --no-save-keras \
    --output-dir ./models/deployment_only
```

Useful when you only need the TFLite file for deployment and don't need the Keras model for further training.

## Example 9: Custom Output Naming

Specify your own output filename prefix.

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --heads "10,5" \
    --output-name "my_custom_model" \
    --output-dir ./models
```

This creates files named `my_custom_model_int8.tflite`, `my_custom_model_report.json`, etc.

## Example 10: Multiple Model Variants

Generate several variants to compare:

```bash
# Small model
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 --heads "2" --output-dir ./models/variants/small

# Medium model
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.5 --heads "2" --output-dir ./models/variants/medium

# Large model
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 1.0 --heads "2" --output-dir ./models/variants/large
```

Compare model sizes, inference speed, and accuracy to choose the best fit for your application.
