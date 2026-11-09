"""Quantization utilities for converting Keras models to TFLite."""

import os
import tempfile
from typing import Dict, Tuple, Optional, Any

import numpy as np
import tensorflow as tf


def create_representative_dataset(
    input_shape: Tuple[int, int, int],
    num_samples: int
):
    """
    Create representative dataset generator for quantization calibration.
    
    Args:
        input_shape: Input shape as (height, width, channels)
        num_samples: Number of calibration samples to generate
        
    Yields:
        Batches of random data matching the input shape
    """
    def generator():
        for _ in range(num_samples):
            data = np.random.random((1,) + input_shape).astype(np.float32)
            yield [data]
    return generator


def quantize_to_tflite(
    model: tf.keras.Model,
    input_shape: Tuple[int, int, int],
    calibration_samples: int = 100,
    output_path: Optional[str] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Convert Keras model to quantized TFLite with uint8 input/output.
    
    Uses representative dataset calibration for post-training quantization.
    The resulting model will have uint8 inputs and outputs.
    
    Args:
        model: Keras model to quantize
        input_shape: Input shape as (height, width, channels)
        calibration_samples: Number of samples for quantization calibration
        output_path: Optional path to save the TFLite model
        
    Returns:
        Tuple of (TFLite model bytes, quantization analysis dictionary)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    representative_dataset = create_representative_dataset(
        input_shape,
        calibration_samples
    )
    converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    # Save first if path provided, then analyze
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        analysis = analyze_tflite_model(tflite_model, output_path)
    else:
        analysis = analyze_tflite_model(tflite_model, None)
    
    return tflite_model, analysis


def analyze_tflite_model(
    tflite_model: bytes,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze TFLite model structure and quantization.
    
    Examines the model's tensors, inputs, outputs, and quantization status.
    
    Args:
        tflite_model: TFLite model as bytes
        model_path: Optional path to the model file (for direct loading)
        
    Returns:
        Dictionary containing model analysis information
    """
    if model_path:
        interpreter = tf.lite.Interpreter(model_path=model_path)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tflite') as f:
            f.write(tflite_model)
            temp_path = f.name
        try:
            interpreter = tf.lite.Interpreter(model_path=temp_path)
        finally:
            os.unlink(temp_path)
    
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    int8_count = sum(1 for t in tensor_details if t['dtype'] == np.int8)
    uint8_count = sum(1 for t in tensor_details if t['dtype'] == np.uint8)
    float32_count = sum(1 for t in tensor_details if t['dtype'] == np.float32)
    quantized_count = int8_count + uint8_count
    
    analysis = {
        'input_details': [
            {
                'name': d['name'],
                'shape': d['shape'].tolist(),
                'dtype': str(d['dtype']),
                'quantization': d.get('quantization_parameters', {})
            }
            for d in input_details
        ],
        'output_details': [
            {
                'name': d['name'],
                'shape': d['shape'].tolist(),
                'dtype': str(d['dtype']),
                'quantization': d.get('quantization_parameters', {})
            }
            for d in output_details
        ],
        'quantization': {
            'total_tensors': len(tensor_details),
            'int8_tensors': int8_count,
            'uint8_tensors': uint8_count,
            'float32_tensors': float32_count,
            'fully_quantized': quantized_count > 0 and float32_count == 0
        }
    }
    
    return analysis


def convert_to_int8_tflite(
    model: tf.keras.Model,
    input_shape: Tuple[int, int, int],
    calibration_samples: int = 100,
    output_path: Optional[str] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Convert Keras model to int8 quantized TFLite.
    
    Args:
        model: Keras model to quantize
        input_shape: Input shape for calibration
        calibration_samples: Number of calibration samples
        output_path: Optional path to save the TFLite model
        
    Returns:
        Tuple of (TFLite model bytes, analysis dictionary)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    representative_dataset = create_representative_dataset(input_shape, calibration_samples)
    converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        analysis = analyze_tflite_model(tflite_model, output_path)
    else:
        analysis = analyze_tflite_model(tflite_model, None)
    
    return tflite_model, analysis


def convert_to_fp16_tflite(
    model: tf.keras.Model,
    output_path: Optional[str] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Convert Keras model to fp16 TFLite.
    
    Args:
        model: Keras model to convert
        output_path: Optional path to save the TFLite model
        
    Returns:
        Tuple of (TFLite model bytes, analysis dictionary)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        analysis = analyze_tflite_model(tflite_model, output_path)
    else:
        analysis = analyze_tflite_model(tflite_model, None)
    
    return tflite_model, analysis


def convert_to_fp32_tflite(
    model: tf.keras.Model,
    output_path: Optional[str] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Convert Keras model to fp32 TFLite.
    
    Args:
        model: Keras model to convert
        output_path: Optional path to save the TFLite model
        
    Returns:
        Tuple of (TFLite model bytes, analysis dictionary)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    tflite_model = converter.convert()
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        analysis = analyze_tflite_model(tflite_model, output_path)
    else:
        analysis = analyze_tflite_model(tflite_model, None)
    
    return tflite_model, analysis


def export_separate_tflite_models(
    architecture,
    input_shape: Tuple[int, int, int],
    output_dir: str,
    base_name: str,
    backbone_format: str = "int8",
    head_format: str = "fp16",
    calibration_samples: int = 100
) -> Dict[str, Dict[str, Any]]:
    """
    Export backbone and heads as separate TFLite files with different precisions.
    
    Args:
        architecture: Multi-head architecture with separable_weights=True
        input_shape: Input shape for calibration (backbone only)
        output_dir: Directory to save TFLite files
        base_name: Base name for output files
        backbone_format: Format for backbone ("int8", "fp16", "fp32")
        head_format: Format for heads ("int8", "fp16", "fp32")
        calibration_samples: Number of calibration samples for int8 conversion
        
    Returns:
        Dictionary with export results and file paths
        
    Raises:
        ValueError: If architecture doesn't have separable_weights enabled
    """
    if not architecture.is_separable:
        raise ValueError(
            "export_separate_tflite_models requires architecture with separable_weights=True"
        )
    
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'backbone': {},
        'heads': {},
        'files': []
    }
    
    # Convert backbone
    backbone_model = architecture.backbone
    backbone_file = output_path / f"{base_name}_backbone_{backbone_format}.tflite"
    
    if backbone_format == "int8":
        _, backbone_analysis = convert_to_int8_tflite(
            backbone_model, input_shape, calibration_samples, str(backbone_file)
        )
    elif backbone_format == "fp16":
        _, backbone_analysis = convert_to_fp16_tflite(
            backbone_model, str(backbone_file)
        )
    elif backbone_format == "fp32":
        _, backbone_analysis = convert_to_fp32_tflite(
            backbone_model, str(backbone_file)
        )
    else:
        raise ValueError(f"Unsupported backbone format: {backbone_format}")
    
    backbone_size = backbone_file.stat().st_size
    results['backbone'] = {
        'file': str(backbone_file),
        'format': backbone_format,
        'size_bytes': backbone_size,
        'size_kb': backbone_size / 1024,
        'analysis': backbone_analysis
    }
    results['files'].append(str(backbone_file))
    
    # Convert heads
    for head_name in architecture.head_names:
        head_model = architecture.head(head_name)
        head_file = output_path / f"{base_name}_{head_name}_{head_format}.tflite"
        
        if head_format == "int8":
            # For heads, we need to get the feature shape from backbone output
            dummy_input = tf.zeros((1,) + input_shape)
            backbone_output = backbone_model(dummy_input)
            feature_shape = backbone_output.shape[1:]
            _, head_analysis = convert_to_int8_tflite(
                head_model, feature_shape, calibration_samples, str(head_file)
            )
        elif head_format == "fp16":
            _, head_analysis = convert_to_fp16_tflite(
                head_model, str(head_file)
            )
        elif head_format == "fp32":
            _, head_analysis = convert_to_fp32_tflite(
                head_model, str(head_file)
            )
        else:
            raise ValueError(f"Unsupported head format: {head_format}")
        
        head_size = head_file.stat().st_size
        results['heads'][head_name] = {
            'file': str(head_file),
            'format': head_format,
            'size_bytes': head_size,
            'size_kb': head_size / 1024,
            'analysis': head_analysis
        }
        results['files'].append(str(head_file))
    
    return results


def validate_model_outputs(
    model: tf.keras.Model,
    input_shape: Tuple[int, int, int]
) -> Dict[str, Any]:
    """
    Validate model structure and test inference.
    
    Runs a test inference to verify the model works correctly and extracts
    output information.
    
    Args:
        model: Keras model to validate
        input_shape: Input shape as (height, width, channels)
        
    Returns:
        Dictionary containing output information (shapes, dtypes)
    """
    test_input = np.random.random((1,) + input_shape).astype(np.float32)
    outputs = model(test_input)
    
    output_info = {}
    if isinstance(outputs, dict):
        for head_name, output in outputs.items():
            output_info[head_name] = {
                'shape': list(output.shape),
                'dtype': str(output.dtype)
            }
    else:
        output_info['output'] = {
            'shape': list(outputs.shape),
            'dtype': str(outputs.dtype)
        }
    
    return output_info

