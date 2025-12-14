"""Report generation utilities for model statistics and metadata."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import tensorflow as tf

# Import architecture type for type hints
try:
    from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture
except ImportError:
    # Fallback for type hints if import fails
    MultiHeadMobileNetV3QATArchitecture = Any


def _convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy int64 and other non-serializable types to native Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        Serializable version of the object
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    return obj


def _normalize_dtype_string(dtype_str: str) -> str:
    """
    Normalize dtype string to readable format.
    
    Args:
        dtype_str: Raw dtype string (e.g., "<class 'numpy.uint8'>")
        
    Returns:
        Normalized dtype string (e.g., "uint8")
    """
    if 'uint8' in dtype_str:
        return 'uint8'
    elif 'int8' in dtype_str:
        return 'int8'
    elif 'float32' in dtype_str:
        return 'float32'
    elif 'float16' in dtype_str:
        return 'float16'
    return dtype_str


def generate_output_name(
    alpha: float,
    input_shape: Tuple[int, int, int],
    head_classes: List[int]
) -> str:
    """
    Generate output filename from configuration.
    
    Args:
        alpha: Width multiplier
        input_shape: Input shape as (height, width, channels)
        head_classes: List of class counts per head
        
    Returns:
        Generated filename base (without extension)
    """
    h, w, c = input_shape
    alpha_str = str(alpha).replace('.', '_')
    head_str = '_'.join(map(str, head_classes))
    return f"mnv3_{alpha_str}_{head_str}_{h}x{w}x{c}"


def save_model_report(
    architecture: MultiHeadMobileNetV3QATArchitecture,
    model: tf.keras.Model,
    output_dir: Path,
    output_name: str,
    quantization_info: Dict[str, Any],
    output_validation: Dict[str, Any],
    vela_compatible: bool = True
) -> None:
    """
    Save model statistics and metadata reports for a newly created model.
    
    Generates JSON reports, text summaries, and quantization info files
    with full architecture details.
    
    Args:
        architecture: Model architecture object with configuration
        model: Keras model instance
        output_dir: Directory to save reports
        output_name: Base name for output files
        quantization_info: Quantization analysis dictionary
        output_validation: Model output validation results
        vela_compatible: Whether model is Vela-compatible
    """
    param_info = architecture.get_parameter_count()
    
    # Use TFLite output details instead of Keras model outputs
    # TFLite outputs will show the correct quantized dtype (uint8)
    tflite_outputs = {}
    if 'output_details' in quantization_info and quantization_info['output_details']:
        output_details = quantization_info['output_details']
        # TFLite outputs are in the same order as the model outputs
        # Match by index to head_configs
        for idx, output_detail in enumerate(output_details):
            if idx < len(architecture.head_configs):
                head_name = architecture.head_configs[idx].name
            else:
                head_name = f"output_{idx}"
            
            dtype_str = _normalize_dtype_string(output_detail['dtype'])
            tflite_outputs[head_name] = {
                'shape': output_detail['shape'],
                'dtype': dtype_str
            }
    
    # Fallback to Keras outputs if TFLite outputs not available
    if not tflite_outputs:
        tflite_outputs = output_validation
    
    report = {
        'model_info': {
            'architecture': architecture.name,
            'input_shape': list(architecture.config.input_shape),
            'alpha': architecture.config.arch_params.get('alpha'),
            'use_pretrained': architecture.config.arch_params.get('use_pretrained', False),
            'vela_compatible': vela_compatible,
            'heads': [
                {
                    'name': h.name,
                    'num_classes': h.num_classes,
                    'activation': h.activation
                }
                for h in architecture.head_configs
            ],
            'total_classes': architecture.total_classes
        },
        'parameters': {
            'total': param_info['total'],
            'trainable': param_info['trainable'],
            'non_trainable': param_info['non_trainable'],
            'backbone': param_info['backbone'],
            'heads': param_info['heads']
        },
        'outputs': tflite_outputs,
        'quantization': quantization_info['quantization']
    }
    
    # Save JSON report
    report_serializable = _convert_to_serializable(report)
    report_path = output_dir / f"{output_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report_serializable, f, indent=2)
    
    # Generate text summary
    summary_lines = [
        f"Model: {architecture.name}",
        f"Input shape: {architecture.config.input_shape}",
        f"Alpha: {architecture.config.arch_params.get('alpha')}",
        f"",
        f"Parameters:",
        f"  Total: {param_info['total']:,}",
        f"  Trainable: {param_info['trainable']:,}",
        f"  Non-trainable: {param_info['non_trainable']:,}",
        f"  Backbone: {param_info['backbone']:,}",
        f"  Heads: {param_info['heads']:,}",
        f"",
        f"Heads:"
    ]
    
    for h in architecture.head_configs:
        summary_lines.append(f"  {h.name}: {h.num_classes} classes ({h.activation})")
    
    quant_info = quantization_info['quantization']
    
    if vela_compatible:
        summary_lines.extend([
            f"",
            f"Vela-Compatible Mode:",
            f"  - Model outputs logits (no softmax activation)",
            f"  - Softmax must be applied in post-processing",
            f"  - This avoids Vela compilation errors with quantized softmax"
        ])
    
    summary_lines.extend([
        f"",
        f"Quantization:",
        f"  Fully quantized: {quant_info['fully_quantized']}",
        f"  UINT8 tensors: {quant_info.get('uint8_tensors', 0)}",
        f"  INT8 tensors: {quant_info.get('int8_tensors', 0)}",
        f"  Float32 tensors: {quant_info['float32_tensors']}"
    ])
    
    summary_path = output_dir / f"{output_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Save quantization info
    quant_info_serializable = _convert_to_serializable(quantization_info)
    quant_info_path = output_dir / f"{output_name}_quantization_info.json"
    with open(quant_info_path, 'w') as f:
        json.dump(quant_info_serializable, f, indent=2)


def save_model_report_for_loaded_model(
    model: tf.keras.Model,
    output_dir: Path,
    output_name: str,
    quantization_info: Dict[str, Any],
    output_validation: Dict[str, Any],
    input_shape: Tuple[int, int, int],
    vela_compatible: bool = True
) -> None:
    """
    Save model statistics and metadata reports for a loaded trained model.
    
    Generates simplified reports for models loaded from checkpoints where
    we don't have full architecture information.
    
    Args:
        model: Loaded Keras model instance
        output_dir: Directory to save reports
        output_name: Base name for output files
        quantization_info: Quantization analysis dictionary
        output_validation: Model output validation results
        input_shape: Input shape as (height, width, channels)
        vela_compatible: Whether model is Vela-compatible
    """
    # Get basic model info
    total_params = model.count_params()
    trainable_params = sum([
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    ])
    non_trainable_params = total_params - trainable_params
    
    # Use TFLite output details
    tflite_outputs = {}
    if 'output_details' in quantization_info and quantization_info['output_details']:
        output_details = quantization_info['output_details']
        for idx, output_detail in enumerate(output_details):
            head_name = f"output_{idx}"
            dtype_str = _normalize_dtype_string(output_detail['dtype'])
            tflite_outputs[head_name] = {
                'shape': output_detail['shape'],
                'dtype': dtype_str
            }
    
    if not tflite_outputs:
        tflite_outputs = output_validation
    
    # Try to extract head info from model outputs
    head_info = []
    if isinstance(output_validation, dict):
        for head_name, output_info in output_validation.items():
            # Infer number of classes from output shape
            output_shape = output_info.get('shape', [])
            if len(output_shape) >= 2:
                num_classes = output_shape[-1]
                head_info.append({
                    'name': head_name,
                    'num_classes': num_classes,
                    'activation': 'unknown'
                })
    
    report = {
        'model_info': {
            'source': 'loaded_from_keras',
            'input_shape': list(input_shape),
            'vela_compatible': vela_compatible,
            'heads': head_info,
            'total_classes': sum(h.get('num_classes', 0) for h in head_info)
        },
        'parameters': {
            'total': int(total_params),
            'trainable': int(trainable_params),
            'non_trainable': int(non_trainable_params)
        },
        'outputs': tflite_outputs,
        'quantization': quantization_info['quantization']
    }
    
    # Save JSON report
    report_serializable = _convert_to_serializable(report)
    report_path = output_dir / f"{output_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report_serializable, f, indent=2)
    
    # Generate text summary
    summary_lines = [
        f"Model: Loaded from Keras checkpoint",
        f"Input shape: {input_shape}",
        f"",
        f"Parameters:",
        f"  Total: {total_params:,}",
        f"  Trainable: {trainable_params:,}",
        f"  Non-trainable: {non_trainable_params:,}",
        f"",
        f"Heads:"
    ]
    
    for h in head_info:
        summary_lines.append(f"  {h['name']}: {h['num_classes']} classes")
    
    quant_info = quantization_info['quantization']
    
    if vela_compatible:
        summary_lines.extend([
            f"",
            f"Vela-Compatible Mode:",
            f"  - Model outputs logits (no softmax activation)",
            f"  - Softmax must be applied in post-processing",
            f"  - This avoids Vela compilation errors with quantized softmax"
        ])
    
    summary_lines.extend([
        f"",
        f"Quantization:",
        f"  Fully quantized: {quant_info['fully_quantized']}",
        f"  UINT8 tensors: {quant_info.get('uint8_tensors', 0)}",
        f"  INT8 tensors: {quant_info.get('int8_tensors', 0)}",
        f"  Float32 tensors: {quant_info['float32_tensors']}"
    ])
    
    summary_path = output_dir / f"{output_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Save quantization info
    quant_info_serializable = _convert_to_serializable(quantization_info)
    quant_info_path = output_dir / f"{output_name}_quantization_info.json"
    with open(quant_info_path, 'w') as f:
        json.dump(quant_info_serializable, f, indent=2)

