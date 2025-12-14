#!/usr/bin/env python3
"""
Create quantized MobileNet V3 models with configurable multi-head outputs.

This script generates fully quantized (uint8) TensorFlow Lite models from
MobileNet V3 architectures with custom classification heads.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import tensorflow as tf

# Add project root to path to access models package
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import HeadConfiguration, create_head_config_from_list
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture


def parse_input_shape(shape_str: str) -> Tuple[int, int, int]:
    """Parse input shape string like '224x224x3' into tuple."""
    try:
        parts = shape_str.split('x')
        if len(parts) != 3:
            raise ValueError("Input shape must be in format HxWxC (e.g., 224x224x3)")
        return tuple(int(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"Invalid input shape format: {e}")


def create_representative_dataset(input_shape: Tuple[int, int, int], num_samples: int):
    """Create representative dataset generator for quantization calibration."""
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
    
    Returns the TFLite model bytes and quantization analysis info.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    representative_dataset = create_representative_dataset(input_shape, calibration_samples)
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


def analyze_tflite_model(tflite_model: bytes, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Analyze TFLite model structure and quantization."""
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


def validate_model_outputs(model: tf.keras.Model, input_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    """Validate model structure and test inference."""
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


def generate_output_name(alpha: float, input_shape: Tuple[int, int, int], head_classes: List[int]) -> str:
    """Generate output filename from configuration."""
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
    """Save model statistics and metadata reports."""
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
            
            # Clean up dtype string format
            dtype_str = output_detail['dtype']
            # Convert "<class 'numpy.uint8'>" or "<class 'numpy.int8'>" to readable format
            if 'uint8' in dtype_str:
                dtype_str = 'uint8'
            elif 'int8' in dtype_str:
                dtype_str = 'int8'
            elif 'float32' in dtype_str:
                dtype_str = 'float32'
            elif 'float16' in dtype_str:
                dtype_str = 'float16'
            
            tflite_outputs[head_name] = {
                'shape': output_detail['shape'],
                'dtype': dtype_str  # This will be uint8 for quantized models
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
    
    def convert_to_serializable(obj):
        """Convert numpy int64 and other non-serializable types to native Python types."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    report_serializable = convert_to_serializable(report)
    report_path = output_dir / f"{output_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report_serializable, f, indent=2)
    
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
    
    def convert_to_serializable(obj):
        """Convert numpy int64 and other non-serializable types to native Python types."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    quant_info_serializable = convert_to_serializable(quantization_info)
    quant_info_path = output_dir / f"{output_name}_quantization_info.json"
    with open(quant_info_path, 'w') as f:
        json.dump(quant_info_serializable, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Create quantized MobileNet V3 models with multi-head outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.25,
        choices=[0.25, 0.50, 0.75, 1.0],
        help='Width multiplier (default: 0.25)'
    )
    
    parser.add_argument(
        '--input-shape',
        type=str,
        default='224x224x3',
        help='Input shape as HxWxC (default: 224x224x3)'
    )
    
    parser.add_argument(
        '--heads',
        type=str,
        required=True,
        help='Comma-separated class counts per head (e.g., "5,2,3")'
    )
    
    parser.add_argument(
        '--head-names',
        type=str,
        default=None,
        help='Optional comma-separated head names (default: auto-generated)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for models and reports'
    )
    
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Base name for output files (default: auto-generated)'
    )
    
    parser.add_argument(
        '--use-pretrained',
        action='store_true',
        help='Use ImageNet pretrained weights (only works with alpha 0.75 or 1.0)'
    )
    
    parser.add_argument(
        '--calibration-samples',
        type=int,
        default=100,
        help='Number of samples for quantization calibration (default: 100)'
    )
    
    parser.add_argument(
        '--save-keras',
        action='store_true',
        default=True,
        help='Save intermediate Keras model (default: True)'
    )
    
    parser.add_argument(
        '--no-save-keras',
        dest='save_keras',
        action='store_false',
        help='Skip saving Keras model'
    )
    
    parser.add_argument(
        '--vela-compatible',
        action='store_true',
        default=True,
        help='Generate Vela-compatible model (removes softmax, outputs logits). Softmax should be applied in post-processing. (default: True)'
    )
    
    parser.add_argument(
        '--with-softmax',
        dest='vela_compatible',
        action='store_false',
        help='Include softmax activation in model (not compatible with Vela compilation)'
    )
    
    args = parser.parse_args()
    
    # Parse input shape
    try:
        input_shape = parse_input_shape(args.input_shape)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Parse head configurations
    try:
        head_classes = [int(x.strip()) for x in args.heads.split(',')]
        if not head_classes or any(c < 1 for c in head_classes):
            raise ValueError("Each head must have at least 1 class")
    except ValueError as e:
        print(f"Error: Invalid head configuration: {e}")
        return 1
    
    # Parse head names if provided
    head_names = None
    if args.head_names:
        head_names = [x.strip() for x in args.head_names.split(',')]
        if len(head_names) != len(head_classes):
            print(f"Error: Number of head names ({len(head_names)}) must match number of heads ({len(head_classes)})")
            return 1
    
    # Validate pretrained weights compatibility
    if args.use_pretrained and args.alpha not in [0.75, 1.0]:
        print(f"Error: Pretrained weights only available for alpha 0.75 or 1.0, got {args.alpha}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output name if not provided
    output_name = args.output_name
    if not output_name:
        output_name = generate_output_name(args.alpha, input_shape, head_classes)
    
    try:
        print(f"Creating MobileNet V3 model...")
        print(f"  Alpha: {args.alpha}")
        print(f"  Input shape: {input_shape}")
        print(f"  Heads: {head_classes}")
        
        # Create head configurations
        head_configs = create_head_config_from_list(head_classes, head_names)
        
        # Override activation for Vela compatibility (default behavior)
        if args.vela_compatible:
            # Heads will output logits (linear activation) instead of softmax
            pass
            for head_config in head_configs:
                head_config.activation = 'linear'  # Remove softmax, output logits
        
        # Create model configuration
        config = MultiHeadModelConfig(
            input_shape=input_shape,
            head_configs=head_configs,
            arch_params={
                'alpha': args.alpha,
                'use_pretrained': args.use_pretrained,
            },
            training_mode='joint',
            inference_mode='all_active'
        )
        
        # Create architecture
        architecture = MultiHeadMobileNetV3QATArchitecture(config)
        
        print(f"  Architecture: {architecture.name}")
        
        # Build model
        print("Building model...")
        model = architecture.get_model()
        
        # Validate model
        print("Validating model...")
        output_validation = validate_model_outputs(model, input_shape)
        print(f"  Model outputs validated: {len(output_validation)} head(s)")
        
        # Save Keras model if requested
        if args.save_keras:
            keras_path = output_dir / f"{output_name}.keras"
            print(f"Saving Keras model to {keras_path}...")
            model.save(str(keras_path))
        
        # Quantize to TFLite
        print("Quantizing to TFLite (uint8)...")
        tflite_path = output_dir / f"{output_name}_int8.tflite"
        tflite_model, quantization_info = quantize_to_tflite(
            model,
            input_shape,
            args.calibration_samples,
            str(tflite_path)
        )
        
        file_size_bytes = os.path.getsize(tflite_path)
        file_size_kb = file_size_bytes / 1024
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        print(f"  TFLite model saved: {tflite_path}")
        print(f"  Size: {file_size_mb:.2f} MB ({file_size_kb:.1f} KB)")
        
        # Validate quantization
        quant_info = quantization_info['quantization']
        if quant_info['fully_quantized']:
            print("  Quantization: Fully quantized (uint8)")
        else:
            uint8_info = f"UINT8: {quant_info.get('uint8_tensors', 0)}, " if quant_info.get('uint8_tensors', 0) > 0 else ""
            int8_info = f"INT8: {quant_info.get('int8_tensors', 0)}, " if quant_info.get('int8_tensors', 0) > 0 else ""
            print(f"  Quantization: Mixed precision ({uint8_info}{int8_info}FP32: {quant_info['float32_tensors']})")
        
        # Generate reports
        print("Generating reports...")
        save_model_report(architecture, model, output_dir, output_name, quantization_info, output_validation, args.vela_compatible)
        
        print(f"\nComplete! Outputs saved to: {output_dir}")
        print(f"  Model: {tflite_path.name}")
        print(f"  Report: {output_name}_report.json")
        print(f"  Summary: {output_name}_summary.txt")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
