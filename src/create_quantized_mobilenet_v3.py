#!/usr/bin/env python3
"""
Create quantized MobileNet V3 models with configurable multi-head outputs.

This script generates fully quantized (uint8) TensorFlow Lite models from
MobileNet V3 architectures with custom classification heads.

It supports two modes:
1. Create new model from scratch and quantize
2. Load trained model and quantize
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path to access models package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import create_head_config_from_list
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture

from src.utils import (
    parse_input_shape,
    load_keras_model,
    quantize_to_tflite,
    validate_model_outputs,
    save_model_report,
    save_model_report_for_loaded_model,
    generate_output_name,
    export_separate_tflite_models
)


def _validate_args(args):
    """Validate command-line arguments."""
    if args.keras_model_path and args.heads:
        print("Warning: --heads argument ignored when loading trained model with --keras-model-path")
    
    if args.keras_model_path and args.unified_output:
        print("Warning: --unified-output is only available for newly created models, not loaded models")
        print("  The flag will be ignored for this loaded model.")
    
    if not args.keras_model_path and not args.heads:
        print("Error: Either --heads (for new model) or --keras-model-path (for trained model) must be provided")
        return False
    
    return True


def _parse_head_config(heads_str: str, head_names_str: str = None):
    """Parse head configuration from command-line arguments."""
    try:
        head_classes = [int(x.strip()) for x in heads_str.split(',')]
        if not head_classes or any(c < 1 for c in head_classes):
            raise ValueError("Each head must have at least 1 class")
    except ValueError as e:
        raise ValueError(f"Invalid head configuration: {e}")
    
    head_names = None
    if head_names_str:
        head_names = [x.strip() for x in head_names_str.split(',')]
        if len(head_names) != len(head_classes):
            raise ValueError(
                f"Number of head names ({len(head_names)}) must match "
                f"number of heads ({len(head_classes)})"
            )
    
    return head_classes, head_names


def _create_new_model(args, input_shape):
    """Create a new MobileNet V3 model from configuration."""
    print("Creating MobileNet V3 model...")
    print(f"  Alpha: {args.alpha}")
    print(f"  Input shape: {input_shape}")
    
    # Parse head configuration
    head_classes, head_names = _parse_head_config(args.heads, args.head_names)
    print(f"  Heads: {head_classes}")
    
    # Check unified output for single-head models
    if args.unified_output and len(head_classes) == 1:
        print("  Note: --unified-output ignored for single-head models (not needed)")
        args.unified_output = False
    
    # Create head configurations
    head_configs = create_head_config_from_list(head_classes, head_names)
    
    # Override activation for Vela compatibility (default behavior)
    if args.vela_compatible:
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
        inference_mode='all_active',
        separable_weights=args.separable_weights
    )
    
    if args.separable_weights:
        print(f"  Separable weights: Enabled")
    
    # Create architecture
    architecture = MultiHeadMobileNetV3QATArchitecture(config, unified_output=args.unified_output)
    print(f"  Architecture: {architecture.name}")
    if args.unified_output:
        head_order = [h.name for h in head_configs]
        print(f"  Unified output: Enabled (concatenated {len(head_classes)} heads)")
        print(f"    Head order: {' -> '.join(head_order)}")
    
    # Build model
    print("Building model...")
    model = architecture.get_model()
    
    return model, architecture, head_classes


def _load_trained_model(args, input_shape):
    """Load a trained Keras model."""
    print("Loading trained model...")
    model = load_keras_model(args.keras_model_path)
    
    # Infer input shape from loaded model if available
    if model.input_shape:
        inferred_shape = tuple(model.input_shape[1:4])  # Skip batch dimension
        if input_shape != inferred_shape:
            print(
                f"  Note: Input shape from model ({inferred_shape}) differs "
                f"from --input-shape ({input_shape})"
            )
            print(f"  Using inferred shape: {inferred_shape}")
            input_shape = inferred_shape
    
    print(f"  Detected input shape: {input_shape}")
    print(f"  Parameters: {model.count_params():,}")
    
    return model, input_shape


def _quantize_and_report(
    model,
    input_shape,
    args,
    output_dir,
    output_name,
    architecture=None,
    head_classes=None
):
    """Quantize model to TFLite and generate reports."""
    # Validate model
    print("Validating model...")
    output_validation = validate_model_outputs(model, input_shape)
    print(f"  Model outputs validated: {len(output_validation)} head(s)")
    
    # Save Keras model if requested (only for newly created models)
    if args.save_keras and architecture is not None:
        keras_path = output_dir / f"{output_name}.keras"
        print(f"Saving Keras model to {keras_path}...")
        model.save(str(keras_path))
    
    # Save separate weights if requested
    if args.save_separate_weights and architecture is not None:
        if not architecture.is_separable:
            print("Warning: --save-separate-weights requires --separable-weights. Skipping.")
        else:
            weights_dir = output_dir / f"{output_name}_weights"
            print(f"Saving separate weights to {weights_dir}...")
            saved_files = architecture.save_all_weights_separately(str(weights_dir))
            print(f"  Saved {len(saved_files)} weight files")
    
    # Export separate TFLite models if requested
    if args.export_separate_tflite and architecture is not None:
        if not architecture.is_separable:
            print("Warning: --export-separate-tflite requires --separable-weights. Skipping.")
        else:
            separate_dir = output_dir / f"{output_name}_separate_tflite"
            print(f"Exporting separate TFLite models to {separate_dir}...")
            print(f"  Backbone format: {args.backbone_format}")
            print(f"  Head format: {args.head_format}")
            
            separate_results = export_separate_tflite_models(
                architecture,
                input_shape,
                str(separate_dir),
                output_name,
                backbone_format=args.backbone_format,
                head_format=args.head_format,
                calibration_samples=args.calibration_samples
            )
            
            # Report separate export results
            bb_info = separate_results['backbone']
            print(f"  Backbone ({bb_info['format']}): {bb_info['size_kb']:.1f} KB")
            for head_name, head_info in separate_results['heads'].items():
                print(f"  Head '{head_name}' ({head_info['format']}): {head_info['size_kb']:.1f} KB")
            print(f"  Total files: {len(separate_results['files'])}")
    
    # Quantize to TFLite (unified model)
    print("Quantizing to TFLite (uint8)...")
    tflite_path = output_dir / f"{output_name}_int8.tflite"
    tflite_model, quantization_info = quantize_to_tflite(
        model,
        input_shape,
        args.calibration_samples,
        str(tflite_path)
    )
    
    # Report file size
    file_size_bytes = os.path.getsize(tflite_path)
    file_size_kb = file_size_bytes / 1024
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"  TFLite model saved: {tflite_path}")
    print(f"  Size: {file_size_mb:.2f} MB ({file_size_kb:.1f} KB)")
    
    # Report quantization status
    quant_info = quantization_info['quantization']
    if quant_info['fully_quantized']:
        print("  Quantization: Fully quantized (uint8)")
    else:
        uint8_info = ""
        if quant_info.get('uint8_tensors', 0) > 0:
            uint8_info = f"UINT8: {quant_info.get('uint8_tensors', 0)}, "
        int8_info = ""
        if quant_info.get('int8_tensors', 0) > 0:
            int8_info = f"INT8: {quant_info.get('int8_tensors', 0)}, "
        print(
            f"  Quantization: Mixed precision "
            f"({uint8_info}{int8_info}FP32: {quant_info['float32_tensors']})"
        )
    
    # Generate reports
    print("Generating reports...")
    if architecture is not None:
        # Full report for newly created models
        save_model_report(
            architecture, model, output_dir, output_name,
            quantization_info, output_validation, args.vela_compatible
        )
    else:
        # Simplified report for loaded models
        save_model_report_for_loaded_model(
            model, output_dir, output_name, quantization_info,
            output_validation, input_shape, args.vela_compatible
        )
    
    print(f"\nComplete! Outputs saved to: {output_dir}")
    print(f"  Model: {tflite_path.name}")
    print(f"  Report: {output_name}_report.json")
    print(f"  Summary: {output_name}_summary.txt")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create quantized MobileNet V3 models with multi-head outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model configuration
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
        default=None,
        help='Comma-separated class counts per head (e.g., "5,2,3"). '
             'Required when creating new model, ignored when loading trained model.'
    )
    
    parser.add_argument(
        '--keras-model-path',
        type=str,
        default=None,
        help='Path to trained Keras model (.keras file) to quantize. '
             'If provided, skips model creation and loads this model instead.'
    )
    
    parser.add_argument(
        '--head-names',
        type=str,
        default=None,
        help='Optional comma-separated head names (default: auto-generated)'
    )
    
    # Output configuration
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
    
    # Model options
    parser.add_argument(
        '--use-pretrained',
        action='store_true',
        help='Use ImageNet pretrained weights (only works with alpha 0.75 or 1.0)'
    )
    
    # Quantization options
    parser.add_argument(
        '--calibration-samples',
        type=int,
        default=100,
        help='Number of samples for quantization calibration (default: 100)'
    )
    
    # Save options
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
    
    # Vela compatibility
    parser.add_argument(
        '--vela-compatible',
        action='store_true',
        default=True,
        help='Generate Vela-compatible model (removes softmax, outputs logits). '
             'Softmax should be applied in post-processing. (default: True)'
    )
    
    parser.add_argument(
        '--with-softmax',
        dest='vela_compatible',
        action='store_false',
        help='Include softmax activation in model (not compatible with Vela compilation)'
    )
    
    # Unified output option
    parser.add_argument(
        '--unified-output',
        action='store_true',
        default=False,
        help='Add unified concatenated output for multi-head models. All head outputs '
             'are concatenated into a single tensor named "unified_heads". '
             'Useful for NPUs that do not support multiple output tensors. '
             'Ignored for single-head models. (default: False)'
    )
    
    # Separable weights option
    parser.add_argument(
        '--separable-weights',
        action='store_true',
        default=False,
        help='Enable separable weight management. Allows saving/loading backbone and '
             'heads separately, feature caching, and dynamic head operations. '
             'Useful for transfer learning and extending models with new heads. '
             '(default: False)'
    )
    
    # Save separate weights option
    parser.add_argument(
        '--save-separate-weights',
        action='store_true',
        default=False,
        help='Save backbone and head weights to separate files (requires --separable-weights). '
             'Creates backbone_weights.h5 and <head_name>_weights.h5 files. (default: False)'
    )
    
    # Export separate TFLite models option
    parser.add_argument(
        '--export-separate-tflite',
        action='store_true',
        default=False,
        help='Export backbone and heads as separate TFLite files (requires --separable-weights). '
             'Allows different precision formats for backbone and heads. (default: False)'
    )
    
    # Backbone precision format
    parser.add_argument(
        '--backbone-format',
        type=str,
        choices=['int8', 'fp16', 'fp32'],
        default='int8',
        help='Precision format for backbone TFLite export (default: int8)'
    )
    
    # Head precision format
    parser.add_argument(
        '--head-format',
        type=str,
        choices=['int8', 'fp16', 'fp32'],
        default='fp16',
        help='Precision format for head TFLite exports (default: fp16)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not _validate_args(args):
        return 1
    
    # Parse input shape
    try:
        input_shape = parse_input_shape(args.input_shape)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Validate pretrained weights compatibility
    if not args.keras_model_path:
        if args.use_pretrained and args.alpha not in [0.75, 1.0]:
            print(
                f"Error: Pretrained weights only available for alpha 0.75 or 1.0, "
                f"got {args.alpha}"
            )
            return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output name if not provided
    output_name = args.output_name
    if not output_name:
        if args.keras_model_path:
            model_name = Path(args.keras_model_path).stem
            output_name = f"{model_name}_quantized"
        else:
            head_classes, _ = _parse_head_config(args.heads)
            output_name = generate_output_name(args.alpha, input_shape, head_classes)
    
    try:
        # Load or create model
        architecture = None
        head_classes = None
        
        if args.keras_model_path:
            # Load existing trained model
            model, input_shape = _load_trained_model(args, input_shape)
        else:
            # Create new model
            model, architecture, head_classes = _create_new_model(args, input_shape)
        
        # Quantize and generate reports
        _quantize_and_report(
            model, input_shape, args, output_dir, output_name,
            architecture, head_classes
        )
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
