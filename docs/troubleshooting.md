# Troubleshooting

Common issues and their solutions.

## Model Creation Errors

### "Pretrained weights only available for alpha 0.75 or 1.0"

**Problem**: You're trying to use `--use-pretrained` with alpha 0.25 or 0.50.

**Solution**: Either:
- Remove `--use-pretrained` flag
- Change to `--alpha 0.75` or `--alpha 1.0`

Alpha 0.25 and 0.50 don't have pretrained weights in Keras MobileNet V3.

### "Invalid input shape format"

**Problem**: The input shape string format is incorrect.

**Solution**: Use format `HxWxC` exactly, like:
- `"224x224x3"` (correct)
- `"224,224,3"` (wrong - use x not comma)
- `224x224x3` (wrong - needs quotes)

### "Number of head names must match number of heads"

**Problem**: You provided `--head-names` but the count doesn't match `--heads`.

**Solution**: Ensure the number of comma-separated values matches:
```bash
--heads "5,2,3" --head-names "a,b,c"  # Correct: 3 heads, 3 names
--heads "5,2,3" --head-names "a,b"    # Wrong: 3 heads, 2 names
```

### "Each head must have at least 1 class"

**Problem**: One of your head class counts is 0 or negative.

**Solution**: Check your `--heads` argument. All values must be positive integers:
```bash
--heads "5,2,3"  # Correct
--heads "5,0,3"  # Wrong: second head has 0 classes
```

## Quantization Issues

### Model not fully quantized

**Problem**: The quantization info shows float32 tensors present.

**Possible causes**:
- Some operations can't be quantized (rare with MobileNet V3)
- Calibration dataset issues
- Model complexity

**Solution**:
- Check the quantization_info.json to see which tensors are float
- Try increasing calibration samples
- This is usually not a problem unless you see significant accuracy loss

### Large model size after quantization

**Problem**: Model is larger than expected after quantization.

**Possible causes**:
- Using alpha 1.0 (largest model)
- Many heads with many classes
- Quantization overhead

**Solution**:
- Use smaller alpha (0.25 or 0.50)
- Reduce number of classes per head if possible
- Check if you're saving the Keras model too (use `--no-save-keras`)

### Poor quantization quality

**Problem**: Accuracy drops significantly after quantization.

**Solution**:
- Increase `--calibration-samples` to 200-300
- Try a larger alpha for more model capacity
- Verify your calibration data is representative of real usage

## Performance Issues

### Model creation takes too long

**Problem**: Script runs slowly.

**Possible causes**:
- Large number of calibration samples
- Large model size (high alpha)
- System resources

**Solution**:
- Reduce `--calibration-samples` (100 is usually enough)
- Use smaller alpha if acceptable
- Ensure you have adequate RAM and CPU

### Model inference is slow

**Problem**: The generated TFLite model runs slowly.

**Possible causes**:
- Large input size
- Large alpha value
- Hardware limitations

**Solution**:
- Use smaller input size (e.g., 128x128 instead of 224x224)
- Use smaller alpha (0.25 instead of 1.0)
- Check if your hardware supports uint8/int8 acceleration (including Vela compiler for Arm Ethos-U NPU)
- Consider using hardware-specific optimizations for your target device

## File and Path Issues

### "Output directory cannot be created"

**Problem**: Permission error or invalid path.

**Solution**:
- Check write permissions in the target directory
- Use an absolute path or ensure relative path is correct
- Create the directory manually first if needed

### Cannot import models package

**Problem**: Python can't find the models module.

**Solution**:
- Run the script from the repository root directory
- Ensure you're using the correct Python environment
- Check that the models package exists in the expected location

## Accuracy Issues

### Model accuracy lower than expected

**Problem**: After quantization, accuracy drops too much.

**Possible causes**:
- Alpha too small for the task complexity
- Input preprocessing mismatch
- Quantization calibration issues

**Solution**:
- Try larger alpha (0.5 or 0.75)
- Verify input preprocessing matches training
- Increase calibration samples
- Check if pretrained weights would help (if using alpha 0.75+)

### Outputs don't make sense

**Problem**: Model outputs unexpected values.

**Possible causes**:
- Incorrect head configuration
- Output quantization not handled correctly
- Model not properly trained (this tool creates untrained models)

**Solution**:
- Verify head class counts match your expectations
- Check output quantization parameters and convert properly (models output logits, apply softmax in post-processing)
- Remember: This tool creates model architecture, not trained models. You need to train the model separately.
- Note: Models default to Vela-compatible mode (logits output). Apply softmax when interpreting outputs.

## Vela Compilation Issues

### "ValueError: negative shift count" during Vela compilation

**Problem**: Model fails to compile with Vela compiler, showing a negative shift count error in softmax operations.

**Cause**: This error occurs when models contain quantized softmax layers with quantization parameters that are incompatible with Vela's graph optimization.

**Solution**: 
- Models are generated Vela-compatible by default (outputs logits, no softmax)
- If you see this error, regenerate the model with the latest tool version
- The default behavior (logits output) avoids this issue
- Apply softmax in post-processing when interpreting outputs

### Model outputs logits instead of probabilities

**Problem**: Model outputs seem wrong or need conversion.

**Explanation**: This is expected and correct. Models default to Vela-compatible mode, which means:
- Outputs are logits (not probabilities)
- Softmax must be applied in post-processing
- This ensures compatibility with Vela compiler and Arm Ethos-U NPU

**Solution**: 
- Convert quantized outputs to float: `logits = (uint8_value - zero_point) * scale`
- Apply softmax: `probabilities = softmax(logits)`
- If you specifically need softmax in the model (non-Vela deployment), use `--with-softmax` flag

## Getting Help

If you encounter issues not covered here:

1. Check the error message carefully - it usually points to the problem
2. Review the model summary and report files for clues
3. Try a simpler configuration first (single head, default alpha)
4. Verify your Python and TensorFlow versions are compatible
5. Check that you have sufficient disk space for outputs
