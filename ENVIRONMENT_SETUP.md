# Python Environment Setup

Guide for setting up the Python environment to use the Multi-Head MobileNet V3 quantization tool.

## Requirements

- Python 3.7 or higher (tested with Python 3.13)
- pip (Python package manager)
- Virtual environment tool (venv, included with Python 3.3+)

## Quick Setup

### 1. Create Virtual Environment

From the project root directory:

```bash
python3 -m venv venv_mobilenet
```

This creates a virtual environment directory named `venv_mobilenet`.

### 2. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv_mobilenet/bin/activate
```

**On Windows:**
```bash
venv_mobilenet\Scripts\activate
```

When activated, your command prompt will show `(venv_mobilenet)` at the beginning.

### 3. Install Dependencies

The tool requires TensorFlow and NumPy. Install them with:

```bash
pip install --upgrade pip
pip install tensorflow numpy
```

For Python 3.13 (or newer), TensorFlow 2.20+ will be installed automatically. For older Python versions, TensorFlow 2.15 may be available.

### 4. Verify Installation

Check that TensorFlow is installed correctly:

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

You should see output like:
```
TensorFlow version: 2.20.0
```

## Using the Tool

Once the environment is set up, you can use the tool:

```bash
python multi_task_mobilenet_v3/examples/create_quantized_mobilenet_v3.py \
    --alpha 0.25 \
    --input-shape "128x128x1" \
    --heads "5,2,5,3,3" \
    --output-dir ./multi_task_mobilenet_v3/output
```

## Deactivating the Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

### TensorFlow Installation Issues

**Problem**: TensorFlow won't install or shows compatibility errors.

**Solution**: 
- Ensure you're using Python 3.7-3.13
- For Python 3.13+, TensorFlow 2.20+ is required
- For older Python versions, try `pip install tensorflow==2.15.0`

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
- Make sure the virtual environment is activated (check for `(venv_mobilenet)` in prompt)
- Reinstall: `pip install tensorflow numpy`

### Python Version Issues

**Problem**: Wrong Python version or virtual environment uses wrong Python.

**Solution**:
- Specify Python version explicitly: `python3.13 -m venv venv_mobilenet`
- Check version: `python --version`

## Minimal Requirements File

If you want to create a minimal requirements file for this tool:

```bash
# requirements_mobilenet.txt
tensorflow>=2.15.0
numpy>=1.22.0
```

Install with:
```bash
pip install -r requirements_mobilenet.txt
```

## Notes

- The virtual environment is project-specific and doesn't affect your system Python
- You need to activate the environment each time you open a new terminal
- The virtual environment directory (`venv_mobilenet`) can be safely deleted and recreated
- Model generation doesn't require CUDA/GPU - CPU-only TensorFlow works fine
