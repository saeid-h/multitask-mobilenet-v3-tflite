"""Model loading utilities with version compatibility."""

import os
from typing import Optional

import tensorflow as tf


def load_keras_model(model_path: str, verbose: bool = True) -> tf.keras.Model:
    """
    Load a trained Keras model with fallback strategies for version compatibility.
    
    This function tries multiple loading strategies to handle different Keras
    versions and model formats:
    1. Standard Keras load_model (compatible with most versions)
    2. Load with safe_mode=False (for newer Keras 3.x versions)
    3. Load as SavedModel format (fallback)
    
    Args:
        model_path: Path to the .keras model file
        verbose: Whether to print loading progress messages
        
    Returns:
        Loaded Keras model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If all loading strategies fail
    """
    if verbose:
        print(f"Loading trained model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Strategy 1: Standard load with compile=False
    try:
        if verbose:
            print("  Trying: Standard Keras load_model")
        model = tf.keras.models.load_model(model_path, compile=False)
        if verbose:
            print("  Success with standard loading")
        return model
    except Exception as e:
        if verbose:
            print(f"  Warning: Standard loading failed: {str(e)[:100]}...")
    
    # Strategy 2: Load with safe_mode=False (for newer Keras versions)
    try:
        if verbose:
            print("  Trying: Keras load_model with safe_mode=False")
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        if verbose:
            print("  Success with safe_mode=False")
        return model
    except Exception as e:
        if verbose:
            print(f"  Warning: Loading with safe_mode=False failed: {str(e)[:100]}...")
    
    # Strategy 3: Try loading as SavedModel
    try:
        if verbose:
            print("  Trying: Load as SavedModel")
        saved_model = tf.saved_model.load(model_path)
        # If it's a SavedModel, we need to extract the Keras model from it
        if hasattr(saved_model, 'signatures'):
            # Create a wrapper model
            # This is a simplified approach - may need adjustment based on actual SavedModel structure
            if verbose:
                print("  Warning: SavedModel detected but may need manual conversion")
            raise RuntimeError("SavedModel format requires manual conversion")
        if verbose:
            print("  Success with SavedModel loading")
        return saved_model
    except Exception as e:
        if verbose:
            print(f"  Warning: SavedModel loading failed: {str(e)[:100]}...")
    
    raise RuntimeError(f"Failed to load model from {model_path} with all strategies")

