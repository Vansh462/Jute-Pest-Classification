#!/usr/bin/env python3
"""
Convert the saved BiT model to TensorFlow Lite for faster loading
Run this script once to create a lighter model version
"""

import tensorflow as tf
import numpy as np

def convert_model_to_tflite():
    """Convert the saved model to TensorFlow Lite format"""
    
    print("Loading the saved model...")
    model = tf.saved_model.load('my_saved_bit_model/')
    
    print("Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_saved_model('my_saved_bit_model/')
    
    # Fix for variable nodes - create concrete function
    concrete_func = model.signatures['serving_default']
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Optimize for size and speed
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open('jute_pest_model_fixed.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved as 'jute_pest_model_fixed.tflite'")
    print(f"TFLite model size: {len(tflite_model) / (1024*1024):.1f}MB")

if __name__ == "__main__":
    convert_model_to_tflite()
