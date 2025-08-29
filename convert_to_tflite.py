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
    
    # Optimize for size and speed
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: Use float16 quantization for even smaller size
    # converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open('jute_pest_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved as 'jute_pest_model.tflite'")
    print(f"Original model size: ~170MB")
    print(f"TFLite model size: {len(tflite_model) / (1024*1024):.1f}MB")

if __name__ == "__main__":
    convert_model_to_tflite()
