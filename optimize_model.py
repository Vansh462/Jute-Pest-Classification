#!/usr/bin/env python3
"""
Create an optimized version of the model for faster loading
"""

import tensorflow as tf
import os

def optimize_saved_model():
    """Create an optimized version of the saved model"""
    
    print("Loading original model...")
    model = tf.saved_model.load('my_saved_bit_model/')
    
    # Create optimized version
    print("Creating optimized model...")
    
    # Save with optimization
    tf.saved_model.save(
        model,
        'my_saved_bit_model_optimized/',
        options=tf.saved_model.SaveOptions(
            experimental_io_device='/job:localhost'
        )
    )
    
    print("Optimized model saved to 'my_saved_bit_model_optimized/'")
    
    # Compare sizes
    original_size = get_directory_size('my_saved_bit_model/')
    optimized_size = get_directory_size('my_saved_bit_model_optimized/')
    
    print(f"Original size: {original_size / (1024*1024):.1f}MB")
    print(f"Optimized size: {optimized_size / (1024*1024):.1f}MB")

def get_directory_size(path):
    """Get total size of directory"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size

if __name__ == "__main__":
    optimize_saved_model()
