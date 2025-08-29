import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Reduce TensorFlow logging verbosity and optimize loading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for faster startup
tf.get_logger().setLevel('ERROR')

# Configure TensorFlow for faster loading
tf.config.optimizer.set_jit(False)  # Disable XLA for faster startup

# Set page config
st.set_page_config(
    page_title="Jute Pest Classifier",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class names from your model
CLASS_NAMES = [
    'Beet Armyworm', 'Black Hairy', 'Cutworm', 'Field Cricket', 
    'Jute Aphid', 'Jute Hairy', 'Jute Red Mite', 'Jute Semilooper', 
    'Jute Stem Girdler', 'Jute Stem Weevil', 'Leaf Beetle', 'Mealybug', 
    'Pod Borer', 'Scopula Emissaria', 'Termite', 'Termite odontotermes (Rambur)', 
    'Yellow Mite'
]

@st.cache_resource
def load_model():
    """Load the trained BiT model"""
    try:
        # Try to load the saved model
        model = tf.saved_model.load('my_saved_bit_model/')
        return model
    except Exception as e:
        st.error(f"Error loading saved model: {str(e)}")
        st.info("The model files appear to be incomplete. Please re-run your training notebook to save the model properly.")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to 512x512 as per your training configuration
    image = image.resize((512, 512))
    
    # Convert to array and normalize
    image_array = np.array(image)
    
    # Ensure RGB format
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Normalize to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    else:
        return None

def predict_pest(model, image_array):
    """Make prediction using the loaded model"""
    try:
        # Get predictions - use the signatures for saved models
        if hasattr(model, 'signatures'):
            # Use the serving signature
            infer = model.signatures['serving_default']
            # Get the input key (usually the first one)
            input_key = list(infer.structured_input_signature[1].keys())[0]
            predictions = infer(**{input_key: tf.constant(image_array)})
            # Get the output (usually there's one output)
            output_key = list(predictions.keys())[0]
            predictions = predictions[output_key]
        else:
            # Fallback: try direct call
            predictions = model(tf.constant(image_array))

        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(predictions).numpy()[0]

        # Get top prediction
        predicted_class_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class_idx])

        return predicted_class_idx, confidence, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error(f"Model type: {type(model)}")
        if hasattr(model, 'signatures'):
            st.error(f"Available signatures: {list(model.signatures.keys())}")
        return None, None, None

def main():
    st.title("🐛 Jute Pest Classifier")
    st.markdown("### Upload an image to identify jute pests")

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses a BiT (Big Transfer) model to classify 17 different types of jute pests.

        **Supported Pest Types:**
        """)
        for i, pest in enumerate(CLASS_NAMES, 1):
            st.write(f"{i}. {pest}")

        st.markdown("---")
        st.write("**Model Performance:**")
        st.write("- Test Accuracy: 95.5%")
        st.write("- Model: BiT-M R101x1")
        st.write("- Input Size: 512x512 pixels")

    # Handle model loading with proper UI
    with st.spinner("🤖 Loading BiT model... This may take 10-15 seconds on first run."):
        model = load_model()

    if model is not None:
        st.success("✅ Model loaded successfully! Ready for pest classification.")

    if model is None:
        st.error("Failed to load the model.")
        st.markdown("""
        ### How to fix this issue:

        1. **Re-run your training notebook** to properly save the model
        2. Make sure the model is saved using `tf.saved_model.save()` or `model.save()`
        3. Ensure the `my_saved_bit_model` directory contains all necessary files:
           - `saved_model.pb`
           - `variables/` directory with `variables.data-*` and `variables.index` files

        ### Alternative solution:
        You can also modify the notebook to save the model in a different format or location.
        """)
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image of a jute pest for classification"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Show image details
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")
        
        with col2:
            st.subheader("Prediction Results")
            
            # Preprocess image
            with st.spinner("Processing image..."):
                processed_image = preprocess_image(image)
            
            if processed_image is not None:
                # Make prediction
                with st.spinner("Classifying pest..."):
                    pred_idx, confidence, probabilities = predict_pest(model, processed_image)
                
                if pred_idx is not None:
                    # Display main prediction
                    predicted_pest = CLASS_NAMES[pred_idx]
                    
                    st.success(f"**Predicted Pest:** {predicted_pest}")
                    st.info(f"**Confidence:** {confidence:.2%}")
                    
                    # Show confidence meter
                    st.progress(confidence)
                    
                    # Show top 3 predictions
                    st.subheader("Top 3 Predictions")
                    top_3_indices = np.argsort(probabilities)[-3:][::-1]
                    
                    for i, idx in enumerate(top_3_indices):
                        pest_name = CLASS_NAMES[idx]
                        prob = probabilities[idx]
                        
                        if i == 0:
                            st.write(f"🥇 **{pest_name}**: {prob:.2%}")
                        elif i == 1:
                            st.write(f"🥈 **{pest_name}**: {prob:.2%}")
                        else:
                            st.write(f"🥉 **{pest_name}**: {prob:.2%}")
                    
                    # Show all probabilities in an expandable section
                    with st.expander("View All Class Probabilities"):
                        prob_data = []
                        for i, prob in enumerate(probabilities):
                            prob_data.append({
                                'Pest Type': CLASS_NAMES[i],
                                'Probability': f"{prob:.4f}",
                                'Percentage': f"{prob:.2%}"
                            })
                        
                        # Sort by probability
                        prob_data.sort(key=lambda x: float(x['Probability']), reverse=True)
                        st.table(prob_data)
                
            else:
                st.error("Error processing image. Please make sure it's a valid RGB image.")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Upload an image of a jute pest using the file uploader above
    2. Wait for the model to process and classify the image
    3. View the prediction results and confidence scores
    4. Check the top 3 predictions for alternative possibilities
    
    ### Tips for better results:
    - Use clear, well-lit images
    - Ensure the pest is the main subject of the image
    - Higher resolution images generally work better
    - Avoid blurry or heavily distorted images
    """)

if __name__ == "__main__":
    main()
