import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = 'C:/python2025/Internship_projects/project1/Xception_best_model.keras'
model = load_model(MODEL_PATH)

# Define class labels and remedies
CLASS_LABELS = ['Eczema', 'Melanoma', 'Psoriasis', 'Tinea Ringworm', 'Melanocytic Nevi']
REMEDIES = {
	'Eczema': "Keep skin moisturized, avoid triggers like allergens, and use prescribed creams.",
	'Melanoma': "Seek medical attention immediately for diagnosis and treatment options.",
	'Psoriasis': "Use medicated shampoos/creams and avoid triggers like stress.",
	'Tinea Ringworm': "Apply antifungal creams and maintain proper hygiene.",
	'Melanocytic Nevi': "Regularly monitor moles and consult a dermatologist if changes occur."
	}

# Function to preprocess image
def preprocess_image(image, target_size=(299, 299)):
	image = image.resize(target_size)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
	return image

# Function to predict the skin disease
def predict_disease(image):
	processed_image = preprocess_image(image)
	predictions = model.predict(processed_image)
	predicted_class = np.argmax(predictions, axis=1)[0]
	confidence = np.max(predictions)
	return CLASS_LABELS[predicted_class], confidence



# Sidebar with image and description
st.sidebar.image('C:/python2025/Internship_projects/project1/image.png', caption="Dermatrix",use_column_width=True)
st.sidebar.markdown(
    "<div class='sidebar-text'>Accurate detection of skin diseases with remedies suggestions to help manage them effectively.</div>",
    unsafe_allow_html=True
)

st.markdown('<div class="title">Skin Disease Detection and with Remedey Suggestion</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image or use your webcam to detect skin diseases and get remedies.</div>', unsafe_allow_html=True)

# Image input
upload_option = st.radio("Choose input method:", ('Upload Image', 'Use Webcam'))

if upload_option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        predict_button = st.button("Predict Disease")
        if predict_button:
            disease, confidence = predict_disease(image)
            st.write(f"**Predicted Disease:** {disease}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")
            st.write(f"**Suggested Remedy:** {REMEDIES[disease]}")

elif upload_option == 'Use Webcam':
    st.write("Enable webcam access below to capture an image.")
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption='Captured Image', use_column_width=True)
        predict_button = st.button("Predict Disease")
        if predict_button:
            disease, confidence = predict_disease(image)
            st.write(f"**Predicted Disease:** {disease}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")
            st.write(f"**Suggested Remedy:** {REMEDIES[disease]}")
