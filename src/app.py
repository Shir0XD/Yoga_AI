import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

# Set Streamlit page configuration
st.set_page_config(
    page_title="Yoga Pose Classification",
    layout="centered",  # Center the content on the screen
    initial_sidebar_state="collapsed",
)

# Load the trained model
model = load_model('C:/Users/Shishir/OneDrive/Documents/Yoga_AI/image-classification-ai/models/model.keras')

# Define the classes based on folder names in the training directory
classes = sorted(os.listdir(r'C:\Users\Shishir\OneDrive\Documents\Yoga_AI\image-classification-ai\data\train')) 

def preprocess_image(image):
    """
    Preprocess the uploaded image to match the model's input requirements.
    """
    image = image.resize((150, 150))  # Resize image to match model input (150x150)
    image_array = np.array(image) / 255.0  # Normalize the image to [0, 1]
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

class YogaPoseDetector(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.classes = classes

    def transform(self, frame):
        # Convert the frame to a PIL image
        img = Image.fromarray(frame.to_ndarray(format="bgr24"))

        # Preprocess the image
        processed_image = preprocess_image(img)

        # Make prediction
        predictions = self.model.predict(processed_image)
        predicted_class = self.classes[np.argmax(predictions)]

        # Add the predicted class to the frame
        frame = frame.to_ndarray(format="bgr24")
        cv2.putText(
            frame,
            f"Pose: {predicted_class}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return frame

# Streamlit App
st.title("Yoga Pose Classification")
st.write("Choose an option below to classify yoga poses:")

# Add a sidebar for navigation
option = st.sidebar.selectbox(
    "Select Input Method",
    ("Upload an Image", "Use Camera")
)

if option == "Upload an Image":
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            processed_image = preprocess_image(image)

            # Make prediction
            predictions = model.predict(processed_image)
            predicted_class = classes[np.argmax(predictions)]

            # Display the predicted class
            st.success(f"Predicted Class: {predicted_class}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif option == "Use Camera":
    # Start the webcam and detect poses
    webrtc_streamer(
        key="yoga-pose-detection",
        video_transformer_factory=YogaPoseDetector,
        media_stream_constraints={"video": True, "audio": False},
    )