import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import Adam
import requests
import os

# Function to download file from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?id=" + id
    response = requests.get(URL)
    with open(destination, 'wb') as f:
        f.write(response.content)

# Define custom MobileNet model
def MobileNetmodelFS(weights_path=None):
    if weights_path:
        base_model = MobileNet(weights=None, include_top=False, input_shape=(224, 224, 3))
        base_model.load_weights(weights_path)
    else:
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return base_model

def create_model(weights_path=None):
    model = tf.keras.models.Sequential([
        MobileNetmodelFS(weights_path=weights_path),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.build((None, 224, 224, 3))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

st.title("Image Classification App")
st.write("This app uses a pre-trained model to classify images.")

# Download weights from Google Drive
file_id = '1JSqm7NLCmOAqGnNJezaURwCPBQSgDCDm'  # Extracted from your Google Drive link
weights_path = 'mobilenet_weights_no_top.h5'

if not os.path.exists(weights_path):
    with st.spinner('Downloading model weights...'):
        try:
            download_file_from_google_drive(file_id, weights_path)
            st.success("Model weights downloaded successfully")
        except Exception as e:
            st.error(f"Error downloading model weights: {e}")

# Create the model using the downloaded weights
try:
    model = create_model(weights_path=weights_path)
    st.write("Model loaded successfully")
except Exception as e:
    st.error(f"Error creating the model: {e}")
    model = None

uploaded_image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image_file is not None:
    image = Image.open(uploaded_image_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    def preprocess_image(image):
        img = image.convert('RGB')  # Ensure RGB format
        img = img.resize((224, 224))  # Resize based on model input
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    def make_prediction(image):
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        class_names = ['Benign', 'Malignant']  # Replace with your actual class names
        return class_names[np.argmax(prediction[0])]  # Get predicted class

    if model:
        with st.spinner('Classifying...'):
            try:
                prediction = make_prediction(image)
                st.success(f"Predicted Class: {prediction}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.error("Model is not loaded properly.")
else:
    st.write("Please upload an image to classify.")
