import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import Adam
import requests
import os
import h5py

# Function to download file from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?id=" + id
    response = requests.get(URL)
    with open(destination, 'wb') as f:
        f.write(response.content)

st.title("Image Classification App - Diagnostic Mode")
st.write("This version of the app is for diagnosing issues with the model weights.")

# Download weights from Google Drive
file_id = '1JSqm7NLCmOAqGnNJezaURwCPBQSgDCDm'  # Your Google Drive file ID
weights_path = 'mobilenet_weights_no_top.h5'

if not os.path.exists(weights_path):
    with st.spinner('Downloading file...'):
        try:
            download_file_from_google_drive(file_id, weights_path)
            st.success("File downloaded successfully")
        except Exception as e:
            st.error(f"Error downloading file: {e}")

if os.path.exists(weights_path):
    file_size = os.path.getsize(weights_path)
    st.write(f"Downloaded file size: {file_size} bytes")
    
    # Try to open the file as HDF5
    try:
        with h5py.File(weights_path, 'r') as f:
            st.write("File is a valid HDF5 file.")
            st.write("Keys in the file:")
            st.write(list(f.keys()))
    except Exception as e:
        st.error(f"Error opening file as HDF5: {e}")
        
        # If it's not HDF5, let's check the first few bytes
        with open(weights_path, 'rb') as f:
            header = f.read(10)
        st.write(f"First 10 bytes of the file: {header}")
        
        # Try to load as a TensorFlow SavedModel
        try:
            model = tf.keras.models.load_model(weights_path)
            st.success("File loaded successfully as a TensorFlow SavedModel")
            st.write(model.summary())
        except Exception as e:
            st.error(f"Error loading as TensorFlow SavedModel: {e}")
        
else:
    st.error("File was not downloaded successfully.")

st.write("Diagnostic information gathering complete.")

 

# Define custom MobileNet model
def MobileNetmodelFS(weights_path=None):
    base_model = MobileNet(weights=None, include_top=False, input_shape=(224, 224, 3))
    if weights_path:
        base_model.load_weights(weights_path)
    return base_model

def create_model(weights_path=None):
    if weights_path:
        try:
            model = tf.keras.models.load_model(weights_path)
            st.write("Full model loaded successfully")
        except:
            st.write("Couldn't load full model, trying to load as custom model...")
            model = tf.keras.models.Sequential([
                MobileNetmodelFS(weights_path=weights_path),
                tf.keras.layers.GlobalAveragePooling2D(),
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
    else:
        model = tf.keras.models.Sequential([
            MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
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

if os.path.exists(weights_path):
    file_size = os.path.getsize(weights_path)
    st.write(f"Downloaded file size: {file_size} bytes")
    if file_size < 1000:  # Adjust this threshold as needed
        st.error("The downloaded file seems too small. It may not contain the weights.")
else:
    st.error("Weights file was not downloaded successfully.")

# Create the model using the downloaded weights
try:
    model = create_model(weights_path=weights_path)
    st.write("Model created successfully")
    st.write(f"Model summary: {model.summary()}")
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
        return class_names[np.argmax(prediction[0])], prediction[0]  # Get predicted class and probabilities

    if model:
        with st.spinner('Classifying...'):
            try:
                predicted_class, probabilities = make_prediction(image)
                st.success(f"Predicted Class: {predicted_class}")
                st.write(f"Probabilities: Benign: {probabilities[0]:.2f}, Malignant: {probabilities[1]:.2f}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.error("Model is not loaded properly.")
else:
    st.write("Please upload an image to classify.")
