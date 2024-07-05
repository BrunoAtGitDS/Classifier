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
def create_model(weights_path=None):
    base_model = MobileNet(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    model = tf.keras.models.Sequential([
        base_model,
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
    
    if weights_path:
        model.load_weights(weights_path)
        st.write("Weights loaded successfully")
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

st.title("Image Classification App")
st.write("This app uses a pre-trained model to classify images.")

# Download weights from Google Drive
file_id = '1JSqm7NLCmOAqGnNJezaURwCPBQSgDCDm'  # Your Google Drive file ID
weights_path = 'weight.h5'  # Changed to match your file name

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
else:
    st.error("Weights file was not downloaded successfully.")

# Create the model using the downloaded weights
try:
    model = create_model(weights_path=weights_path)
    st.write("Model created successfully")
    st.write("Model summary:")
    model.summary(print_fn=lambda x: st.text(x))
except Exception as e:
    st.error(f"Error creating the model: {e}")
    model = None

uploaded_image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image_file is not None:
    image = Image.open(uploaded_image_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    def preprocess_image(image):
        img = image.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def make_prediction(image):
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        class_names = ['Benign', 'Malignant']
        return class_names[np.argmax(prediction[0])], prediction[0]

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
