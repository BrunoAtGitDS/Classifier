import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from PIL import Image
import numpy as np
import os
from tensorflow.keras.optimizers import Adam

# Define custom MobileNet model
def MobileNetmodelFS(local_weights_path=None):
    try:
        if local_weights_path:
            base_model = MobileNet(weights=local_weights_path, include_top=False, input_shape=(224, 224, 3))
        else:
            base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        return base_model
    except Exception as e:
        st.error(f"Error loading MobileNet model: {e}")
        raise

def create_model(local_weights_path=None):
    model = tf.keras.models.Sequential([
        MobileNetmodelFS(local_weights_path=local_weights_path),
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
    try:
        model.build((None, 224, 224, 3))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        st.error(f"Error building or compiling the model: {e}")
        raise
    return model

# Create the model
try:
    local_weights_path = 'path_to_your_local_mobilenet_weights_no_top.h5'
    model = create_model(local_weights_path=local_weights_path)
except Exception as e:
    st.error(f"Error creating the model: {e}")
    model = None

if model:
    st.write("Model loaded successfully")

# Function to preprocess the image for the model
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

st.title("Image Classification App")
st.write("Upload an image to classify using your trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if model:
        with st.spinner('Classifying...'):
            try:
                prediction = make_prediction(image)
                st.success(f"Predicted Class: {prediction}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.error("Model is not loaded properly.")
