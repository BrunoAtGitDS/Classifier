import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import Adam
import tempfile
import os
import py7zr

import streamlit as st 

st.config.set_option("server.maxUploadSize", 5000)
st.config.set_option("server.maxMessageSize", 5000)
st.config.set_option("server.enableWebsocketCompression", 'true')


def create_model(weights_file_path=None):
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
    
    if weights_file_path:
        try:
            model.load_weights(weights_file_path)
            st.write("Weights loaded successfully")
        except Exception as e:
            st.error(f"Error loading weights: {e}")

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

st.title("Image Classification App")
st.write("This app uses a pre-trained model to classify images.")

# File uploader for 7z split chunks
uploaded_chunks = []
chunk_number = 1
while True:
    chunk = st.file_uploader(f"Upload weight chunk {chunk_number} (.{chunk_number:03d}.7z)", type=['7z'])
    if chunk is None:
        break
    uploaded_chunks.append(chunk)
    chunk_number += 1

# Ensure all chunks are combined
if uploaded_chunks:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save each uploaded chunk to temporary files
            temp_files = []
            for i, chunk in enumerate(uploaded_chunks):
                temp_chunk_path = os.path.join(temp_dir, f"modelFS.weights.7z.{i+1:03d}")
                with open(temp_chunk_path, 'wb') as f:
                    f.write(chunk.read())
                temp_files.append(temp_chunk_path)

            # Combine 7z split files
            combined_7z_path = os.path.join(temp_dir, "combined_weights.7z")
            with open(combined_7z_path, 'wb') as combined_7z:
                for temp_file in temp_files:
                    with open(temp_file, 'rb') as f:
                        combined_7z.write(f.read())

            # Debug: Check if the combined 7z file is created and its size
            st.write(f"Combined 7z file size: {os.path.getsize(combined_7z_path)} bytes")

            # Extract the combined 7z file to get the .h5 file
            extracted_h5_path = None
            with py7zr.SevenZipFile(combined_7z_path, mode='r') as archive:
                extracted_files = archive.extractall(path=temp_dir)
                extracted_files = extracted_files['files']
                st.write(f"Extracted files: {extracted_files}")

                for file in extracted_files:
                    if file.endswith('.h5'):
                        extracted_h5_path = os.path.join(temp_dir, file)
                        break

            if extracted_h5_path:
                # Create the model using the extracted weights file
                model = create_model(weights_file_path=extracted_h5_path)
                st.write("Model created successfully")
                st.write("Model summary:")
                model.summary(print_fn=lambda x: st.text(x))
            else:
                st.error("No .h5 file found in the extracted files.")

    except Exception as e:
        st.error(f"Error processing the weights: {e}")
        model = None

else:
    st.warning("Please upload the weight chunks.")
    model = None

# Image uploader and classification
uploaded_image_file = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])

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
    
