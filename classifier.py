import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import Adam
import io
import os
import tempfile

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

# File uploader for weight chunks
uploaded_chunks = []
chunk_number = 1
while True:
    chunk = st.file_uploader(f"Upload weight chunk {chunk_number} (.{chunk_number:03d})", type=['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013'])
    if chunk:
        uploaded_chunks.append(chunk)
        chunk_number += 1
    else:
        break

if uploaded_chunks:
    # Combine chunks and save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.weights.h5') as temp_file:
        for chunk in uploaded_chunks:
            temp_file.write(chunk.read())
        temp_file_path = temp_file.name

    st.write(f"Combined weights size: {os.path.getsize(temp_file_path)} bytes")
    
    # Create the model using the weights
    try:
        model = create_model(weights_path=temp_file_path)
        st.write("Model created successfully")
        st.write("Model summary:")
        model.summary(print_fn=lambda x: st.text(x))
    except Exception as e:
        st.error(f"Error creating the model: {e}")
        model = None
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)
else:
    st.warning("Please upload the weight chunks.")
    model = None

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
