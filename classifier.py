import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import Adam
import tempfile
import os
import traceback
st.config.set_option("server.maxUploadSize", 1024)
st.config.set_option("server.maxMessageSize", 1024)
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
            st.error(traceback.format_exc())

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

### Streamlit Application Code

st.title("Image Classification App")
st.write("This app uses a pre-trained model to classify images.")

# File uploader for weights file
uploaded_weights_file = st.file_uploader("Upload the weights .h5 file", type='h5')

if uploaded_weights_file is not None:
    st.write("Weights file uploaded successfully!")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_file:
            temp_file.write(uploaded_weights_file.getbuffer())
            weights_file_path = temp_file.name
        
        model = create_model(weights_file_path=weights_file_path)
        st.write("Model created successfully")
        st.write("Model summary:")
        model.summary(print_fn=lambda x: st.text(x))
    except Exception as e:
        st.error(f"Error processing the weights file: {e}")
        st.error(traceback.format_exc())
        model = None
else:
    model = None
    st.warning("Please upload the weights .h5 file.")

# Image uploader and classification
uploaded_image_file = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])

if uploaded_image_file is not None:
    if model is None:
        st.warning("Please upload the weights file first.")
    else:
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

        with st.spinner('Classifying...'):
            try:
                predicted_class, probabilities = make_prediction(image)
                st.success(f"Predicted Class: {predicted_class}")
                st.write(f"Probabilities: Benign: {probabilities[0]:.2f}, Malignant: {probabilities[1]:.2f}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
else:
    st.write("Please upload an image to classify.") 
