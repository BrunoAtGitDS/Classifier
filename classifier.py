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
