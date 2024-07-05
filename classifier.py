import streamlit as st
import requests
import os

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

st.title("Model Weights Download and Verification")

file_id = '1JSqm7NLCmOAqGnNJezaURwCPBQSgDCDm'  # Your Google Drive file ID
weights_path = 'modelFS.weights.h5'  # Use the actual file name

if not os.path.exists(weights_path):
    with st.spinner('Downloading model weights... This may take a while for a 309MB file.'):
        try:
            download_file_from_google_drive(file_id, weights_path)
            st.success("Model weights downloaded successfully")
        except Exception as e:
            st.error(f"Error downloading model weights: {e}")

if os.path.exists(weights_path):
    file_size = os.path.getsize(weights_path)
    st.write(f"Downloaded file size: {file_size} bytes")
    if file_size > 300000000:  # Check if file is close to the expected 309MB
        st.success("File size looks correct. You should now be able to use this file to load your model weights.")
    else:
        st.warning("File size is smaller than expected. There might still be an issue with the download.")
else:
    st.error("Weights file was not downloaded successfully.")

st.write("Download and verification complete.")
