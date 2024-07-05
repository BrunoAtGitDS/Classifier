import streamlit as st
import requests
import os

# Function to download file from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?id=" + id
    response = requests.get(URL)
    with open(destination, 'wb') as f:
        f.write(response.content)

st.title("File Download Diagnostic")

# Download file from Google Drive
file_id = '1JSqm7NLCmOAqGnNJezaURwCPBQSgDCDm'  # Your Google Drive file ID
file_path = 'downloaded_file.bin'  # Generic binary file name

if not os.path.exists(file_path):
    with st.spinner('Downloading file...'):
        try:
            download_file_from_google_drive(file_id, file_path)
            st.success("File downloaded successfully")
        except Exception as e:
            st.error(f"Error downloading file: {e}")

if os.path.exists(file_path):
    file_size = os.path.getsize(file_path)
    st.write(f"Downloaded file size: {file_size} bytes")
    
    # Read and display the first 100 bytes of the file
    with open(file_path, 'rb') as f:
        content = f.read(100)
    st.write("First 100 bytes of the file:")
    st.code(content)
    
    # Try to read the file as text
    try:
        with open(file_path, 'r') as f:
            text_content = f.read()
        st.write("File content (if it's a text file):")
        st.code(text_content)
    except UnicodeDecodeError:
        st.write("The file is not a text file.")
else:
    st.error("File was not downloaded successfully.")

st.write("Diagnostic complete.")
