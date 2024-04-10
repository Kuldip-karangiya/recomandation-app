import streamlit as st

import os
import shutil
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
import tensorflow
import pickle
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

# Mukund edited
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import tqdm


st.title('Recommender System')

def extract_features(img_path, model):
    img = Image.open(img_path).convert("L")  # Open image in grayscale mode
    img = img.convert("RGB")  # Convert grayscale to RGB
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def train_model():
    Image.MAX_IMAGE_PIXELS = 202662810
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    filenames = []

    for file in os.listdir('FolderUploads'):
        filenames.append(os.path.join('FolderUploads', file))

    feature_list = []

    for file in tqdm.tqdm(filenames):
        feature_list.append(extract_features(file, model))

    pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
    pickle.dump(filenames, open('filename.pkl', 'wb'))

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)
st.write('Please select a folder to train new model:')
clicked = st.button('Browse Folder')

folder_path = ""
errors = []

if clicked:
    dirname = str(filedialog.askdirectory(master=root))
    folder_path = dirname
    st.write(f"Selected folder: {folder_path}")
    if not dirname:  # Check if no folder was selected
        st.warning("No folder selected. Please select a folder.")
    else:
        destination_folder = "FolderUploads" # Folder name where to upload images
        shutil.rmtree(destination_folder, ignore_errors=True) # Truncate folder before uploading files

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        images_reports = [file for file in os.listdir(dirname) if os.path.splitext(file)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif']]
        # Copy files to the 'images' folder with error handling
        for file in images_reports:
            try:
                shutil.copy(os.path.join(dirname, file), destination_folder)
            except Exception as e:
                errors.append(f"Error copying file '{file}': {str(e)}")
        
        if not errors:
            with st.spinner("Please be patient while we train the model..."):
                train_model()
            st.success("Model trained successfully ✅")
        else:
            st.error("Error while copying files..! Please try again.")

# Load pre-trained model and features if FolderUploads directory exists
if os.path.exists("FolderUploads"):
    try:
        feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
        filenames = pickle.load(open('filename.pkl', 'rb'))
    except FileNotFoundError:
        st.write("Pickle files not found. Training models...")
        train_model()
        feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
        filenames = pickle.load(open('filename.pkl', 'rb'))

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])


def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            while True:
                # Read 10 MB chunks of the file at a time
                chunk = uploaded_file.read(10 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return file_path
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def feature_extraction(img_path, model):
    img = Image.open(img_path).convert("L")  # Open image in grayscale mode
    img = img.convert("RGB")  # Convert grayscale to RGB
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    folder_path = "FolderUploads"  # Path to the folder
    if os.path.exists(folder_path):  # Check if the folder exists
        file_count = len(os.listdir(folder_path))
        if file_count < 20:
            neighbors = file_count
        else:
            neighbors = 20

    neighbors = NearestNeighbors(n_neighbors=neighbors, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader('Choose a Image file')

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path is not None:
        st.image(uploaded_file)
        check_clicked = st.button("Check")
        if check_clicked:
            st.write("Generating recommendations...")
            # Simulated animation
            with st.spinner("Please wait while recommendations are generated..."):
                feature = feature_extraction(file_path, model)
                indices = recommend(feature, feature_list)
            st.success("Recommendations generated!")

            # Displaying images
            num_rows = 4
            num_cols = 5
            col_width = 1 / num_cols

            for i in range(num_rows):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    index = i * num_cols + j
                    if index < len(indices[0]):
                        cols[j].image(filenames[indices[0][index]], use_column_width=True)
                        cols[j].write(os.path.basename(filenames[indices[0][index]]))
                    else:
                        cols[j].write("")  # Empty placeholder for cells without images
