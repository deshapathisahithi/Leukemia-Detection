import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tkinter as tk
from tkinter import filedialog, messagebox

# Load Pre-trained Model (ResNet50)
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features from an image
def extract_features(img_path):
    if not os.path.isfile(img_path):  # Ensure it's a file
        print(f"Skipping: {img_path} (Not a file)")
        return None  

    try:
        img = image.load_img(img_path, target_size=(224, 224))  # Resize image
        x = image.img_to_array(img)  # Convert to array
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        x = preprocess_input(x)  # Preprocess for ResNet50
        
        features = base_model.predict(x)  # Extract features
        return features.flatten()  # Flatten output
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Function to process images in a directory
def process_directory():
    folder_path = filedialog.askdirectory(title="Select Image Folder")
    
    if not folder_path:
        messagebox.showwarning("Warning", "No folder selected!")
        return
    
    features_list = []
    
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        
        # Check if it's an image file
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing: {img_path}")
            features = extract_features(img_path)
            if features is not None:
                features_list.append(features)
        else:
            print(f"Skipping (not an image): {img_path}")

    messagebox.showinfo("Completed", f"Processed {len(features_list)} images successfully!")

# GUI Setup (Tkinter)
root = tk.Tk()
root.title("Leukemia Image Feature Extractor")
root.geometry("400x200")

btn_select = tk.Button(root, text="Select Image Folder", command=process_directory, padx=10, pady=5)
btn_select.pack(pady=20)

root.mainloop()
