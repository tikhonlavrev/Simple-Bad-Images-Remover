import os
import sys
import subprocess
import argparse
import urllib.request
import json

# Install required dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "clip", "Pillow"])

# Download the pre-trained MLP model
model_url = "https://huggingface.co/camenduru/improved-aesthetic-predictor/resolve/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
model_path = "sac+logos+ava1-l14-linearMSE.pth"
if not os.path.exists(model_path):
    urllib.request.urlretrieve(model_url, model_path)

import torch
import torch.nn as nn
import clip
from PIL import Image

# Remove pixel limitation
Image.MAX_IMAGE_PIXELS = None

# Set the device to run on: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CLIP model
clip_model, preprocess = clip.load("ViT-L/14", device=device)

# Define the MLP class with the updated architecture
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Create an instance of the MLP model with the updated architecture
mlp = MLP(768).to(device)

# Load the state dictionary of the pre-trained MLP model into the MLP model instance
state_dict = torch.load(model_path, map_location=device)
mlp.load_state_dict(state_dict)

# Define a function to predict the aesthetic score of an image using the CLIP+MLP Aesthetic Score Predictor
def predict_aesthetic_score(image_path):
    try:
        # Load the image and preprocess it using the CLIP preprocessing function
        image = Image.open(image_path)
        image_tensor = preprocess(image).unsqueeze(0).to(device)
    except UnidentifiedImageError:
        print(f"Warning: Unable to open image file {image_path} skipping the process")
        return None
    
    # Compute the CLIP embeddings for the image
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
    
    # Convert the image_features tensor to the same data type as the MLP model's parameters
    image_features = image_features.to(mlp.layers[0].weight.dtype)
    
    # Use the MLP to predict the aesthetic score of the image based on its CLIP embeddings
    score = mlp(image_features).item()
    
    return score

def main():
    parser = argparse.ArgumentParser(description="Remove images with low aesthetic scores.")
    parser.add_argument("-th", "--threshold", type=float, default=14.0, help="Threshold for removing images with low aesthetic scores.")
    parser.add_argument("-d", "--directory", default="D:\\Absurdres", help="Directory containing the images to process.")
    args = parser.parse_args()

    threshold = args.threshold
    images_dir = args.directory
    save_state_file = "save_state.json"

    # Load the saved state from file, if it exists
    if os.path.exists(save_state_file):
        try:
            with open(save_state_file, "r") as f:
                processed_images = set(json.load(f))
        except json.JSONDecodeError:
            print(f"Warning: Unable to load saved state from {save_state_file} the worker start from beginning")
            processed_images = set()
    else:
        processed_images = set()

    # Loop through all the files in the directory
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Skip images that have already been processed
            if filename in processed_images:
                continue

            image_path = os.path.join(images_dir, filename)
            score = predict_aesthetic_score(image_path)
            
            if score < threshold:
                # Remove both the image and its corresponding .txt file with same filename.
                os.remove(image_path)
                txt_path = os.path.splitext(image_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                print(f"Removed image: {filename} (Aesthetic Score: {score})")
            else:
                print(f"Kept image: {filename} (Aesthetic Score: {score})")

            # Add the image to the set of processed images and save the updated state to file
            processed_images.add(filename)
            with open(save_state_file, "w") as f:
                json.dump(list(processed_images), f)

if __name__ == "__main__":
    main()
