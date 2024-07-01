import os
import sys
import subprocess
import argparse
import json
import easyocr
import time
import csv
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
import clip
from PIL import Image, UnidentifiedImageError
import numpy as np

# Global flag to signal script termination
terminate = False

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

def load_dependencies():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "clip", "Pillow", "easyocr"])

def load_models(device):
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    mlp = MLP(768).to(device)
    state_dict = torch.load("sac+logos+ava1-l14-linearMSE.pth", map_location=device)
    mlp.load_state_dict(state_dict)
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=torch.cuda.is_available())
    return clip_model, mlp, reader, preprocess

def predict_aesthetic_score(image_path, clip_model, mlp, reader, preprocess, device):
    try:
        image = Image.open(image_path)
        result = reader.readtext(np.array(image), detail=0)
        if result:
            return -1

        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
        
        image_features = image_features.to(mlp.layers[0].weight.dtype)
        score = mlp(image_features).item()
        return score

    except (UnidentifiedImageError, UnboundLocalError):
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_image(args):
    global terminate
    if terminate:
        return None, None, None
    filename, images_dir, threshold, clip_model, mlp, reader, preprocess, device = args
    image_path = os.path.join(images_dir, filename)
    score = predict_aesthetic_score(image_path, clip_model, mlp, reader, preprocess, device)

    if terminate:
        return None, None, None

    if score is None or score == -1:
        os.remove(image_path)
        txt_path = os.path.splitext(image_path)[0] + '.txt'
        if os.path.exists(txt_path):
            os.remove(txt_path)
        action = "Removed"
    elif score < threshold:
        os.remove(image_path)
        action = "Removed"
    else:
        action = "Kept"

    print(f"{action} image: {filename} (Aesthetic Score: {score})")
    return filename, score, action

def load_processed_images(save_state_file):
    if os.path.exists(save_state_file):
        with open(save_state_file, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_images(save_state_file, processed_images):
    with open(save_state_file, "w") as f:
        json.dump(list(processed_images), f)

def save_csv(aesthetic_scores, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Aesthetic Score', 'Action'])
        for filename, (score, action) in aesthetic_scores.items():
            writer.writerow([filename, score, action])

def signal_handler(signum, frame):
    global terminate
    print("\nReceived interrupt signal. Stopping the script properly...")
    terminate = True

def main():
    global terminate
    load_dependencies()
    Image.MAX_IMAGE_PIXELS = None

    parser = argparse.ArgumentParser(description="Remove images with low aesthetic scores.")
    parser.add_argument("-th", "--threshold", type=float, default=14.0, help="Threshold for removing images with low aesthetic scores.")
    parser.add_argument("-d", "--directory", default="D:\\Absurdres", help="Directory containing the images to process.")
    parser.add_argument("-t", "--threads", type=int, default=os.cpu_count(), help="Number of threads to use.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    clip_model, mlp, reader, preprocess = load_models(device)

    save_state_file = "save_state.json"
    csv_file = "aesthetic_scores.csv"
    processed_images = load_processed_images(save_state_file)
    aesthetic_scores = {}

    start_time = time.time()

    image_files = [f for f in os.listdir(args.directory) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f not in processed_images]

    # Set up signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_image, (filename, args.directory, args.threshold, clip_model, mlp, reader, preprocess, device)) for filename in image_files]
        
        for future in as_completed(futures):
            if terminate:
                executor.shutdown(wait=False)
                break
            filename, score, action = future.result()
            if filename is not None:
                aesthetic_scores[filename] = (score, action)
                processed_images.add(filename)

            if len(processed_images) % 100 == 0:
                save_processed_images(save_state_file, processed_images)
                save_csv(aesthetic_scores, csv_file)

            if terminate:
                break

    save_processed_images(save_state_file, processed_images)
    save_csv(aesthetic_scores, csv_file)

    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Summary: {sum(1 for _, action in aesthetic_scores.values() if action == 'Kept')} images kept, {sum(1 for _, action in aesthetic_scores.values() if action == 'Removed')} images removed")

if __name__ == "__main__":
    main()
