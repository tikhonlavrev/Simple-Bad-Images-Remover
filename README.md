# July 1st 2024 update

We've update the script into better scoring performance and more stable right now. There is changes includding:

- OCR library changes, from pytesseract into easyOCR for faster OCR performace
- Normalized scoring, minus score only appeared when text is detected
- Multi-threading that I hope is running (it eat my 40 cores CPU though, so I don't know if this will working in other machine)
- Save state function
- Aesthetic score is outputed in CSV format

# Simple-Bad-Images-Remover

This NC-AI™ Simple-Bad-Images-Remover build from CLIP+MLP architecture by Python script. This script is using Aesthetic scoring method to removing bad images from the folder by utilizing CLIP "ViT-L/14". The script is adapted from [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) using pre-trained MLP, sac+logos+ava1-l14-linearMSE.pth 

## Prerequisites

Make sure you have Python and CUDA toolkit 11.7 with torch for CUDA 11.7. If you already have one that run on different version maybe the NC-AI™ Simple Bad Images Remover will be run from CPU

## Installation

Just clone the this repository using git clone and run using python and . I will create the bat file soon for single click run from windows

## Usage

To launch the Simple Bad Images Remover, run the following command:
python remove_bad_images.py

By default, the threshold configuration is set to 14. You can change it using the `-th` argument and choose your directory using `-th`. For example, if you want to use 16 as the threshold:

```sh
python remove_bad_images.py -th 16 -d <bad_images_location>
```
Replace `<bad_images_location>` with the path to the directory containing your bad images. The supported image format is .jpg, .jpeg, and .png

## Recommended Threshold
I've tested with some picture, here the recommended threshold so far:
Anime Style: 14 (It's enough for acceptable looking art)
