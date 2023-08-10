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
