"""
Run Landslide Segmentation Inference using GeoAI
-------------------------------------------------

This script loads a trained semantic segmentation model and applies it
to a full mosaic to generate a landslide classification map.

Requirements:
- geoai
- segmentation-models-pytorch

Inputs:
- Full S2 mosaic (multi-band GeoTIFF)
- Trained model (.pth)
- Model parameters: architecture, encoder, bands, classes

Outputs:
- Classified raster (.tif) with predicted classes
"""

import geoai
import os

# ----------------- Parameters -----------------

# Input Sentinel-2 mosaic (same band order used in training)
mosaic_path = "/Users/brookeengland/Documents/Internship/Project/Training Data/Aotea_S2/S2_mosaic_2018.tif"

# Trained model (.pth)
model_path = "/Users/brookeengland/Documents/Internship/Project/GeoAI/models/unet_2018/best_model.pth"

# Output prediction file
output_path = "/Users/brookeengland/Documents/Internship/Project/GeoAI/predictions/S2_2018_geoai_prediction.tif"

# Model configuration (must match training)
architecture = "unet"
encoder_name = "resnet34"
num_channels = 26
num_classes = 5

# Inference tiling window (match training tile size and stride)
window_size = 512
overlap = 256
batch_size = 4

# --------------------- Run Inference ------------------------

print("Running semantic segmentation prediction...")

geoai.semantic_segmentation(
    input_path=mosaic_path,
    output_path=output_path,
    model_path=model_path,
    architecture=architecture,
    encoder_name=encoder_name,
    num_channels=num_channels,
    num_classes=num_classes,
    window_size=window_size,
    overlap=overlap,
    batch_size=batch_size
)

print(f"Prediction complete. Output saved to:\n{output_path}")
