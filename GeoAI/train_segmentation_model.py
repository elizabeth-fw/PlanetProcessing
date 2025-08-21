"""
Train Landslide Segmentation Model using GeoAI
----------------------------------------------

This script trains a semantic segmentation model using tiled mosaic images
and rasterized landslide labels. The output is a trained model and performance plots.

Requirements:
- geoai
- segmentation-models-pytorch

Inputs:
- Tile directory with images/ and labels/
- Parameters: architecture, encoder, band count, class count

Outputs:
- Trained model weights (.pth)
- Loss/accuracy plots
"""

import geoai
import os

def main():
    # ----------------- Parameters -----------------
    # Tile folders
    tiles_dir = "/Users/brookeengland/Documents/Internship/Project/GeoAI/tiles/geoai_tiles_2018"
    images_dir = os.path.join(tiles_dir, "images")
    labels_dir = os.path.join(tiles_dir, "labels")

    # Output folder for trained models
    output_dir = "/Users/brookeengland/Documents/Internship/Project/GeoAI/models/unet_2018"

    # Sentinel-2 configuration
    num_channels = 26        # Number of bands in S2 mosaic (e.g., B2–B8A, B11–B12)
    num_classes = 5
    batch_size = 8
    num_epochs = 2
    learning_rate = 0.0005
    val_split = 0.2

    # Model architecture
    architecture = "unet"
    encoder_name = "resnet34"
    encoder_weights = "imagenet"

    # -------------------- Train Model --------------------------

    print("📦 Starting GeoAI model training...")

    geoai.train_segmentation_model(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        num_channels=num_channels,
        num_classes=num_classes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        val_split=val_split,
        verbose=True,
        plot_curves=True
    )

    print(f"Model training complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
