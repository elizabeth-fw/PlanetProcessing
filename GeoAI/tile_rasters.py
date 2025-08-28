"""
GeoAI Tiling Script for Landslide Classification
-------------------------------------------------
This script breaks large mosaic rasters and their corresponding
rasterized slip label masks into smaller tiles (512x512 pixels).
These tiles are compatible with GeoAI's semantic segmentation algorithm.

Input:
    - Multi-band mosaic image (.tif)
    - Single-band rasterized slip mask (.tif)
Output:
    - /images/*.tif
    - /labels/*.tif
Parameters:
    - tile_size: Size of each square tile (default = 512)
    - stride: Step size between tiles (default = 256)
"""
# Imports
import os
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm

def tile_rasters(image_path, label_path, out_dir, tile_size=512, stride=256):
    """
        Tiles a Sentinel-2 image and corresponding slip label raster

        Args:
            image_path (str): Path to the full mosaic image (.tif)
            label_path (str): Path to the full rasterized slip mask (.tif)
            out_dir (str): Output directory to save tiled images and labels
            tile_size (int): Size of square tiles (default = 512)
            stride (int): Stride between tiles (default = 256)
        """
    # Create output folders
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)

    # Open the raster files
    with rasterio.open(image_path) as src_img, rasterio.open(label_path) as src_lbl:
        img_width, img_height = src_img.width, src_img.height

        # Iterate through rasters
        for i in tqdm(range(0, img_height - tile_size + 1, stride)):
            for j in range(0, img_width - tile_size + 1, stride):
                window = Window(i, j, tile_size, tile_size)

                # Read image and label tiles within each window
                img_tile = src_img.read(window=window)
                lbl_tile = src_lbl.read(1, window=window)

                # Skip empty label tiles
                if np.all(img_tile == 0):
                    continue

                tile_id = f"tile_{i}_{j}"

                # Write image tile
                img_meta = src_img.meta.copy()
                img_meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": src_img.window_transform(window)
                })
                out_img = os.path.join(out_dir, "images", f"{tile_id}.tif")
                with rasterio.open(out_img, "w", **img_meta) as dst:
                    dst.write(img_tile)

                # Write label tile
                lbl_meta = src_lbl.meta.copy()
                lbl_meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": src_lbl.window_transform(window),
                    "count": 1
                })
                out_lbl = os.path.join(out_dir, "labels", f"{tile_id}.tif")
                with rasterio.open(out_lbl, "w", **lbl_meta) as dst:
                    dst.write(lbl_tile, 1)

    print(f"Tiles written to {out_dir}/images and /lables")

if __name__ == "__main__":
    image_tif = "/Users/brookeengland/Documents/Internship/Project/Training Data/Aotea_S2/S2_mosaic_2018.tif"
    label_tif = "/Users/brookeengland/Documents/Internship/Project/Training Data/Remapped/S2_2018_rasterized_slips.tif"
    out_folder = "/Users/brookeengland/Documents/Internship/Project/GeoAI/tiles/geoai_tiles_2018"

    tile_rasters(image_tif, label_tif, out_folder, tile_size=512, stride=256)


