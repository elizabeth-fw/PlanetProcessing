"""
Batch Remap Slip Class Rasters for Landslide Segmentation
----------------------------------------------------------
This script reads rasterized slip label files (.tif), remaps class values,
and saves the updated rasters.

Original slip classes:
    0 control (known landslides, mapped every image)
    1 high confidence
    2 medium
    3 low
    4 bare
    5 error (cloud/shadow/satellite-error)
    8 forest
    9 urban

Remapped classes:
    0 = background / stable (from 99)
    1 = merged landslide (from 0, 1)
    2 = medium landslide (from 2)
    3 = low landslide (from 3)
    4 = forest or non-landslide (from 8, 9)
    5 = Error

Usage:
    - To remap all rasters in a folder and write to a new folder.
    - Set `overwrite=True` to overwrite original files.
"""

import os
import rasterio
import numpy as np
from glob import glob

def remap_classes(input_raster, output_raster, overwrite=False):
    """
    Remap slip classes in a raster

    Args:
        input_raster (str): Path to original slip label raster
        output_raster (str): Output path to save remapped raster
        overwrite (bool): If True, replace input raster
    """
    with rasterio.open(input_raster) as src:
        label = src.read(1)
        meta = src.meta.copy()

    # Update metadata for GeoAI segmentation masks
    meta.update(dtype=rasterio.uint8, count=1)

    # Remap logic
    remapped = np.copy(label)
    remapped[np.isin(label, [0, 1])] = 1  # High confidence
    remapped[label == 2] = 2 # Medium confidence
    remapped[label == 3] = 3 # Low confidence
    remapped[np.isin(label, [4, 8, 9])] = 4  # Not a landslide (bare, forest, urban)
    remapped[label == 5] = 5 # Error
    remapped[np.isin(label, [99, 255])] = 0  # Background

    # Determine output path
    out_path = output_raster if not overwrite else input_raster

    # Write remapped raster
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(remapped, 1)

    print(f"Saved remapped raster: {out_path}")

def batch_remap_slip_labels(input_dir, output_dir=None, overwrite=False):
    """
    Remap all .tif slip rasters in a directory.

    Args:
        input_dir (str): Folder containing original slip label rasters
        output_dir (str): Folder to save remapped rasters (ignored if overwrite=True)
        overwrite (bool): If True, modifies input rasters in place
    """
    if not overwrite:
        os.makedirs(output_dir, exist_ok=True)

    slip_files = sorted(glob(os.path.join(input_dir, "*.tif")))

    for path in slip_files:
        filename = os.path.basename(path)
        output_path = path if overwrite else os.path.join(output_dir, filename)
        remap_classes(path, output_path, overwrite=overwrite)

# ------------------------------ Run This ------------------------------
if __name__ == "__main__":
    input_dir = "/Users/brookeengland/Documents/Internship/Project/Training Data/Rasterized"
    output_dir = "/Users/brookeengland/Documents/Internship/Project/GeoAI/Remapped"

    batch_remap_slip_labels(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=False  # Set to True to overwrite original rasters
    )
