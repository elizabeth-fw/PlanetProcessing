# This code is written with the aid of ChatGPT and DeepSeek

import rasterio
import numpy as np
import os
import glob
import shutil  # For copying auxiliary files

# Define paths and constants
data_folder = 'Z:/Raw_data/Aotea/Planet/2023-planet_psscene_analytic_8b_sr_udm2/PSScene'
output_folder = 'Z:/Raw_data/Aotea/Planet/Outputs'
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

NODATA_VALUE = -9999  # Define a nodata value

# Get all image files
image_files = glob.glob(os.path.join(data_folder, '*_clip_file_format.tif'))
print(f"Found {len(image_files)} image files.")

# Loop through each image file
print("Processing the following unique image IDs:")
for image_file in image_files:
    # Extract the unique identifier (first 26 characters) to match UDM
    unique_id = os.path.basename(image_file)[:26]
    print(unique_id)  # Print the unique ID of the current image being processed

    udm_file = os.path.join(data_folder, f"{unique_id}_udm2_clip_file_format.tif")
    if not os.path.exists(udm_file):
        print(f"UDM file not found for {image_file}, skipping...")
        continue

    # Read the image and UDM file
    with rasterio.open(image_file) as src:
        image = src.read()  # Read all bands of the image
        profile = src.profile.copy()  # Get the image metadata for saving later
        crs = src.crs  # Store the CRS
        transform = src.transform  # Store the transform

    with rasterio.open(udm_file) as udm_src:
        # Read UDM bands: shadow (3), haze_light (4), haze_heavy (5), cloud (6)
        shadow = udm_src.read(3)
        haze_light = udm_src.read(4)
        haze_heavy = udm_src.read(5)
        cloud = udm_src.read(6)

        # Create a combined mask for flagged areas
        combined_mask = (shadow == 0) & (haze_light == 0) & (haze_heavy == 0) & (cloud == 0)

    # Apply the combined mask to the image data
    masked_image = np.full_like(image, NODATA_VALUE, dtype=np.float32)  # Initialize with nodata
    for band in range(image.shape[0]):
        # Keep valid pixels and mask flagged areas with nodata value
        masked_image[band] = np.where(combined_mask, image[band], NODATA_VALUE)

    # Update the profile for saving
    profile.update(
        dtype=rasterio.float32,
        count=image.shape[0],
        nodata=NODATA_VALUE,
        transform=transform,
        crs=crs
    )

    # Save the masked image to the output folder
    output_path = os.path.join(output_folder, f"{unique_id}_cloudmasked.tif")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(masked_image)

    print(f"Cloud-masked image saved to {output_path}")

    # Copy auxiliary metadata files
    for ext in ['.aux.xml', '.json', '_metadata_clip.xml']:
        aux_file = os.path.join(data_folder, f"{unique_id}{ext}")
        if os.path.exists(aux_file):
            shutil.copy(aux_file, output_folder)
            print(f"Copied auxiliary file: {aux_file}")

    # Copy additional files (e.g., UDM auxiliary files)
    udm_aux_file = os.path.join(data_folder, f"{unique_id}_udm2_clip_file_format.tif.aux.xml")
    if os.path.exists(udm_aux_file):
        shutil.copy(udm_aux_file, output_folder)
        print(f"Copied UDM auxiliary file: {udm_aux_file}")
