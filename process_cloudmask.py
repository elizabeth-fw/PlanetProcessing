import collections
import glob
import geopandas as gpd
import os
import re
import numpy as np
import rasterio

from cloud_clear.rapideye import RapidEye
from cloud_clear.planetscope_4band import PlanetScope4Band
from cloud_clear.planetscope_8band import PlanetScope8Band
from compositing import create_median_composite

def main():
    # Define directories
    base_dir = "X:/Aotea/Planet/"
    output_base_dir = "X:/Aotea/Planet/Output"
    aoi = gpd.read_file('X:/Aotea/Planet/AOI/aotea.shp').to_crs('EPSG:2193')

    # Create output subfolders
    os.makedirs(os.path.join(output_base_dir, "RapidEye"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "PlanetScope4Band"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "PlanetScope8Band"), exist_ok=True)

    # Get folders
    folders = glob.glob(f"{base_dir}/*")
    print(f"Found {len(folders)} folders in {base_dir}")

    # Dictionary to store processed files by year
    processed_files_by_year = {}

    # Loop through folders
    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        folder_name = os.path.basename(folder)
        match = re.match(r'^(20\d{2})-',folder_name)

        # Extract year
        if not match:
            print(f"Could not extract year from folder: {folder}. Skipping.")
            continue

        year = match.group(1)

        # --------- Rapid Eye -----------
        if 'reorthotile_analytic_sr' in folder.lower():
            processor_class = RapidEye
            subfolder = "RapidEye"
            image_pattern = f"{folder}/REOrthoTile/*Analytic_SR_clip_file_format.tif"

        # ------ Planet Scope ------
        elif 'psscene_analytic_8b_sr_udm2' in folder.lower(): # 8 band
            processor_class = PlanetScope8Band
            subfolder = "PlanetScope8Band"
            image_pattern = f"{folder}/PSScene/*AnalyticMS_SR_8b_harmonized_clip_file_format.tif"

        elif 'psscene_analytic_sr_udm2' in folder.lower(): # 4 Band
            processor_class = PlanetScope4Band
            subfolder = "PlanetScope4Band"
            image_pattern = f"{folder}/PSScene/*AnalyticMS_SR_harmonized_clip_file_format.tif"

        else:
            print(f"Unsupported folder: {folder}. Skipping.")
            continue

        # Get image files
        file_list = glob.glob(image_pattern)
        print(f"Found {len(file_list)} .tif files in {image_pattern}")

        # Create per-year output directory
        yearly_output_dir = os.path.join(output_base_dir, subfolder, year)
        os.makedirs(yearly_output_dir, exist_ok=True)

        # Process files
        for file in file_list:
            print(f"\nProcessing file: {file}")

            # Initialize processor
            processor = processor_class(
                tmp_dir=f"{folder}/tmp",
                output_dir = yearly_output_dir,
                aoi=aoi
            )

            # Build UDM file path
            if processor_class in [PlanetScope8Band, PlanetScope4Band]:
                udm_file = re.sub(r'AnalyticMS_SR(?:_8b)?_harmonized', 'udm2', file, count=1)
            else:
                udm_file = file.replace('Analytic_SR', 'udm')


            # Check if UDM file exists
            if not os.path.exists(udm_file):
                print(f"UDM file not found: {udm_file}. Skipping {file}")
                continue

            # Process Analytic file
            analytic_clipped = processor.reproject_and_clip(file, 'analytic')

            # Process UDM file
            udm_clipped = processor.reproject_and_clip(udm_file, 'udm')

            # Check alignment
            if not processor.check_file_properties(analytic_clipped, udm_clipped):
                print(f"Files not aligned: {analytic_clipped}, {udm_clipped}. Skipping.")
                continue

            # Store paths
            cleaned_paths = {
                # Individual masks
                #"udm": processor.udm_mask(udm_clipped, analytic_clipped),
                "udmbuffer": processor.udm_buffer_mask(udm_clipped, analytic_clipped, 25),
                #"cs": processor.cs_mask(analytic_clipped),
                #"lowcsbuffer": processor.apply_cs_buffer_mask(analytic_clipped, buffer_type="low"),
                #"highcsbuffer": processor.apply_cs_buffer_mask(analytic_clipped, buffer_type="high"),

                # Combinations
                "udm_lowcsbuffer": processor.combined_mask(analytic_clipped, udm_clipped, combo_type="udm_lowcsbuffer"),
                #"udmbuffer_lowcsbuffer": processor.combined_mask(analytic_clipped, udm_clipped, combo_type="udmbuffer_lowcsbuffer"),
                #"udmbuffer_cs": processor.combined_mask(analytic_clipped, udm_clipped, combo_type="udmbuffer_cs"),
                "udmbuffer_highcsbuffer": processor.combined_mask(analytic_clipped, udm_clipped, combo_type="udmbuffer_highcsbuffer")
            }

            # Add to year record
            for mask_type, path in cleaned_paths.items():
                processed_files_by_year.setdefault(year, []).append(path)


        # Cleanup -> Delete temp folders
        #if os.path.exists(processor.tmp_dir):
            #shutil.rmtree(processor.tmp_dir)
        #print(f"Temporary files deleted for {folder}.")

    # Print yearly file counts
    print("\nFiles grouped by year:")
    for year, files in processed_files_by_year.items():
        print(f"Year: {year}, Files: {len(files)} files.")


# ----- Generate composites ------
def create_composites(output_dir, aoi):
    composites_dir = os.path.join(output_dir, "composites")
    os.makedirs(composites_dir, exist_ok=True)

    # Find all cleaned files
    all_files = (
        glob.glob(os.path.join(output_dir, "RapidEye", "*", "*_cleaned.tif")) +
        glob.glob(os.path.join(output_dir, "PlanetScope4Band", "*", "*_cleaned.tif")) +
        glob.glob(os.path.join(output_dir, "PlanetScope8Band", "*", "*_cleaned.tif"))
    )

    # Organize by mask type and year
    composite_groups = collections.defaultdict(list)
    for tif in all_files:
        filename = os.path.basename(tif)

        # Extract mask type from the filename
        match = re.search(r'_clipped_([a-z_]+)_cleaned\.tif$', filename, re.IGNORECASE)
        if not match:
            continue
        mask_type = match.group(1)

        # Extract year from parent folder
        year = os.path.basename(os.path.dirname(tif))

        # Group by year and mask type
        key = f"{year}_{mask_type}"
        composite_groups[key].append(tif)

    # Generate composite for each combination
    created_count = 0
    for key in sorted(composite_groups):
        year, mask_type = key.split('_', 1)
        files = composite_groups[key]
        source = next((sensor for sensor in ["RapidEye", "PlanetScope4Band", "PlanetScope8Band"] if sensor in files[0]), "UnknownSensor")
        composite_path = os.path.join(composites_dir, f"{source}_median_{mask_type}_comp_{year}.tif")

        if os.path.exists(composite_path):
            print(f"Composite already exists: {composite_path}. Skipping.")
            continue

        print(f"Creating composite for {mask_type} mask in {year} from {len(files)} images....")
        if create_median_composite(files, composite_path, aoi=aoi):
            print(f"Composite created: {composite_path}")
            created_count += 1

    print(f"Total composites created: {created_count}")

# -------- Mosaicking ----------
def mosaic_images(image_paths, output_path, nodata_val=-9999):
    data_list = []
    meta = None

    for path in image_paths:
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            if meta is None:
                meta = src.meta.copy()
            nodata = src.nodata if src.nodata is not None else nodata_val
            data[data == nodata] = np.nan
            data_list.append(data)

    mosaic = data_list[-1]
    for data in reversed(data_list[:-1]):
        mask = ~np.isnan(data)
        mosaic = np.where(mask, data, mosaic)

    mosaic[np.isnan(mosaic)] = nodata_val
    meta.update({'dtype': 'float32', 'nodata': nodata_val})

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mosaic.astype('float32'))

    print(f"Mosaic saved to {output_path}")


def create_mosaic(output_dir):
    """
       Generate final mosaic for each year using composite outputs
    """
    composites_dir = os.path.join(output_dir, "composites")
    mosaic_output_dir = os.path.join(output_dir, "mosaics")
    os.makedirs(mosaic_output_dir, exist_ok=True)

    # Find composites grouped by year
    composites_by_year = collections.defaultdict(list)
    all_composites = glob.glob(os.path.join(composites_dir, "*.tif"))
    for comp in all_composites:
        filename = os.path.basename(comp)
        match = re.search(r'_comp_(\d{4})\.tif', filename)
        if match:
            year = match.group(1)
            composites_by_year[year].append(comp)

    # Create mosaic for each year
    for year, files in composites_by_year.items():
        # Prioritize specific composites
        priority_files = []
        for name in ["udmbuffer_highcsbuffer", "udm_lowcsbuffer", "udmbuffer"]:
            matches = [
                f for f in files
                    if f"median_{name}_comp_" in os.path.basename(f)
                ]
            priority_files.extend(matches)

        # Remove any accidental duplicates
        priority_files = list(dict.fromkeys(priority_files))

        if not priority_files:
            print(f"No matching composites to mosaic for {year}")
            continue

        print(f"Creating mosaic for {year} with {len(priority_files)} composites...")
        for f in priority_files:
            print(f" - {os.path.basename(f)}")

        # Determine prefix based on source
        if any("RapidEye" in os.path.basename(f) for f in priority_files):
            prefix = "re_mosaic"
        else:
            prefix = "ps_mosaic"

        mosaic_output_path = os.path.join(mosaic_output_dir, f"{prefix}_{year}.tif")

        if os.path.exists(mosaic_output_path):
            print(f"Mosaic already exists: {mosaic_output_path}. Skipping.")
            continue

        mosaic_images(priority_files, mosaic_output_path)


if __name__ == "__main__":
    aoi = gpd.read_file('X:/Aotea/Planet/AOI/aotea.shp').to_crs('EPSG:2193')
    main()
    create_composites("X:/Aotea/Planet/Output", aoi)
    create_mosaic("X:/Aotea/Planet/Output")
