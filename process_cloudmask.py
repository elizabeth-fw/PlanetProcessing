import glob
import geopandas as gpd
import shutil
import os
from cloud_clear.planetscope import PlanetScope
from cloud_clear.rapideye import RapidEye
from compositing import create_median_composite

# Define directories
base_dir = "/Users/brookeengland/Documents/Internship/Project/Planet/Planet Practice Files"
output_base_dir = "/Users/brookeengland/Documents/Internship/Project/Planet Output"
aoi = gpd.read_file('/Users/brookeengland/Documents/Internship/Project/Planet/aotea/aotea.shp').to_crs('EPSG:2193')

# Create output subfolders
os.makedirs(os.path.join(output_base_dir, "RapidEye"), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, "PlanetScope"), exist_ok=True)

# Get folders
folders = glob.glob(f"{base_dir}/*")
print(f"Found {len(folders)} folders in {base_dir}")

# Dictionary to store processed files by year
processed_files_by_year = {}

for folder in folders:
    print(f"\nProcessing folder: {folder}")
    
    if 'psscene' in folder.lower():
        # PlanetScope processing (unchanged)
        file_list = glob.glob(f"{folder}/PSScene/*.tif")
        print(f"Found {len(file_list)} .tif files in {folder}/PSScene")
        processor = PlanetScope(
            tmp_dir=f"{folder}/tmp",
            output_dir=os.path.join(output_base_dir, "PlanetScope"),
            aoi=aoi
        )
    elif 'reorthotile' in folder.lower():
        # RapidEye processing with custom cloud scoring
        file_list = glob.glob(f"{folder}/REOrthoTile/*.tif")
        print(f"Found {len(file_list)} .tif files in {folder}/REOrthoTile")
        processor = RapidEye(
            tmp_dir=f"{folder}/tmp",
            output_dir=os.path.join(output_base_dir, "RapidEye"),
            aoi=aoi
        )
        processor.use_custom_cloud_score = True  # Activate custom scoring
    else:
        print(f"Skipping unsupported folder: {folder}")
        continue

    for file in file_list:
        print(f"\nProcessing file: {file}")
        
        # Find UDM file (still needed for metadata even with custom scoring)
        if 'Analytic_SR' in file:
            udm_file = file.replace('Analytic_SR', 'udm')
        elif 'AnalyticMS_SR_8b_harmonized' in file:
            udm_file = file.replace('AnalyticMS_SR_8b_harmonized', 'udm2')
        else:
            print(f"Unsupported file naming convention: {file}. Skipping.")
            continue

        # Only require UDM file if custom scoring is disabled
        if not os.path.exists(udm_file) and not getattr(processor, 'use_custom_cloud_score', False):
            print(f"UDM file not found: {udm_file}. Skipping.")
            continue

        print(f"Found UDM file: {udm_file}" if os.path.exists(udm_file) else "Using custom cloud scoring")

        # Process analytic file
        analytic_clipped_path = processor.reproject_and_clip(file, 'analytic')
        
        # Only process UDM file if custom scoring is disabled
        udm_clipped_path = None
        if os.path.exists(udm_file) and not getattr(processor, 'use_custom_cloud_score', False):
            udm_clipped_path = processor.reproject_and_clip(udm_file, 'udm')
            if not processor.check_file_properties(analytic_clipped_path, udm_clipped_path):
                print(f"Files not aligned: {analytic_clipped_path}, {udm_clipped_path}. Skipping.")
                continue

        cleaned_path = processor.apply_udm_mask(udm_clipped_path, analytic_clipped_path)
        print(f"Processed file: {cleaned_path}")

        # Group by year
        year = file.split('_')[1]
        print(f"Grouping file {file} under year: {year}")
        processed_files_by_year.setdefault(year, []).append(cleaned_path)

    # Cleanup
    if os.path.exists(processor.tmp_dir):
        shutil.rmtree(processor.tmp_dir)
    print(f"Temporary files deleted for {folder}.")

# Print yearly file counts
print("\nFiles grouped by year:")
for year, files in processed_files_by_year.items():
    print(f"Year: {year}, Files: {len(files)}")

def create_composites(output_dir):
    """Create median composites (unchanged)"""
    composites_dir = os.path.join(output_dir, "composites")
    os.makedirs(composites_dir, exist_ok=True)

    re_files = glob.glob(os.path.join(output_dir, "RapidEye", "*_cleaned.tif"))
    ps_files = glob.glob(os.path.join(output_dir, "PlanetScope", "*_cleaned.tif"))

    if re_files:
        re_output = os.path.join(composites_dir, "REOrthoTile_median_composite.tif")
        print(f"\nCreating RapidEye composite from {len(re_files)} images...")
        if create_median_composite(re_files, re_output):
            print(f"Successfully created RapidEye composite at {re_output}")

    if ps_files:
        ps_output = os.path.join(composites_dir, "PSScene_median_composite.tif")
        print(f"\nCreating PlanetScope composite from {len(ps_files)} images...")
        if create_median_composite(ps_files, ps_output):
            print(f"Successfully created PlanetScope composite at {ps_output}")

if __name__ == "__main__":
    create_composites("/Users/brookeengland/Documents/Internship/Project/Planet Output")
