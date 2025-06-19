import collections
import glob
import geopandas as gpd
import shutil
import os
import re
from cloud_clear.planetscope import PlanetScope
from cloud_clear.rapideye import RapidEye
from compositing import create_median_composite

def main():
    # Define directories
    base_dir = "/Users/brookeengland/Documents/Internship/Project/Planet/Planet Files"
    output_base_dir = "/Users/brookeengland/Documents/Internship/Project/Planet Output"
    aoi = gpd.read_file('/Users/brookeengland/Documents/Internship/Project/Planet/aotea/aotea.shp').to_crs('EPSG:2193')

    # Create output subfolders
    os.makedirs(os.path.join(output_base_dir, "RapidEye"), exist_ok=True)
    #os.makedirs(os.path.join(output_base_dir, "PlanetScope"), exist_ok=True) # Planet Scope

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
        if match:
            year = match.group(1)
        else:
            print(f"Could not extract year from folder: {folder}. Skipping.")
            continue

        # ------ Planet Scope ------
       ## if 'psscene' in folder.lower():
            # PlanetScope processing (unchanged)
          ##  file_list = glob.glob(f"{folder}/PSScene/*.tif")
           ## print(f"Found {len(file_list)} .tif files in {folder}/PSScene")
           ## processor = PlanetScope(
           ##     tmp_dir=f"{folder}/tmp",
            ##    output_dir=os.path.join(output_base_dir, "PlanetScope"),
            ##    aoi=aoi
           ## )

        # --------- Rapid Eye -----------
        if 'reorthotile' not in folder.lower():
            print(f"Skipping unsupported folder: {folder}")
            continue

        # Get image files
        file_list = glob.glob(f"{folder}/REOrthoTile/*.tif")
        print(f"Found {len(file_list)} .tif files in {folder}/REOrthoTile")

        # Create per-year output directory
        yearly_output_dir = os.path.join(output_base_dir, "RapidEye", year)
        os.makedirs(yearly_output_dir, exist_ok=True)

        # Process files
        for file in file_list:
            print(f"\nProcessing file: {file}")

            # Initialize processor
            processor = RapidEye(
                tmp_dir=f"{folder}/tmp",
                output_dir = yearly_output_dir,
                aoi=aoi
            )

            # Get UDM file
            if 'Analytic_SR' in file:
                udm_file = file.replace('Analytic_SR', 'udm')
            elif 'AnalyticMS_SR_8b_harmonized' in file:
                udm_file = file.replace('AnalyticMS_SR_8b_harmonized', 'udm2')
            else:
                print(f"Unsupported file naming convention: {file}. Skipping.")
                continue

            # UDM file not found
            if not os.path.exists(udm_file):
                print(f"UDM file not found: {udm_file}. Skipping.")
                continue

            # Process analytic file
            analytic_clipped = processor.reproject_and_clip(file, 'analytic')

            # Process UDM file
            udm_clipped = processor.reproject_and_clip(udm_file, 'udm')

            if not processor.check_file_properties(analytic_clipped, udm_clipped):
                print(f"Files not aligned: {analytic_clipped}, {udm_clipped}. Skipping.")
                continue

            # Individual masks
            udm = processor.udm_mask(udm_clipped, analytic_clipped)
            udm_buffer = processor.udm_buffer_mask(udm_clipped, analytic_clipped)
            cs = processor.cs_mask(analytic_clipped)
            cs_buffer = processor.cs_buffer_mask(analytic_clipped)

            # Store paths
            cleaned_paths = {
                "udm": udm,
                "udmbuffer": udm_buffer,
                "cs": cs,
                "csbuffer": cs_buffer,

                # Combinations
                "udm_cs": processor.combined_mask(analytic_clipped, udm_clipped, combo_type="udm_cs"),
                "udm_csbuffer": processor.combined_mask(analytic_clipped, udm_clipped, combo_type="udm_csbuffer"),
                "udmbuffer_cs": processor.combined_mask(analytic_clipped, udm_clipped, combo_type="udmbuffer_cs"),
                "udmbuffer_csbuffer": processor.combined_mask(analytic_clipped, udm_clipped, combo_type="udmbuffer_csbuffer")
            }

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
def create_composites(output_dir):
    composites_dir = os.path.join(output_dir, "composites")
    os.makedirs(composites_dir, exist_ok=True)

    # Find all cleaned files
    all_files = glob.glob(os.path.join(output_dir, "RapidEye", "*", "*_cleaned.tif"))

    # Organize by mask type and year
    composite_groups = collections.defaultdict(list)
    for tif in all_files:
        filename = os.path.basename(tif)

        # Extract mask type from the filename
        match = re.search(r'_analytic_clipped_([a-z_]+)_cleaned\.tif$', filename)
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
        composite_path = os.path.join(composites_dir, f"REOrthoTile_median_{mask_type}_composite_{year}.tif")
        print(f"Creating composite for {mask_type} mask in {year} from {len(files)} images....")

        if create_median_composite(files, composite_path):
            print(f"Composite created: {composite_path}")
            created_count += 1

    print(f"Total composites created: {created_count}")


if __name__ == "__main__":
    main()
    create_composites("/Users/brookeengland/Documents/Internship/Project/Planet Output")

