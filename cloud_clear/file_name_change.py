import os
import re

# directory containing files
directory = '/Users/brookeengland/Documents/Internship/Project/Planet/Planet Files/2015-planet_reorthotile_analytic_sr/REOrthoTile' # change to actual file path

# Patterns
metadata_pattern = re.compile(r"(\d{8})_(\d{6})_(\d+)_([^_]+)_metadata\.json")
data_pattern = re.compile(r"(\d{8})_(\d{6})_(\d+)_([^_]+)\.json$")

# Loop through each file
for filename in os.listdir(directory):
    # Error check
    if not filename.endswith('.json'):
        continue # skip non-JSON files

    # Check if the file name patches the pattern
    match = metadata_pattern.match(filename)
    is_metadata = True

    if not match:
        match = data_pattern.match(filename)
        is_metadata = False

    if match:
        date_str, time_str, grid_cell, satellite = match.groups()

        #Reformat date from YYYYMMDD to YYYY-MM-DD
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        # Add suffix
        suffix = "_metadata.json" if is_metadata else ".json"

        # Create new file name
        new_filename = f"{grid_cell}_{formatted_date}_{time_str}_{satellite}{suffix}"

        # Full paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        try:
            # Rename
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")
        except Exception as e:
            print(f"Error renaming: {filename}: {e}")
    else:
        print(f"Skipped (no match): {filename}")

