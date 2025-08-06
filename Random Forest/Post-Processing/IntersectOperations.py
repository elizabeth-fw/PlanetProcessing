"""
IntersectOperations.py
This class processes multi-year landslide vector files to:
    1. Assign unique identifier to landslide polygons across all years (SlipID)
    2. Track changes in:
        - Area and percentage change
        - Maximum distance
        - NDVI
        - Percentage intersection with previous year polygons
    3. Indentify whether a landslide is new, existing, or needs review:
        - 'new': no sufficient overlap with previous years
        - 'existing': >= overlap threshold
        - 'review': < overlap threshold and above 0, or within buffer distance of prior polygons
    4. Output:
        - Yearly labeled GeoPackage files with new attributes
        - A combined master GeoPackage with all years and change metrics

    Usage:
        - Provide a sorted list of yearly vector file paths (GeoPackages)
        - Define an output folder for labeled GeoPackages
        - Set overlap_threshold (%) to define when polygons are considered the same
        - Set buffer_dist (meters) for review flagging
"""

# Imports
import geopandas as gpd
import numpy as np
import pandas as pd
import os


class IntersectOperations:
    @staticmethod
    def compute_changes(vector_paths, output_folder, overlap_threshold=0.2, buffer_dist=10):
        """
            Processes multiple years of landslide polygons and assigns:
                - SlipID (unique ID for tracking multi-year landslides)
                - Tracks area, NDVI, and distance changes
                - Flags polygons for review if < threshold overlap or within buffer distance
                - Adds Sensor and Resolution columns

            Args:
                vector_paths (list): list of paths to vector files
                output_folder (str): path to output folder
                overlap_threshold (float): overlap threshold (%)
                buffer_dist (int): buffer distance (meters)
        """
        fid_counter = 1
        master_gdf = None

        for year_idx, vector_path in enumerate(sorted(vector_paths)):
            gdf = gpd.read_file(vector_path)
            gdf["Year"] = int(os.path.basename(vector_path).split("_")[1])  # Extract year from filename

            # Initialize columns
            gdf["SlipID"] = np.nan          # Unique slip ID across all years
            gdf["IntersectYear"] = np.nan   # Year of intersection
            gdf["PctIntersect"] = 0.0       # Percentage intersection
            gdf["ReviewFlag"] = False       # Review flag
            gdf["ChangeArea"] = np.nan      # Change in area
            gdf["PctChangeArea"] = np.nan   # % change in area
            gdf["ChangeDist"] = np.nan      # Change in Max Distance
            gdf["ChangeNDVI"] = np.nan      # Change in NDVI
            gdf["Status"] = "new"           # Default = new landslide

            # Extract sensor from filename
            sensor_name = os.path.basename(vector_path).split("_")[0]

            # Map sensor to resolution
            sensor_resolution_map = {
                "S2": 10,
                "PlanetScope": 3,
                "RapidEye": 5,
                "L457": 30,
                "L89": 30
            }

            resolution = sensor_resolution_map.get(sensor_name, 0) # Default to 0 if unknown

            # Assign sensor and resolution
            gdf["Sensor"] = sensor_name     # Sensor type
            gdf["Resolution"] = resolution  # Pixel resolution in meters


            # Remove original vector IDs
            for col in ["object_id", "UID", "FeatureID"]:
                if col in gdf.columns:
                    gdf = gdf.drop(columns=[col])

            # First year → all new landslides
            if year_idx == 0:
                gdf["SlipID"] = [fid_counter + i for i in range(len(gdf))]
                fid_counter += len(gdf)
                master_gdf = gdf.copy()

            else:
                # Compare current year to master_gdf
                for idx, row_curr in gdf.iterrows():
                    geom_curr = row_curr.geometry
                    fid = None
                    max_overlap = 0.0
                    best_match = None

                    # Compare with all prior features
                    for _, row_prev in master_gdf.iterrows():
                        geom_prev = row_prev.geometry
                        if geom_curr.intersects(geom_prev):
                            intersection = geom_curr.intersection(geom_prev)
                            overlap_pct = intersection.area / geom_curr.area

                            if overlap_pct > max_overlap:
                                max_overlap = overlap_pct
                                best_match = row_prev

                    # Determine feature linkage
                    if best_match is not None and max_overlap >= overlap_threshold:
                        # Existing landslide
                        fid = best_match["SlipID"]
                        gdf.at[idx, "SlipID"] = fid
                        gdf.at[idx, "Status"] = "existing"

                        # Most recent previous observation for this SlipID
                        prev_records = master_gdf[
                            (master_gdf["SlipID"] == fid) & (master_gdf["Year"] < row_curr["Year"])]
                        if not prev_records.empty:
                            recent_prev = prev_records.sort_values("Year").iloc[-1]
                            recent_year = recent_prev["Year"]
                        else:
                            recent_prev = best_match
                            recent_year = best_match["Year"]

                        # Set intersect year as most recent observed
                        gdf.at[idx, "IntersectYear"] = recent_year
                        gdf.at[idx, "PctIntersect"] = max_overlap * 100

                        # Compute changes
                        curr_area = row_curr.get("Area", np.nan)
                        prev_area = recent_prev.get("Area", np.nan)
                        gdf.at[idx, "ChangeArea"] = curr_area - prev_area
                        gdf.at[idx, "ChangeDist"] = row_curr.get("Max Distance", np.nan) - recent_prev.get("Max Distance", np.nan)
                        gdf.at[idx, "ChangeNDVI"] = row_curr.get("NDVI", np.nan) - recent_prev.get("NDVI", np.nan)

                        if prev_area and prev_area > 0:
                            gdf.at[idx, "PctChangeArea"] = ((curr_area - prev_area) / prev_area) * 100
                        else:
                            gdf.at[idx, "PctChangeArea"] = np.nan

                    else:
                        # If overlap > 0 but < threshold, mark for review
                        if 0 < max_overlap < overlap_threshold:
                            gdf.at[idx, "ReviewFlag"] = True
                            gdf.at[idx, "Status"] = "review"
                            gdf.at[idx, "IntersectYear"] = best_match["Year"] if best_match is not None else np.nan
                            gdf.at[idx, "PctIntersect"] = max_overlap * 100

                        # If no overlap but within buffer, also mark for review
                        else:
                            for _, row_prev in master_gdf.iterrows():
                                if geom_curr.distance(row_prev.geometry) <= buffer_dist:
                                    gdf.at[idx, "ReviewFlag"] = True
                                    gdf.at[idx, "IntersectYear"] = row_prev["Year"]
                                    break

                        # New SlipID
                        fid = fid_counter
                        fid_counter += 1
                        gdf.at[idx, "SlipID"] = fid

                # Merge current year into master_gdf
                master_gdf = gpd.GeoDataFrame(pd.concat([master_gdf, gdf], ignore_index=True), crs=gdf.crs)

            # Save yearly labeled output
            output_path = os.path.join(output_folder, f"{gdf['Year'].iloc[0]}_labeled.gpkg")
            gdf.to_file(output_path, driver="GPKG", index=False)
            print(f"Processed and saved: {output_path}")

        # Clean master_gdf before saving
        for col in ["object_id", "UID", "FeatureID"]:
            if col in master_gdf.columns:
                master_gdf = master_gdf.drop(columns=[col])

        # Fill year gaps for all SlipIDs
        master_gdf = fill_data_gaps(master_gdf)

        # Sort by SlipID and Year
        master_gdf = master_gdf.sort_values(["SlipID", "Year"]).reset_index(drop=True)

        # Save master file with all years combined
        master_output = os.path.join(output_folder, "all_years_labeled.gpkg")
        master_gdf.to_file(master_output, driver="GPKG", index=False)
        print(f"\nMaster file saved: {master_output}")


def fill_data_gaps(master_gdf):
    """
        Ensures all SlipIDs have continuous year sequences.
        If a slip is missing a year in between its min and max year, add a placeholder row with NaNs for all other rows

        Args:
            master_gdf (.gpkg): master file with all combined years labeled
        Returns:
            master_gdf (.gpkg): Updated master file with missing years filled
    """
    filled_rows = []

    for slip_id, group in master_gdf.groupby("SlipID"):
        years_present = sorted(group["Year"].unique())
        if len(years_present) <= 1:
            continue

        # Determine all years between first and last observation
        full_year_range = list(range(years_present[0], years_present[-1] + 1))

        # Find missing years
        missing_years = set(full_year_range) - set(years_present)

        if missing_years:
            last_known = group.sort_values("Year").iloc[-1]
            for missing_year in sorted(missing_years):
                # Add a placeholder for missing years
                placeholder = {
                    "geometry": None,
                    "Year": missing_year,
                    "SlipID": slip_id,
                    "IntersectYear": np.nan,
                    "PctIntersect": np.nan,
                    "ReviewFlag": False,
                    "ChangeArea": np.nan,
                    "PctChangeArea": np.nan,
                    "ChangeDist": np.nan,
                    "ChangeNDVI": np.nan,
                    "Status": "missing",
                    "Sensor": last_known.get("Sensor", "Sentinel-2"),
                    "Resolution": last_known.get("Resolution", 10)
                }
                filled_rows.append(placeholder)

    # Append all missing rows
    if filled_rows:
        master_gdf = pd.concat(
            [master_gdf, gpd.GeoDataFrame(filled_rows, crs=master_gdf.crs)],
            ignore_index=True
        )

    return master_gdf

if __name__ == "__main__":
    # List of yearly vector files (sorted by year)
    vector_files = [
        "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Vectors/S2_2018_vectorized.gpkg",
        "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Vectors/S2_2019_vectorized.gpkg",
        "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Vectors/S2_2020_vectorized.gpkg",
        "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Vectors/S2_2021_vectorized.gpkg",
        "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Vectors/S2_2022_vectorized.gpkg",
        "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Vectors/S2_2023_vectorized.gpkg",
        "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Vectors/S2_2024_vectorized.gpkg"
    ]

    # Output folder for labeled files
    output_folder = "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Intersect Operations/Labeled"

    # Run multi-year intersection + change detection
    IntersectOperations.compute_changes(
        vector_paths=vector_files,
        output_folder=output_folder,
        overlap_threshold=0.2,  # 20% overlap to be considered the same feature
        buffer_dist=10  # 10m distance for review flag
    )