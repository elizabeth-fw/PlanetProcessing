'''
Despeckling Class:
This class performs post-processing on Random Forest landslide prediction rasters

Functions:
    1. process_prediction_raster()
        - Removes small landslide clusters (speckles) below a minimum pixel threshold
        - Reassigns small clusters to the majority neighboring class or to forest if isolated

    2. calculate_slope_aspect()
        - Calculates slope and aspect from a DEM raster for each pixel

    3. generate_object_feature_stack()
        - Generates a multi-band raster feature stack for object based analysis
            1. Landslide class
            2. Year
            3. Accuracy
            4. Mean slope per object
            5. Mean aspect per object
            6. Mean elevation per object
            7. Mean NDVI per object
            8. Area per object
            9. Max distance per object
           10. Object ID
           11. Touching obscuring class flag (1 or 0)

    Workflow:
        1. Despeckle random forest prediction raster
        2. Compute slope and aspect from DEM
        3. Generate object-level feature statistics and export multi-band raster
'''

import rasterio
import numpy as np
from scipy.ndimage import label, find_objects, generate_binary_structure
from collections import Counter
import json

class Despeckling:
    # Static priority map (higher = more certain)
    class_priority = {
        1: 6,  # High confidence landslide
        2: 5,  # Medium confidence landslide
        3: 4,  # Low confidence landslide
        4: 2,  # Not a landslide (bare, forest, urban)
        0: 1,  # Background
        5: 0,  # Clouds
        6: 0   # Shadows
    }

    @staticmethod
    def process_prediction_raster(input_path, output_path, forest_class=4, min_pixels=3):
        """
            Removes small landslide clusters (speckles) below a minimum pixel threshold.
            Reassigns small clusters to the majority neighboring class or to forest if isolated.

            Args:
                input_path (string): Path to the input raster file
                output_path (string): Path to the output raster file
                forest_class (int): Class ID representing forest pixels
                min_pixels (int): Minimum pixel threshold for despeckling
        """
        with rasterio.open(input_path) as src:
            pred = src.read(1)
            meta = src.meta.copy()

        # Create mask of landslide pixels (classes 1–3)
        landslide_mask = np.isin(pred, [1, 2, 3])

        # Label connected landslide components (8-connectivity)
        structure = generate_binary_structure(2, 2)
        labeled_array, num_features = label(landslide_mask, structure=structure)

        processed = pred.copy()

        # Logging counters
        total_clusters = 0
        tie_breaks = 0
        reassigned_counts = Counter()

        slices = find_objects(labeled_array)
        for i, slice_yx in enumerate(slices, start=1):
            if slice_yx is None:
                continue
            component = (labeled_array[slice_yx] == i)
            pixel_indices = np.where(component)

            # Get absolute positions
            rows = pixel_indices[0] + slice_yx[0].start
            cols = pixel_indices[1] + slice_yx[1].start
            class_values = pred[rows, cols]

            # Increment total clusters
            total_clusters += 1

            # Handle small speckles
            if len(class_values) < min_pixels:
                # Find neighbors around the cluster
                neighbor_classes = []
                for r, c in zip(rows, cols):
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # skip itself
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < pred.shape[0] and 0 <= nc < pred.shape[1]:
                                # Only consider neighbors not in the cluster
                                if labeled_array[nr, nc] != i:
                                    neighbor_classes.append(pred[nr, nc])

                if neighbor_classes:
                    # Determine majority class of neighbors
                    counts = Counter(neighbor_classes)
                    max_count = max(counts.values())
                    tied_classes = [cls for cls, cnt in counts.items() if cnt == max_count]

                    if len(tied_classes) == 1:
                        final_class = tied_classes[0]
                    else:
                        # Tie-breaker: use mean of tied classes
                        tie_breaks += 1
                        mean_val = np.mean(tied_classes)
                        # Pick class closest to the mean
                        final_class = min(tied_classes, key=lambda cls: (abs(cls - mean_val), -cls))
                else:
                    # If no neighbors (isolated pixel), fallback to forest
                    final_class = forest_class

                processed[rows, cols] = final_class
                reassigned_counts[final_class] += 1

        # Save cleaned raster
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(processed, 1)

        # Print log summary
        print(f"\n--- Post-Processing Summary ---")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Total clusters processed:        {total_clusters}")
        print(f"Clusters with tie broken:        {tie_breaks}")
        print(f"Class reassignment counts:")
        for cls_id, count in reassigned_counts.items():
            print(f"  Class {cls_id}: {count} clusters reassigned")


        print(f"Cleaned prediction saved to: {output_path}")

    # calculate_slope_aspect() - derives slope and aspect from DEM
    @staticmethod
    def calculate_slope_aspect(dem_array, pixel_size):
        """
            Calculates slope and aspect from a DEM raster for each pixel.

            Args:
                dem_array (array): Array containing DEM raster data
                pixel_size (float): Pixel size in meters

            Returns:
                slope_deg: Slope
                aspect_deg: Aspect
        """
        # Gradient in x (cols) and y (rows)
        dz_dy, dz_dx = np.gradient(dem_array, pixel_size)

        # Slope in radians then degrees
        slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
        slope_deg = np.degrees(slope)

        # Aspect in radians then degrees
        aspect_rad = np.arctan2(-dz_dy, dz_dx)
        aspect_deg = np.degrees(aspect_rad)

        # Wrap aspect to compass convention (0–360° clockwise from north)
        aspect_deg = (90.0 - aspect_deg) % 360.0

        # Set aspect to NaN where slope is 0 (flat)
        aspect_deg = np.where(slope_deg == 0, np.nan, aspect_deg)

        return slope_deg, aspect_deg


    def generate_object_feature_stack(cleaned_raster_path, dem_path, ndvi_path, output_path, year_value=2018, accuracy_value=0.86, landslide_classes=(1, 2, 3), obscuring_classes=(0,5,6)):
        """
            Creates a multi-band raster feature stack for object-based landslide analysis.

            Args:
                cleaned_raster_path (string): Path to the cleaned raster file
                dem_path (string): Path to the DEM raster file
                ndvi_path (string): Path to the NDVI raster file
                output_path (string): Path to the output raster file
                year_value (int): Year value
                accuracy_value (float): Accuracy value
                landslide_classes (tuple): Class IDs for Landslide classes
                obscuring_classes (tuple): Class IDs for Obscuring classes (background, cloud, shadow)

            Output Bands:
                1. Landslide class (0–6)
                2. Year (constant)
                3. Accuracy (constant)
                4. Slope (mean per object)
                5. Aspect (mean per object)
                6. Elevation (mean per object)
                7. NDVI (mean per object)
                8. Area
                9. Max distance
                10. Object ID (unique per connected landslide cluster)
                11. Touching obscuring class (0 = No, 1 = Yes)
            """
        # Load rasters
        with rasterio.open(cleaned_raster_path) as src_class, rasterio.open(dem_path) as src_dem, rasterio.open(ndvi_path) as src_ndvi:
            class_data = src_class.read(1)
            dem_data = src_dem.read(1)
            ndvi_data = src_ndvi.read(1)
            transform = src_dem.transform
            meta = src_class.meta.copy()

        pixel_size = src_dem.res[0]  # assume square pixels
        height, width = class_data.shape

        slope_data, aspect_data = Despeckling.calculate_slope_aspect(dem_data, pixel_size)

        # Initialize output bands
        class_band = class_data.astype(np.float32)
        year_band = np.full((height, width), year_value, dtype=np.uint16)
        acc_band = np.full((height, width), accuracy_value, dtype=np.float32)
        slope_band = np.full((height, width), np.nan, dtype=np.float32)
        aspect_band = np.full((height, width), np.nan, dtype=np.float32)
        elev_band = np.full((height, width), np.nan, dtype=np.float32)
        ndvi_band = np.full((height, width), np.nan, dtype=np.float32)
        area_band = np.full((height, width), np.nan, dtype=np.float32)
        dist_band = np.full((height, width), np.nan, dtype=np.float32)
        id_band = np.full((height, width), 0, dtype=np.uint16)
        touching_band = np.full((height, width), 0, dtype=np.uint8)

        # Label landslide clusters
        structure = generate_binary_structure(2, 2)
        mask = np.isin(class_data, landslide_classes)
        labeled_array, num_objects = label(mask, structure=structure)

        print(f"Found {num_objects} objects.")

        slices = find_objects(labeled_array)
        for obj_id, slice_yx in enumerate(slices, start=1):
            if slice_yx is None:
                continue

            region_mask = (labeled_array[slice_yx] == obj_id)

            rows = region_mask.nonzero()[0] + slice_yx[0].start
            cols = region_mask.nonzero()[1] + slice_yx[1].start
            pixels = (rows, cols)

            slope_vals = slope_data[pixels]
            aspect_vals = aspect_data[pixels]
            elev_vals = dem_data[pixels]
            ndvi_vals = ndvi_data[pixels]

            # Calculate area (pixel count * pixel area)
            pixel_area = pixel_size ** 2
            area_band[pixels] = len(rows) * pixel_area

            # Max distance across object
            coords = np.column_stack((rows, cols)) * pixel_size  # convert to meters
            if len(coords) > 1:
                dists = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))
                max_dist = np.max(dists)
            else:
                max_dist = 0
            dist_band[pixels] = max_dist

            # Determine if object touches obscuring class
            touches_obscuring = False
            for r, c in zip(rows, cols):
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if class_data[nr, nc] in obscuring_classes and labeled_array[nr, nc] != obj_id:
                                touches_obscuring = True
                                break
                if touches_obscuring:
                    break

            # Fill feature bands for this object
            slope_band[pixels] = np.nanmean(slope_vals)
            aspect_band[pixels] = np.nanmean(aspect_vals)
            elev_band[pixels] = np.nanmean(elev_vals)
            ndvi_band[pixels] = np.nanmean(ndvi_vals)
            id_band[pixels] = obj_id
            touching_band[pixels] = 1 if touches_obscuring else 0

        # Write output
        meta.update({
            "dtype": "float32",  # All bands written as float32
            "count": 11,  # 11 bands
            "nodata": None
        })

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(class_band, 1)                            # Band 1: Landslide Class
            dst.write(year_band.astype(np.float32), 2)          # Band 2: Year
            dst.write(acc_band.astype(np.float32), 3)           # Band 3: Accuracy
            dst.write(slope_band.astype(np.float32), 4)         # Band 4: Slope
            dst.write(aspect_band.astype(np.float32), 5)        # Band 5: Aspect
            dst.write(elev_band.astype(np.float32), 6)          # Band 6: Elevation
            dst.write(ndvi_band.astype(np.float32), 7)          # Band 7: NDVI
            dst.write(area_band.astype(np.float32), 8)          # Band 8: Area
            dst.write(dist_band.astype(np.float32), 9)          # Band 9: Max distance
            dst.write(id_band.astype(np.float32), 10)           # Band 10: Object ID band
            dst.write(touching_band.astype(np.float32), 11)     # Band 11: Touching obscuring class (1 or 0)



if __name__ == "__main__":
    input_raster = "/Users/brookeengland/Documents/Internship/Project/Random Forest/Predictions/Batch/S2_mosaic_2018_prediction.tif"
    output_raster = "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Despeckling/Cleaned Rasters/S2_2018_rf_prediction_cleaned.tif"
    final_output = "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Despeckling/Object Features/S2_2018_object_features.tif"
    dem_path = "/Users/brookeengland/Documents/Internship/Project/Training Data/DEM mosaic/mosaic_dem_resampled.tif"
    accuracy_json = "/Users/brookeengland/Documents/Internship/Project/Random Forest/Output/model_accuracy.json"
    ndvi_path = "/Users/brookeengland/Documents/Internship/Project/Random Forest/Output/NDVI output/S2_mosaic_2018_ndvi.tif"

    # Step 1: Despeckle prediction raster
    Despeckling.process_prediction_raster(
        input_path=input_raster,
        output_path=output_raster,
        forest_class=4,
        min_pixels=3
    )

    # Step 2: Load model accuracy from JSON
    with open(accuracy_json, "r") as f:
        accuracy_value = json.load(f)["accuracy"]

    # Step 3: Generate feature stack
    Despeckling.generate_object_feature_stack(
        cleaned_raster_path=output_raster,
        dem_path=dem_path,
        ndvi_path=ndvi_path,
        output_path=final_output,
        year_value=2018, # Update for each year
        accuracy_value=accuracy_value
    )


