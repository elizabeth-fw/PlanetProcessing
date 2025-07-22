# Imports
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import random
import glob
from rasterio.merge import merge
import geopandas as gpd
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from datetime import datetime

# ----------------- Txt Log -------------------
def log_to_csv(log_path, row, headers=None):
    file_exists = os.path.exists(log_path)
    with open(log_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not file_exists and headers:
            writer.writeheader()
        writer.writerow(row)


# ---------------- File Paths ------------------
model_output_path = '/Users/brookeengland/Documents/Internship/Project/Random Forest/Output/S2_multiyear_rf_model.pkl'
prediction_output_dir = '/Users/brookeengland/Documents/Internship/Project/Random Forest/Predictions/Batch'

# ----------------- Configuration ----------------
invalid_class = 9
nodata_value = -9999
n_estimators = 100
test_size = 0.2

# --------------------------- Step 1: Add Control Pixels ---------------------------
# Adds stable background pixels (class 99) to the slip raster to improve classification balance
def add_control_pixels(mosaic_path, slip_path, output_path, n_samples=5000):
    with rasterio.open(mosaic_path) as mosaic, rasterio.open(slip_path) as slips:
        mosaic_data = mosaic.read()
        slip_data = slips.read(1)
        meta = slips.meta.copy()

    # Define invalid and landslide areas
    mask_invalid = (slip_data == 255) | (slip_data == 9) | (slip_data == -9999)
    mask_landslide = (slip_data >= 0) & (slip_data <=3)

    # Identify background using mosaic: valid if all bands are non-zero
    mosaic_valid = np.all(mosaic_data != 0, axis=0)

    # Background = mosaic has invalid data AND slip raster is invalid or nodata
    mask_background = (~mask_landslide) & mask_invalid & mosaic_valid

    # Debug: show how many candidates we have
    num_candidates = np.sum(mask_background)
    print(f"Background candidate pixels: {num_candidates}")

    if num_candidates < n_samples:
        print(f"Warning: only found {num_candidates} background pixels, reducing sample size.")
        n_samples = num_candidates

    if n_samples == 0:
        print("No control pixels could be sampled. Skipping control pixel addition.")
    else:
        # Sample Control Pixels
        rows, cols = np.where(mask_background)
        sample_idxs = random.sample(range(len(rows)), n_samples)
        sample_rows = rows[sample_idxs]
        sample_cols = cols[sample_idxs]

        # Add control pixels to slip_data as class 99
        slip_data[(sample_rows, sample_cols)] = 99 # new class: background stable terrain

    # Save updates slip raster
    meta.update(nodata=-9999)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(slip_data, 1)

    print(f"Saved slip raster with {n_samples} control pixels to: {output_path}")


# ------------------------- Step 2: Extract Training Samples -------------------------
# For each year:
#   - Adds DEM, slope, and aspect features
#   - Filters out invalid pixels
#   - Remaps slip classes
#   - Collects labeled feature vectors for training
def extract_training_data(mosaic_dir, slip_dir, years, n_samples=5000):
    X_all = []
    y_all = []

    # Load DEM
    dem_path = "/Users/brookeengland/Documents/Internship/Project/Training Data/DEM mosaic/mosaic_dem_resampled.tif"
    with rasterio.open(dem_path) as dem_src:
        dem_data = dem_src.read(1)

    for year in years:
        print(f"\nProcessing year: {year}")
        mosaic_path = os.path.join(mosaic_dir, f'S2_mosaic_{year}.tif')
        slip_path = os.path.join(slip_dir, f'S2_{year}_rasterized_slips.tif')
        filtered_path = os.path.join(slip_dir, f'S2_{year}_filtered_slips.tif')

        # Add control pixels
        add_control_pixels(mosaic_path, slip_path, filtered_path, n_samples=n_samples)

        # Extract features + labels
        with rasterio.open(mosaic_path) as mosaic, rasterio.open(filtered_path) as slips:
            mosaic_data = mosaic.read()
            slip_data = slips.read(1)

            # Check shape match
            if dem_data.shape != mosaic_data.shape[1:]:
                raise ValueError(f"DEM shape {dem_data.shape} doesn't match mosaic {mosaic_data.shape[1:]}")

            # Add DEM as extra band
            dem_band = np.expand_dims(dem_data, axis=0)
            combined_data = np.concatenate((mosaic_data, dem_band), axis=0)

            # Compute slope and aspect
            pixel_size = 10
            slope, aspect = calculate_slope_aspect(dem_data, pixel_size)

            # Stack slope and aspect as additional bands
            slope_band = np.expand_dims(slope, axis=0)
            aspect_band = np.expand_dims(aspect, axis=0)
            combined_data = np.concatenate((combined_data, slope_band, aspect_band), axis=0)

            # Create mask of valid pixels
            invalid_values = [nodata_value, 255]
            mask = ~np.isin(slip_data, invalid_values)

            X = combined_data[:, mask].T
            y = slip_data[mask]

            # Apply Class Remapping
            y_remapped = np.copy(y)
            y_remapped[np.isin(y, [0, 1])] = 1  # Merge 0 + 1 -> high confidence landslide
            y_remapped[y == 2] = 2  # Medium confidence
            y_remapped[y == 3] = 3  # Low confidence
            y_remapped[y == 8] = 4  # Forest
            y_remapped[y == 9] = 4  # Non-Landslide
            y_remapped[y == 99] = 0 # Background

            print(f"Extracted {X.shape[0]} training samples")
            X_all.append(X)
            y_all.append(y_remapped)

    # Combine across years
    X_combined = np.vstack(X_all)
    y_combined = np.concatenate(y_all)

    # Print unique classes
    unique_classes = np.unique(y_combined)
    print(f"Final training classes: {unique_classes}")

    # Total training samples
    print(f"\nTotal training samples from {len(years)} years: {X_combined.shape[0]}")
    return X_combined, y_combined

# --------------------------------- Step 3: Train Model ----------------------------------
# - Splits data into training and test sets
# - Trains a Random Forest classifier with class balancing
# - Evaluates performance using classification report
def train_rf_classifier(X, y, txt_report_path=None, model_name=None, dataset_years=None):
    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=42)

    # Build Random Forest Model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        random_state=42
    )
    # Fit model
    clf.fit(X_train, y_train)

    # Make Predictions
    y_pred = clf.predict(X_test)

    # Classification report
    report=classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    # Save text report
    if txt_report_path:
        with open(txt_report_path, 'a') as f:
            f.write('\n' + "=" * 80 + "\n")
            f.write(f"Model trained: {datetime.now().isoformat()}\n")
            f.write(f"Model file: {model_name or 'N/A'}\n")
            f.write(f"Dataset years: {dataset_years if dataset_years else 'N/A'}\n")
            f.write(f"Input samples: {len(X)} | Features: {X.shape[1]}\n\n")
            f.write(f"Features: DEM + Slope + Aspect\n\n")
            f.write("Class Label Definitions:\n")
            f.write("    0 - Background \n")
            f.write("    1 - High confidence landslide (merged from original 0 + 1)\n")
            f.write("    2 - Medium confidence landslide\n")
            f.write("    3 - Low confidence landslide\n")
            f.write("    4 - Forest / Non-landslide (original 8 + 9)\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\n")
        print(f"Appended classification report to: {txt_report_path}")

    return clf

# ----------------------------- Step 4: Predict Over Entire Raster -----------------------------
# For each yearly mosaic:
#   - Adds NDVI, DEM, slope, and aspect
#   - Applies trained model across full raster
#   - Saves prediction as GeoTIFF
def batch_predict_all():
    mosaic_dir = '/Users/brookeengland/Documents/Internship/Project/Training Data/Aotea_S2/'
    output_dir = '/Users/brookeengland/Documents/Internship/Project/Random Forest/Predictions/Batch'

    print(f"\nRunning batch predictions from: {mosaic_dir}")
    model = joblib.load(model_output_path)

    # Get Mosaic files
    mosaic_files = sorted(glob.glob(os.path.join(mosaic_dir, 'S2_mosaic_*.tif')))
    print(f"Found {len(mosaic_files)} mosaics")

    for mosaic_path in mosaic_files:
        base_name = os.path.basename(mosaic_path)

        # Skip buffer file in folder
        if "_buffer" in base_name:
            continue

        base_name = base_name.replace('.tif', '')
        class_output_path = os.path.join(output_dir, f'{base_name}_prediction.tif')

        # Process Files
        print(f"\nProcessing: {base_name}")
        with rasterio.open(mosaic_path) as src:
            data = src.read()
            meta = src.meta.copy()
            height, width = data.shape[1], data.shape[2]

        # Add DEM
        dem_path = "/Users/brookeengland/Documents/Internship/Project/Training Data/DEM mosaic/mosaic_dem_resampled.tif"
        with rasterio.open(dem_path) as dem_src:
            dem_data = dem_src.read(1)

            if dem_data.shape != data.shape[1:]:
                raise ValueError(f"DEM shape {dem_data.shape} doesn't match mosaic {data.shape[1:]}")

            # Add DEM band
            dem_band = np.expand_dims(dem_data, axis=0)
            data = np.concatenate((data, dem_band), axis=0)

            # Add slope and aspect
            pixel_size = 10  # meters for Sentinel-2
            slope, aspect = calculate_slope_aspect(dem_data, pixel_size)
            slope_band = np.expand_dims(slope, axis=0)
            aspect_band = np.expand_dims(aspect, axis=0)
            data = np.concatenate((data, slope_band, aspect_band), axis=0)

            X = data.reshape(data.shape[0], -1).T
            preds = model.predict(X)
            pred_image = preds.reshape(height, width)

            meta.update({"count": 1, "dtype": 'int32', "nodata": -9999})
            os.makedirs(output_dir, exist_ok=True)
            with rasterio.open(class_output_path, 'w', **meta) as dst:
                dst.write(pred_image.astype('int32'), 1)
            print(f"Saved class prediction to: {class_output_path}")



# ------------------- DEM Processing -----------------------
# mosaic_dems() - Combines multiple DEM tiles
def mosaic_dems(input_folder, output_path):
    # Find all .tif files
    search_path = os.path.join(input_folder, "*.tif")
    dem_files = glob.glob(search_path)

    # Open and collect datasets
    src_files_to_mosaic = [rasterio.open(fp) for fp in dem_files]

    # Merge
    mosaic, out_transform = merge(src_files_to_mosaic)

    # Use metadata from the first file
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "count": mosaic.shape[0]
    })

    # Write output
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Mosaic saved to: {output_path}")


# clip_raster() - clips DEM mosaic to AOI shapefile
def clip_raster(raster_path, shapefile_path, output_path):
    with rasterio.open(raster_path) as src:
        aoi = gpd.read_file(shapefile_path).to_crs(src.crs)
        geoms = aoi.geometry.values
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)


# resample_raster_to_match() - Matches DEM resolution to S2 mosaics
def resample_raster_to_match(src_path, target_path, output_path):
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_dtype = src.dtypes[0]

    with rasterio.open(target_path) as target:
        target_meta = target.meta.copy()
        target_shape = (target_meta['height'], target_meta['width'])

        # Create empty array to hold resampled data
        resampled_data = np.empty(target_shape, dtype=src_dtype)

        reproject(
            source=src_data,
            destination=resampled_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_meta['transform'],
            dst_crs=target_meta['crs'],
            dst_shape=target_shape,
            resampling=Resampling.bilinear
        )

        # Update metadata
        target_meta.update({
            "height": resampled_data.shape[0],
            "width": resampled_data.shape[1],
            "dtype": src_dtype,
            "count": 1
        })

        with rasterio.open(output_path, 'w', **target_meta) as dst:
            dst.write(resampled_data, 1)

    print(f"Resampled DEM saved to: {output_path}")


# calculate_slope_aspect() - derives slope and aspect from DEM
def calculate_slope_aspect(dem_array, pixel_size):
    # Gradient in x (cols) and y (rows)
    dz_dy, dz_dx = np.gradient(dem_array, pixel_size)

    # Slope in radians then degrees
    slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope)

    # Aspect in radians then degrees
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, 90.0 - aspect_deg, aspect_deg)

    return slope_deg, aspect_deg


# ---------------------- Main Workflow ---------------------
# Executes DEM prep, data extraction, model training, and predictions
def main():
    txt_report_path = "/Users/brookeengland/Documents/Internship/Project/Random Forest/Output/classification_report.txt"
    model_name = os.path.basename(model_output_path)

    years = [2018, 2019, 2020, 2021, 2022, 2023]    # years for training data
    mosaic_dir = '/Users/brookeengland/Documents/Internship/Project/Training Data/Aotea_S2/'
    slip_dir = '/Users/brookeengland/Documents/Internship/Project/Training Data/Rasterized/'

    # DEM mosaic
    mosaic_dems("/Users/brookeengland/Documents/Internship/Project/Training Data/lds-new-zealand-lidar-1m-dem-GTiff",
                "/Users/brookeengland/Documents/Internship/Project/Training Data/DEM mosaic/mosaic_dem.tif")

    clip_raster("/Users/brookeengland/Documents/Internship/Project/Training Data/DEM mosaic/mosaic_dem.tif",
                "/Users/brookeengland/Documents/Internship/Project/Planet/aotea/aotea.shp",
                "/Users/brookeengland/Documents/Internship/Project/Training Data/DEM mosaic/mosaic_dem_clipped.tif")

    # Resample DEM to match one of the Sentinel-2 mosaics
    reference_mosaic = "/Users/brookeengland/Documents/Internship/Project/Training Data/Aotea_S2/S2_mosaic_2018.tif"
    resample_raster_to_match(
        src_path="/Users/brookeengland/Documents/Internship/Project/Training Data/DEM mosaic/mosaic_dem_clipped.tif",
        target_path=reference_mosaic,
        output_path="/Users/brookeengland/Documents/Internship/Project/Training Data/DEM mosaic/mosaic_dem_resampled.tif"
    )

    print("Extracting multi-year training data...")
    X, y = extract_training_data(mosaic_dir, slip_dir, years, n_samples=20000)

    print("Training Random Forest classifier...")
    model = train_rf_classifier(X, y,
                                txt_report_path=txt_report_path,
                                model_name=model_name,
                                dataset_years=years)

    print("Saving Trained Model...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Saved model to {model_output_path}")


if __name__ == "__main__":
    main()
    batch_predict_all()
