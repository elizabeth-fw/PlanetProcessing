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
#   - Adds NDVI, EVI, and SAVI bands from S2 mosaic
#   - Filters out invalid pixels
#   - Remaps slip classes
#   - Collects labeled feature vectors for training
def extract_training_data(mosaic_dir, slip_dir, years, n_samples=5000):
    X_all = []
    y_all = []

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

            # Define bands
            blue = mosaic_data[1, :, :]
            red = mosaic_data[3, :, :]
            nir = mosaic_data[7, :, :]

            # NDVI
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi = np.expand_dims(np.clip(ndvi, -1, 1), axis=0)

            # EVI
            G = 2.5
            C1 = 6
            C2 = 7.5
            L_evi = 1.0
            evi = G * (nir - red) / (nir + C1 * red - C2 * blue + L_evi + 1e-6)
            evi = np.expand_dims(np.clip(evi, -1, 1), axis=0)

            # SAVI
            L_savi = 0.5
            savi = ((nir - red) / (nir + red + L_savi + 1e-6)) * (1 + L_savi)
            savi = np.expand_dims(np.clip(savi, -1, 1), axis=0)

            # Stack features
            mosaic_data_with_ndvi_evi_savi = np.concatenate((mosaic_data, ndvi, evi, savi), axis=0)

            invalid_values = [nodata_value, 255]
            mask = ~np.isin(slip_data, invalid_values)

            X = mosaic_data_with_ndvi_evi_savi[:, mask].T
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
            f.write(f"Features: NDVI + EVI + SAVI\n\n")
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
#   - Adds NDVI, EVI, and SAVI bands
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

            # Define bands
            blue = data[1, :, :]
            red = data[3, :, :]
            nir = data[7, :, :]

            # NDVI
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi = np.expand_dims(np.clip(ndvi, -1, 1), axis=0)

            # EVI
            G = 2.5
            C1 = 6
            C2 = 7.5
            L_evi = 1.0
            evi = G * (nir - red) / (nir + C1 * red - C2 * blue + L_evi + 1e-6)
            evi = np.expand_dims(np.clip(evi, -1, 1), axis=0)

            # SAVI
            L_savi = 0.5
            savi = ((nir - red) / (nir + red + L_savi + 1e-6)) * (1 + L_savi)
            savi = np.expand_dims(np.clip(savi, -1, 1), axis=0)

            # Append both bands
            data = np.concatenate((data, ndvi, evi, savi), axis=0)

            X = data.reshape(data.shape[0], -1).T
            preds = model.predict(X)
            pred_image = preds.reshape(height, width)

            meta.update(count=1, dtype='int32')
            os.makedirs(output_dir, exist_ok=True)
            with rasterio.open(class_output_path, 'w', **meta) as dst:
                dst.write(pred_image.astype('int32'), 1)
            print(f"Saved class prediction to: {class_output_path}")

# ---------------------- Main Workflow ---------------------
# Executes data extraction, model training, and predictions
def main():
    txt_report_path = "/Users/brookeengland/Documents/Internship/Project/Random Forest/Output/classification_report.txt"
    model_name = os.path.basename(model_output_path)

    years = [2018, 2019, 2020, 2021, 2022, 2023]    # years for training data
    mosaic_dir = '/Users/brookeengland/Documents/Internship/Project/Training Data/Aotea_S2/'
    slip_dir = '/Users/brookeengland/Documents/Internship/Project/Training Data/Rasterized/'


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
