import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import random
import glob
from sklearn.utils import resample

# ---------------- File Paths ------------------

# Single file
#mosaic_path = '/Users/brookeengland/Documents/Internship/Project/Training Data/Aotea_S2/S2_mosaic_2018.tif'
#slip_path = '/Users/brookeengland/Documents/Internship/Project/Training Data/Rasterized/S2_2018_rasterized_slips.tif'
#filtered_slip_path = '/Users/brookeengland/Documents/Internship/Project/Training Data/Rasterized/S2_2018_filtered_slip.tif'
#prediction_output_path = '/Users/brookeengland/Documents/Internship/Project/Random Forest/Predictions/S2_2018_prediction.tif'
#model_output_path = '/Users/brookeengland/Documents/Internship/Project/Random Forest/Output/S2_2018_rf_model.pkl'

# Batch predictons
model_output_path = '/Users/brookeengland/Documents/Internship/Project/Random Forest/Output/L89_multiyear_rf_model.pkl'
prediction_output_dir = '/Users/brookeengland/Documents/Internship/Project/Random Forest/Predictions/Batch'

# ----------------- Configuration ----------------
invalid_class = 9
nodata_value = -9999
n_estimators = 100
test_size = 0.2

# -------------- Step 1: Add Control Pixels ------------------
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

# --------------- Step 1: Clean Slip Raster ------------- (Not needed with add control pixels function)
"""def clean_slip_raster(input_path, output_path):
    with rasterio.open(input_path) as src:
        data = src.read(1)
        meta = src.meta.copy()

    # Treat both 9 and 255 as invalid classes
    data[(data == invalid_class) | (data== 255)] = nodata_value
    meta.update(nodata=nodata_value)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data, 1)

    print(f"Slip raster saved to: {output_path}")

    # Check unique class values
    unique_classes = np.unique(data)
    print("Filtered slip classes:", unique_classes)"""


# ------------- Step 2: Extract Training Samples ------------------
def extract_training_data(mosaic_path, slip_path):
    with rasterio.open(mosaic_path) as mosaic, rasterio.open(slip_path) as slips:
        mosaic_data = mosaic.read()
        slip_data = slips.read(1)

        invalid_values = [nodata_value, 9, 255]
        mask = ~np.isin(slip_data, invalid_values)

        X = mosaic_data[:, mask].T
        y = slip_data[mask]

        print(f"Extracted {X.shape[0]} training samples with {X.shape[1]} features")
        return X, y


# Multi-year training data
def extract_training_data_multi(mosaic_dir, slip_dir, years, n_samples=5000):
    X_all = []
    y_all = []

    for year in years:
        print(f"\n Processing year: {year}")
        mosaic_path = os.path.join(mosaic_dir, f'L89_mosaic_{year}.tif')
        slip_path = os.path.join(slip_dir, f'L89_{year}_rasterized_slips.tif')
        filtered_path = os.path.join(slip_dir, f'L89_{year}_filtered_slips.tif')

        # Add control pixels
        add_control_pixels(mosaic_path, slip_path, filtered_path, n_samples=n_samples)

        # Extract features + labels
        with rasterio.open(mosaic_path) as mosaic, rasterio.open(filtered_path) as slips:
            mosaic_data = mosaic.read()
            slip_data = slips.read(1)

            invalid_values = [nodata_value, 9, 255]
            mask = ~np.isin(slip_data, invalid_values)

            X = mosaic_data[:, mask].T
            y = slip_data[mask]

            print(f"Extracted {X.shape[0]} training samples")
            X_all.append(X)
            y_all.append(y)

    # Combine across years
    X_combined = np.vstack(X_all)
    y_combined = np.concatenate(y_all)

    print(f"\nTotal training samples from {len(years)} years: {X_combined.shape[0]}")
    return X_combined, y_combined


# --------------------- Step 3: Train Model -------------------------
def train_rf_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return clf

# ------------------ Step 4: Predict Over Entire Raster -----------------
def predict_full_raster(mosaic_path, model, output_path):
    with rasterio.open(mosaic_path) as src:
        data = src.read()
        meta = src.meta.copy()

        height, width = data.shape[1], data.shape[2]
        X_all = data.reshape(data.shape[0], -1).T

        preds = model.predict(X_all)
        pred_image = preds.reshape(height, width)

        meta.update(count=1, dtype='int32')

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(pred_image.astype('int32'), 1)

    print(f"Saved predicted raster to: {output_path}")


def batch_predict_all():
    mosaic_dir = '/Users/brookeengland/Documents/Internship/Project/Training Data/Aotea_L89/'
    output_dir = '/Users/brookeengland/Documents/Internship/Project/Random Forest/Predictions/Batch'

    print(f"\nRunning batch predictions from: {mosaic_dir}")
    model = joblib.load(model_output_path)

    mosaic_files = sorted(glob.glob(os.path.join(mosaic_dir, 'L89_mosaic_*.tif')))
    print(f"Found {len(mosaic_files)} mosaics")

    for mosaic_path in mosaic_files:
        base_name = os.path.basename(mosaic_path).replace('.tif', '')
        class_output_path = os.path.join(output_dir, f'{base_name}_prediction.tif')

        print(f"\nProcessing: {base_name}")
        with rasterio.open(mosaic_path) as src:
            data = src.read()
            meta = src.meta.copy()
            height, width = data.shape[1], data.shape[2]

            X = data.reshape(data.shape[0], -1).T
            preds = model.predict(X)
            pred_image = preds.reshape(height, width)

            meta.update(count=1, dtype='int32')
            os.makedirs(output_dir, exist_ok=True)
            with rasterio.open(class_output_path, 'w', **meta) as dst:
                dst.write(pred_image.astype('int32'), 1)
            print(f"Saved class prediction to: {class_output_path}")


# ---------------------- Main Workflow ---------------------
def main():
    #print("Cleaning slip raster...")
    #clean_slip_raster(slip_path, filtered_slip_path)

    # Multi-file processing
    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]    # years for training data
    mosaic_dir = '/Users/brookeengland/Documents/Internship/Project/Training Data/Aotea_L89/'
    slip_dir = '/Users/brookeengland/Documents/Internship/Project/Training Data/Rasterized/'

    print("Extracting multi-year training data...")
    X, y = extract_training_data_multi(mosaic_dir, slip_dir, years, n_samples=20000)

    print("Training Random Forest classifier...")
    model = train_rf_classifier(X, y)

    print("Saving Trained Model...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Saved model to {model_output_path}")


    # Single file processing....
    """print("Adding control pixels to slip raster...")
    add_control_pixels(
        mosaic_path=mosaic_path,
        slip_path=slip_path,
        output_path=filtered_slip_path,
        n_samples=5000
    )


    print("Extracting training data...")
    X, y = extract_training_data(mosaic_path, filtered_slip_path)

    print("Training Random Forest classifier...")
    model = train_rf_classifier(X, y)

    print("Predicting over full mosaic...")
    os.makedirs(os.path.dirname(prediction_output_path), exist_ok=True)
    predict_full_raster(mosaic_path, model, prediction_output_path)

    print("Saving trained model...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)

    print(f"Saved model to {model_output_path}")"""


if __name__ == "__main__":
    main()
    batch_predict_all()
