# Landslide Post-Processing Workflow

This repository contains Python scripts for **post-processing** multi-year landslide classification results from Random Forest models.

---
## Workflow Overview

The post-processing workflow performs the following tasks:

### 1. **Despeckling and Feature Cleaning**
- Script: **Despeckling.py**
- Removes very small or isolated predicted landslides (1-2 pixels) by:
    1. Reassigning small features to the **majority class** of their neighbors
    2. Marking isolated features as **non-landslide (Forest)**.
- Output: Cleaned rasters for vectorization

---

### 2. **Vectorization of Raster Predictions**
- Script: **Vectorize.py**
- Converts classified Random Forest Prediction rasters (.tif) into vector polygons (.gpkg).
- Attributes added:
  - **Class ID** (landslide class)
  - **Year**
  - **Accuracy Score**
  - **Slope, Aspect, Elevation** (from DEM)
  - **NDVI** (read from raster)
- Filters only **landslide classes** (1 = High, 2 = Medium, 3 = Low).

---

### 3. **Multi-Year Intersection and Change Detection**
- Script: **IntersectOperations.py**
- Tracks landslides across multiple years using vector polygons.
- Main tasks:
    1. **Assigns unique SlipID** to each landslide across all years.
    2. Detects **existing**, **new**, and **review** features based on overlap.
    3. Computes change detection metrics:
       - **ChangeArea** (area difference in meters)
       - **PctChangeArea** (percent change from last observation)
       - **ChangeDist** (change in maximum distance)
       - **ChangeNDVI** (vegetation recovery or loss)
    4. Tracks **IntersectYear**: the most recent year the landslide was observed.
    5. Flags polygons for **review** if:
       - Overlap is below the existing threshold but greater than 0
       - Within a specified buffer distance of prior polygons
    6. Adds **Sensor** and **Resolution** automatically from filenames.

- Outputs:
  - **Yearly labeled GeoPackages** (YYYY_labeled.gpkg)
  - **Master file** with all years (all_years_labeled.gpkg)

---

### 4. **Data Gap Filling**
- Function: **fill_data_gaps()** in IntersectOperations.py
- Ensures **continuous year sequences** for all SlipIDs:
  - If a landslide is missing in a year between first and last observation, a placeholder row is added with:
    - **fid**
    - **Year**
    - **SlipID**
    - All change metrics = **NaN**
    - Status = **missing**

---
    
## How to Run
After running Random Forest Model and generating predictions:
- Run each of the scripts in the following order
  - **Note**: Despeckling.py and Vectorize.py must be run for each year individually
1. **Despeckling.py**
   - Input:
     - Random Forest Prediction (S2_mosaic_YYYY_prediction.tif)
     - DEM path (mosaic_dem_resampled.tif)
     - Accuracy file (model_accuracy.json)
     - NDVI path (S2_mosaic_YYYY_ndvi.tif)
   - Output:
     - Cleaned Prediction raster (S2_YYYY_rf_prediction_cleaned.tif)
     - Object Features raster (S2_YYYY_object_features.tif)
2. **Vectorize.py**
   - Input:
     - Object Features raster (S2_YYYY_object_features.tif)
   - Output:
     - Vectorized landslides (S2_YYYY_vectorized.gpkg)

3. **IntersectOperations.py**
   - Input:
     - List of yearly vector files sorted by year (S2_YYYY_vectorized.gpkg, ... , ...)
     - Specify desired overlap_threshold and buffer_dist
   - Output:
     - Labeled GeoPackages for each year (YYYY_labeled.gpkg)
     - Master file with all years (all_years_labeled.gpkg)

