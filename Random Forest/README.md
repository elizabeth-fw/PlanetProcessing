# Random Forest Landslide Classification

This repository contains a **Random Forest based landslide classification** for multi-year analysis

---
## Overview
The script performs the following steps:
1. **Dem Preparation**
2. **Slip Raster Preparation**
3. **Training Data Extraction**
4. **Model Training**
5. **Batch Prediction**

---

## Input Files
- Annual Sentinel-2 mosaics: **S2_mosaic_YYYY.tif**
- Geopackage containing mannually mapped landslides per year: **S2_slips.gpkg**
- Folder of DEM tiles: **lds-new-zealand-lidar-1m-dem-GTiff**
- AOI shapefile for clipping DEM: **aotea.shp**

## Class Labels
| Class | Description                                                |
|-------|------------------------------------------------------------|
| 0 | Background (area surrounding land)                         |
| 1 | High confidence landslide (merged from original IDs 0 + 1) |
| 2 | Medium confidence landslide                                |
| 3 | Low confidence landslide                                   |
| 4 | Not a landslide (bare, forest, urban)                      |
| 5 | Cloud                                                      |
| 6 | Shadow                                                     |

---

## Outputs
- Trained Random Forest model: **S2_multiyear_rf_model.pkl**
- Accuracy and class metrics: **classification_report.txt**
- Saved model accuracy score: **model_accuracy.json**
- Classified prediction raster for each year: **S2_mosaic_YYYY_prediction.tif**
- NDVI raster for each year: **S2_mosaic_YYYY_ndvi.tif**

---

## Usage
1. **Update File Paths**
2. **Run the script**
3. **Check predictions**
    - Found in **/Predictions/Batch/**
    - Open in **QGIS** to visualize results
4. After prediction, this model connects directly to the **Post-Processing** workflow


