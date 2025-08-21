"""
Vectorize.py
This class converts post-processed Random Forest landslide classification rasters into vector GeoPackages.

Steps:
    1. Reads in the multi-band raster containing:
        - Band 1  : Landslide class
        - Band 2  : Year
        - Band 3  : Accuracy
        - Band 4  : Slope
        - Band 5  : Aspect
        - Band 6  : Elevation
        - Band 7  : NDVI
        - Band 8  : Area
        - Band 9  : Max Distance
        - Band 10 : Object ID
        - Band 11 : Touching class flag

    2. Vectorizes polygons using the objectID band, merging connected pixels into features
    3. Ensures that each landslide is a single feature
    4. Extracts representative attribute values for each polygon from the raster
    5. Outputs a GeoPackage (.gpkg) containing vectorized landslide features with all attributes

Output Columns:
    - class
    - Year
    - Accuracy
    - Slope
    - Aspect
    - Elevation
    - NDVI
    - Area
    - Max Distance
    - object_id
    - Touching Class
"""

import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
import numpy as np

class Vectorize:
    @staticmethod
    def vectorize_raster(raster_path, output_vector_path):
        """
            Converts post-processed Random Forest landslide classification rasters into vector GeoPackages.

            Args:
                raster_path: Path to raster file
                output_vector_path: Path to vectorized raster file
        """
        print(f"\nVectorizing raster: {raster_path}")

        # Open raster and read class band
        with rasterio.open(raster_path) as src:
            obj_ids = src.read(10).astype(np.int32)  # Object ID
            class_data = src.read(1) # Landslide class
            band2 = src.read(2)  # Year
            band3 = src.read(3)  # Accuracy
            band4 = src.read(4)  # Slope
            band5 = src.read(5)  # Aspect
            band6 = src.read(6)  # Elevation
            band7 = src.read(7)  # NDVI
            band8 = src.read(8)  # Area
            band9 = src.read(9)  # Max Distance
            band11 = src.read(11) # Touching-flag band
            transform = src.transform
            crs = src.crs

        mask = obj_ids > 0
        shapes_gen = shapes(obj_ids, mask=mask, transform=transform)

        polygons, obj_values = [], []
        for geom, val in shapes_gen:
            if val is None:
                continue
            polygons.append(shape(geom))
            obj_values.append(int(val))

        # Create initial GeoDataFrame
        gdf = gpd.GeoDataFrame({
            "object_id": obj_values,
            "geometry": polygons
        }, crs=crs)

        # Merge all parts with same object ID
        gdf = gdf.dissolve(by="object_id", as_index=False)

        # Extract values from one pixel inside each polygon
        sample_points = [geom.representative_point().coords[0] for geom in gdf.geometry]

        # Get band values
        with rasterio.open(raster_path) as src:
            class_vals, year_vals, acc_vals, slope_vals, aspect_vals, elev_vals, ndvi_vals, area_vals, dist_vals, touch_vals = ([] for _ in range(10))
            for x, y in sample_points:
                row, col = src.index(x, y)
                class_vals.append(class_data[row, col])
                year_vals.append(band2[row, col])
                acc_vals.append(band3[row, col])
                slope_vals.append(band4[row, col])
                aspect_vals.append(band5[row, col])
                elev_vals.append(band6[row, col])
                ndvi_vals.append(band7[row, col])
                area_vals.append(band8[row, col])
                dist_vals.append(band9[row, col])
                touch_vals.append(band11[row, col])

        # Add values to each band
        gdf["class"] = class_vals
        gdf["Year"] = year_vals
        gdf["Accuracy"] = acc_vals
        gdf["Slope"] = slope_vals
        gdf["Aspect"] = aspect_vals
        gdf["Elevation"] = elev_vals
        gdf["NDVI"] = ndvi_vals
        gdf["Area"] = area_vals
        gdf["Max Distance"] = dist_vals
        gdf["Touching Class"] = touch_vals

        # Save to file
        gdf.to_file(output_vector_path, driver="GPKG")
        print(f"Vector file saved to: {output_vector_path}")

if __name__ == "__main__":
    vector_output = "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Vectors/S2_2018_vectorized.gpkg"
    raster_input = "/Users/brookeengland/Documents/Internship/Project/Post-Processing/Despeckling/Object Features/S2_2018_object_features.tif"

    Vectorize.vectorize_raster(
        raster_path=raster_input,
        output_vector_path=vector_output
    )

