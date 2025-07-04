import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

def create_median_composite(input_files, output_path, aoi, resolution=(5,5), nodata_val=-9999):
    """
    Creates a median composite of cloud-masked images aligned to a grid defined by the AOI.

    Args:
        input_files (list): List of cleaned image paths
        output_path (str): Output GeoTIFF path
        aoi (GeoDataFrame): AOI polygon in EPSG:2193
        resolution (tuple): (xres, yres) in output (e.g. (5,5))
        nodata_val (numeric): Nodata fill value

    """
    if len(input_files) < 2:
        print("Need at least 2 images to create composite")
        return False

    try:
        # --- Define output grid from AOI ---
        minx, miny, maxx, maxy = aoi.total_bounds
        xres, yres = resolution
        width = int((maxx - minx) / xres)
        height = int((maxy - miny) / yres)
        transform = from_origin(minx, maxy, xres, yres)
        dst_crs = 'EPSG:2193'

        # Get metadata from first image
        with rasterio.open(input_files[0]) as ref:
            bands = ref.count
            dtype = 'float32'  # Safe default, you're already using reflectance-scaled float32

        # Allocate aligned image stack
        all_data = []

        for file in input_files:
            with rasterio.open(file) as src:
                # Prepare empty aligned array
                aligned = np.full((bands, height, width), np.nan, dtype=np.float32)

                reproject(
                    source=rasterio.band(src, list(range(1, bands + 1))),
                    destination=aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

                # Convert source nodata to nan if not already
                src_nodata = src.nodata if src.nodata is not None else nodata_val
                aligned[aligned == src_nodata] = np.nan

                all_data.append(aligned)

            # Stack and compute median across input images
        stack = np.stack(all_data, axis=0)  # (num_images, bands, height, width)
        median = np.nanmedian(stack, axis=0)  # (bands, height, width)

        # Convert nan back to nodata for saving
        median[np.isnan(median)] = nodata_val

        # Write final composite
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': bands,
            'dtype': 'float32',
            'crs': dst_crs,
            'transform': transform,
            'nodata': nodata_val
        }

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(median.astype('float32'))

        #print(f"Composite written to {output_path}")
        return True

    except Exception as e:
        print(f"Error creating composite: {e}")

        return False






