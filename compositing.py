import numpy as np
import bottleneck as bn
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from tqdm import tqdm

def create_median_composite(input_files, output_path, aoi, resolution=(5,5), nodata_val=-9999, block_size=512):
    """
    Creates a memory-efficient median composite of cloud-masked images aligned to AOI grid using windowed processing.

    Args:
        input_files (list): List of cleaned image paths
        output_path (str): Output GeoTIFF path
        aoi (GeoDataFrame): AOI polygon in EPSG:2193
        resolution (tuple): (xres, yres)
        nodata_val (numeric): Nodata value
        block_size (int): Size of read/write blocks (tiles)
    """
    if len(input_files) < 2:
        print("Need at least 2 images to create composite.")
        return False

    try:
        # Define output grid based on AOI
        minx, miny, maxx, maxy = aoi.total_bounds
        xres, yres = resolution
        width = int((maxx - minx) / xres)
        height = int((maxy - miny) / yres)
        transform = from_origin(minx, maxy, xres, yres)
        dst_crs = 'EPSG:2193'

        with rasterio.open(input_files[0]) as ref:
            bands = ref.count
            dtype = 'float32'

        # Set output metadata
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': bands,
            'dtype': dtype,
            'crs': dst_crs,
            'transform': transform,
            'nodata': nodata_val,
            'tiled': True,
            'blockxsize': block_size,
            'blockysize': block_size,
            'compress': 'lzw'
        }

        with rasterio.open(output_path, 'w', **meta) as dst:
            for y in tqdm(range(0, height, block_size), desc="Processing Rows"):
                for x in range(0, width, block_size):
                    win_width = min(block_size, width - x)
                    win_height = min(block_size, height - y)
                    window = Window(x, y, win_width, win_height)

                    tile_stack = []

                    for file in input_files:
                        with rasterio.open(file) as src:
                            temp_array = np.full((bands, win_height, win_width), np.nan, dtype=np.float32)

                            try:
                                reproject(
                                    source=rasterio.band(src, list(range(1, bands + 1))),
                                    destination=temp_array,
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=transform * window.transform(),
                                    dst_crs=dst_crs,
                                    resampling=Resampling.nearest
                                )
                                # Replace nodata with NaN
                                src_nodata = src.nodata if src.nodata is not None else nodata_val
                                temp_array[temp_array == src_nodata] = np.nan
                                tile_stack.append(temp_array)
                            except Exception as e:
                                print(f"Warning: Failed to reproject {file} for window {window}. Skipping. Reason: {e}")

                    if tile_stack:
                        stack = np.stack(tile_stack, axis=0)  # Shape: (N, bands, h, w)
                        stack = np.transpose(stack, (1, 0, 2, 3))  # Shape: (bands, N, h, w)
                        median = bn.nanmedian(stack, axis=1)
                        median[np.isnan(median)] = nodata_val
                        dst.write(median.astype(np.float32), window=window)

        print(f"Composite successfully written to {output_path}")
        return True

    except Exception as e:
        print(f"Error creating windowed median composite: {e}")
        return False






