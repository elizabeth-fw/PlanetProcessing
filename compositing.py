import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def create_median_composite(input_files, output_path):
    """
    Creates median composite with transparent background (proper nodata handling)
    """
    if len(input_files) < 2:
        print("Need at least 2 images to create composite")
        return False

    try:
        # Get union bounds and resolution from all images
        bounds = []
        resolutions = []
        crs_list = []
        nodata_values = []
        
        for f in input_files:
            with rasterio.open(f) as src:
                bounds.append(src.bounds)
                resolutions.append(src.res)
                crs_list.append(src.crs)
                nodata_values.append(src.nodata)

        # Use most common nodata value or default to 0
        nodata = max(set(nodata_values), key=nodata_values.count) if nodata_values else 0
        
        # Calculate output bounds and resolution
        left = min(b[0] for b in bounds)
        bottom = min(b[1] for b in bounds)
        right = max(b[2] for b in bounds)
        top = max(b[3] for b in bounds)
        res = min(resolutions)  # Use finest resolution

        # Create output transform and dimensions
        transform = rasterio.transform.from_origin(left, top, res[0], res[1])
        width = int((right - left) / res[0])
        height = int((top - bottom) / res[1])

        # Initialize output with transparency (nodata)
        with rasterio.open(input_files[0]) as src:
            count = src.count
            dtype = src.meta['dtype']
        
        output = np.full((count, height, width), nodata, dtype=dtype)
        valid_pixels = np.zeros((height, width), dtype=np.uint8)

        # Process each image
        all_data = []
        for file in input_files:
            with rasterio.open(file) as src:
                data = np.full((count, height, width), nodata, dtype=dtype)
                reproject(
                    source=rasterio.band(src, range(1, count + 1)),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )
                all_data.append(data)
                valid_pixels += (data[0] != nodata).astype(np.uint8)

        # Calculate composite
        # 1. Areas with only one image
        single_mask = (valid_pixels == 1)
        for i, data in enumerate(all_data):
            img_mask = (data[0] != nodata) & single_mask
            output[:, img_mask] = data[:, img_mask]

        # 2. Overlapping areas (median)
        overlap_mask = (valid_pixels >= 2)
        if overlap_mask.any():
            stacked = np.stack([d[:, overlap_mask] for d in all_data], axis=0)
            median_values = np.median(stacked, axis=0)
            output[:, overlap_mask] = median_values

        # Write output with transparency
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': count,
            'dtype': dtype,
            'crs': crs_list[0],
            'transform': transform,
            'nodata': nodata  # Critical for transparency
        }

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(output)

        return True

    except Exception as e:
        print(f"Error creating composite: {e}")
        return False