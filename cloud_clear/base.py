import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask

class CloudClearBase:
    def _skip_if_exists(self, output_file):
        if os.path.exists(output_file):
            print(f"Skipping existing file: {output_file}")
            return output_file
        return None

    def __init__(self, tmp_dir, output_dir, aoi):
        self.tmp_dir = tmp_dir
        self.output_dir = output_dir
        self.aoi = aoi
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def check_file_properties(self, analytic_file, udm_file):
        """Check if the analytic and UDM files have the same CRS, transform, and shape."""
        with rasterio.open(analytic_file) as src_analytic, rasterio.open(udm_file) as src_udm:
            return (src_analytic.crs == src_udm.crs and 
                    src_analytic.transform == src_udm.transform and
                    src_analytic.shape == src_udm.shape)

    def reproject_and_clip(self, file, output_suffix):
        """Reproject and clip a file to the AOI, saving in tmp directory."""
        # Define expected output path
        clipped_path = os.path.join(
            self.tmp_dir,
            os.path.basename(file).replace('.tif', f'_{output_suffix}_clipped.tif')
        )

        # Skip if clipped file already exists
        if os.path.exists(clipped_path):
            print(f"Clipped file already exists: {clipped_path}")
            return clipped_path

        # Step 1: Reproject
        with rasterio.open(file) as src:
            transform, width, height = calculate_default_transform(
                src.crs, 'EPSG:2193', src.width, src.height, *src.bounds
            )
            metadata = src.meta.copy()
            metadata.update({'crs': 'EPSG:2193', 'transform': transform, 'width': width, 'height': height})

            reprojected_path = os.path.join(
                self.tmp_dir,
                os.path.basename(file).replace('.tif', f'_{output_suffix}_reprojected.tif')
            )

            # Skip reprojection if already done
            if not os.path.exists(reprojected_path):
                with rasterio.open(reprojected_path, 'w', **metadata) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs='EPSG:2193',
                            resampling=Resampling.nearest
                        )
            else:
                print(f"Reprojected file already exists: {reprojected_path}")

        # Step 2: Clip
        with rasterio.open(reprojected_path) as src:
            out_image, out_transform = mask(src, self.aoi.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform
            })

            with rasterio.open(clipped_path, 'w', **out_meta) as dst:
                dst.write(out_image)

        return clipped_path