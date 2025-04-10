from .base import CloudClearBase
import rasterio
import numpy as np
from scipy.ndimage import binary_dilation
import os

class RapidEye(CloudClearBase):
    def __init__(self, tmp_dir, output_dir, aoi):
        super().__init__(tmp_dir, output_dir, aoi)
        self.use_custom_cloud_score = True  # Toggle between UDM and custom cloud scoring
        # Inheritance: uses CloudClearBase for shared functionality (reprojection, clipping).
        # Toggle: use_custom_cloud_score switches between UDM and custom masking.


    def _calculate_cloud_score(self, img_data):
        """
        Custom cloud scoring (adapted from GEE JavaScript).
        Assumes RapidEye bands are ordered as [B, G, R, NIR, RedEdge].
        """
        blue = img_data[0, :, :].astype('float32') / 10000.0  # Scaled to reflectance
        green = img_data[1, :, :] / 10000.0
        red = img_data[2, :, :] / 10000.0
        nir = img_data[3, :, :] / 10000.0
        rededge = img_data[4, :, :] / 10000.0  # Optional: Use RedEdge for better cloud detection

        # Cloud score logic (simplified for RapidEye)
        score = np.ones_like(blue)  # Start with score=1 (clear)
        
        # Brightness in blue band (adjust thresholds for RapidEye)
        blue_score = (np.clip(blue, 0.05, 0.3) - 0.05) / 0.25
        score = np.minimum(score, blue_score)
        
        # Brightness in visible bands (R+G+B)
        visible_score = (np.clip(red + green + blue, 0.1, 0.8) - 0.1) / 0.7
        score = np.minimum(score, visible_score)
        
        # Brightness in NIR/RedEdge (approximates SWIR)
        nir_score = (np.clip(nir + rededge, 0.15, 0.8) - 0.15) / 0.65
        score = np.minimum(score, nir_score)
        
        return score

    def apply_udm_mask(self, udm_file, analytic_file):
        """Apply either UDM mask or custom cloud score mask."""
        with rasterio.open(analytic_file) as src_analytic:
            analytic_data = src_analytic.read()
            meta = src_analytic.meta.copy()

        if self.use_custom_cloud_score:
            # Use custom cloud scoring (ignore UDM file)
            cloud_score = self._calculate_cloud_score(analytic_data)
            cloud_mask = cloud_score > 0.05  # Threshold (adjust as needed)
            buffered_mask = binary_dilation(cloud_mask, iterations=3)  # Buffer 3 pixels
            mask = np.where(buffered_mask, 0, 1)  # 1=clear, 0=cloudy
        else:
            # Original UDM-based masking (from your code)
            with rasterio.open(udm_file) as src_udm:
                udm = src_udm.read(1)
                unusable_mask = udm == 2
                buffered_mask = binary_dilation(unusable_mask, iterations=3)
                mask = np.where(buffered_mask, 0, 1)

        # Apply mask and save
        masked_data = (analytic_data / 10000.0) * mask[np.newaxis, :, :]  # Scale to reflectance
        output_file = os.path.join(self.output_dir, os.path.basename(analytic_file).replace('.tif', '_cleaned.tif'))
        
        meta.update({'dtype': 'float32'})
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(masked_data.astype('float32'))

        print(f"Processed file saved to: {output_file}")
        return output_file
