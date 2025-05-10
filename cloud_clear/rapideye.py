from .base import CloudClearBase
import rasterio
import numpy as np
from scipy.ndimage import binary_dilation
import os

class RapidEye(CloudClearBase):
    def __init__(self, tmp_dir, output_dir, aoi):
        super().__init__(tmp_dir, output_dir, aoi)
        self.use_custom_cloud_score = True
        # Dark pixel threshold - adjust this number between 0.0-1.0 if needed
        # Higher values = more aggressive dark pixel masking
        self.dark_threshold = 0.05  # Default threshold (5% reflectance)

    def _mask_dark_pixels(self, scaled_data):
        """
        Mask all dark pixels below the threshold.
        Args:
            scaled_data: Image data already scaled to reflectance (0-1)
        Returns:
            Binary mask where 1=valid pixels, 0=dark pixels to be masked
        """
        # Calculate mean brightness across all bands
        mean_brightness = np.mean(scaled_data, axis=0)
        
        # Create mask (True = bright enough, False = too dark)
        dark_mask = mean_brightness < self.dark_threshold
        
        # Buffer the mask slightly to avoid edge artifacts
        buffered_mask = binary_dilation(dark_mask, iterations=2)
        
        return ~buffered_mask  # Invert so 1=valid, 0=dark

    def _calculate_cloud_score(self, scaled_data):
        """
        Custom cloud scoring using reflectance values (0-1).
        Assumes RapidEye bands are ordered as [B, G, R, NIR, RedEdge].
        """
        blue = scaled_data[0, :, :]
        green = scaled_data[1, :, :]
        red = scaled_data[2, :, :]
        nir = scaled_data[3, :, :]
        rededge = scaled_data[4, :, :]

        score = np.ones_like(blue)  # Start with score=1 (clear)
        
        # Brightness in blue band
        blue_score = (np.clip(blue, 0.05, 0.3) - 0.05) / 0.25
        score = np.minimum(score, blue_score)
        
        # Brightness in visible bands (R+G+B)
        visible_score = (np.clip(red + green + blue, 0.1, 0.8) - 0.1) / 0.7
        score = np.minimum(score, visible_score)
        
        # Brightness in NIR/RedEdge
        nir_score = (np.clip(nir + rededge, 0.15, 0.8) - 0.15) / 0.65
        score = np.minimum(score, nir_score)
        
        return score

    def apply_udm_mask(self, udm_file, analytic_file):
        """Apply either UDM mask or custom cloud score mask."""
        with rasterio.open(analytic_file) as src_analytic:
            # Read data and scale to reflectance ONCE
            raw_data = src_analytic.read()
            scaled_data = raw_data.astype('float32') / 10000.0
            meta = src_analytic.meta.copy()

        if self.use_custom_cloud_score:
            # Calculate cloud score using scaled reflectance values
            cloud_score = self._calculate_cloud_score(scaled_data)
            cloud_mask = cloud_score > 0.05  # Cloud threshold
            buffered_cloud_mask = binary_dilation(cloud_mask, iterations=3)
            
            # Get dark pixel mask using same scaled data
            dark_pixel_mask = self._mask_dark_pixels(scaled_data)
            
            # Combine masks (1=clear, 0=cloudy or dark)
            final_mask = np.logical_and(~buffered_cloud_mask, dark_pixel_mask)
            mask = final_mask.astype('float32')
        else:
            # Original UDM-based masking
            with rasterio.open(udm_file) as src_udm:
                udm = src_udm.read(1)
                unusable_mask = udm == 2
                buffered_mask = binary_dilation(unusable_mask, iterations=3)
                mask = np.where(buffered_mask, 0, 1)

        # Apply mask to scaled data
        masked_data = scaled_data * mask[np.newaxis, :, :]
        
        # Save output
        output_file = os.path.join(self.output_dir, os.path.basename(analytic_file).replace('.tif', '_cleaned.tif'))
        meta.update({'dtype': 'float32'})
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(masked_data.astype('float32'))

        print(f"Processed file saved to: {output_file}")
        return output_file
