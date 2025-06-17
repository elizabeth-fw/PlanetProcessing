from .base import CloudClearBase
import rasterio
import numpy as np
from scipy.ndimage import binary_dilation
import os

class PlanetScope(CloudClearBase):
    def _scale_to_reflectance(self, analytic_file):
        """
        Reads and scales image data from an analytic file to reflectance (0-1)
        Args:
            analytic_file: Path to the input image file

        Returns:
            scaled_data (np.ndarray): Relfectance-scaled image data
            meta (dict): Metadata from the input file
        """
        with rasterio.open(analytic_file) as src_analytic:
            # Read data
            raw_data = src_analytic.read()
            meta = src_analytic.meta.copy()
            # Scale to reflectance
            scaled_data = raw_data.astype('float32') / 10000.0
            # Return scaled data and meta data
            return scaled_data, meta

    def apply_udm_mask(self, udm_file, analytic_file):
        """
        Applies the UDM mask to the analytic file and saves the cleaned image in the output directory.
        A buffer of 3 pixels is applied to the UDM mask to make it slightly larger.
        """
        # Read and scale the analytic data
        analytic_data, out_meta = self._scale_to_reflectance(analytic_file)

        # Open UDM and read mask bands
        with rasterio.open(udm_file) as src_udm:
            udm = src_udm.read() # Shape: (8, height, width)

        # Create a combined mask where any band (except Band 1: clear) indicates unusable data
        unusable_mask = np.zeros_like(udm[0], dtype=bool)  # Initialize mask as False (usable)
        for band_idx in range(1, 7):  # Check bands 2-7 (snow, shadow, haze, cloud, etc.)
            unusable_mask = np.logical_or(unusable_mask, udm[band_idx] == 1)  # Mark pixels as unusable if any band indicates it

        # Apply a buffer of 3 pixels to the unusable mask
        buffer_size = 1
        buffered_mask = binary_dilation(unusable_mask, iterations=buffer_size)

        # Invert the mask: usable pixels = 1, unusable pixels = 0
        mask = np.where(buffered_mask, 0, 1)

        # Apply the mask to the scaled analytic data
        masked_data = analytic_data * mask[np.newaxis, :, :]

        # Update metadata for the output file
        out_meta.update({'dtype': 'float32'})  # Update data type to float32 for scaled data

        # Save the masked and scaled image
        output_file = os.path.join(self.output_dir, os.path.basename(analytic_file).replace('.tif', '_cleaned.tif'))
        with rasterio.open(output_file, 'w', **out_meta) as dst:
            dst.write(masked_data.astype('float32'))  # Ensure data is saved as float32

        print(f"Masked and scaled image saved at: {output_file}")
        return output_file