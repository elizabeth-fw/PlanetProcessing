from .base import CloudClearBase
import rasterio
import numpy as np
from scipy.ndimage import binary_dilation
import os

class RapidEye(CloudClearBase):
    def apply_udm_mask(self, udm_file, analytic_file):
        """
        Applies the UDM mask to the analytic file and saves the cleaned image in the output directory.
        A buffer of 3 pixels is applied to the UDM mask to make it slightly larger.
        """
        with rasterio.open(analytic_file) as src_analytic, rasterio.open(udm_file) as src_udm:
            # Apply scaling factor (divide by 10,000 to convert 0-1500 to 0-0.15)
            analytic_data = src_analytic.read() / 10000.0

            # Read UDM mask (no scaling needed for UDM files)
            udm = src_udm.read(1)
            unusable_mask = udm == 2  # Create mask where UDM == 2

            # Apply a buffer of 3 pixels to the unusable mask
            buffer_size = 3
            buffered_mask = binary_dilation(unusable_mask, iterations=buffer_size)

            # Invert the mask: usable pixels = 1, unusable pixels = 0
            mask = np.where(buffered_mask, 0, 1)

            # Apply the mask to the scaled analytic data
            masked_data = analytic_data * mask[np.newaxis, :, :]

            # Update metadata for the output file
            out_meta = src_analytic.meta.copy()
            out_meta.update({'dtype': 'float32'})  # Update data type to float32 for scaled data

            # Save the masked and scaled image
            output_file = os.path.join(self.output_dir, os.path.basename(analytic_file).replace('.tif', '_cleaned.tif'))
            with rasterio.open(output_file, 'w', **out_meta) as dst:
                dst.write(masked_data.astype('float32'))  # Ensure data is saved as float32

        print(f"Masked and scaled image saved at: {output_file}")
        return output_file