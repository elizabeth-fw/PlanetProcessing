from .base import CloudClearBase
import rasterio
import numpy as np
from scipy.ndimage import binary_dilation
import os

class PlanetScope4Band(CloudClearBase):
    def __init__(self, tmp_dir, output_dir, aoi):
        super().__init__(tmp_dir, output_dir, aoi)
        # Dark pixel threshold - adjust this number between 0.0-1.0 if needed
        # Higher values = more aggressive dark pixel masking
        self.dark_threshold = 0.05  # Default threshold (5% reflectance)

    def _scale_to_reflectance(self, analytic_file):
        """
        Reads and scales image data from an analytic file to reflectance (0-1)

        Args:
            analytic_file: Path to the input image file

        Returns:
            scaled_data (np.array): Reflectance-scaled image data
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

    def _calculate_cloud_score(self, scaled_data):
        """
        Custom cloud scoring using reflectance values (0-1).
        Assumes PlanetScope bands are ordered as [B, G, R, NIR].
        """
        blue = scaled_data[0, :, :]
        green = scaled_data[1, :, :]
        red = scaled_data[2, :, :]
        nir = scaled_data[3, :, :]

        score = np.ones_like(blue)  # Start with score=1 (clear)

        # Brightness in blue band
        blue_score = (np.clip(blue, 0.05, 0.3) - 0.05) / 0.25
        score = np.minimum(score, blue_score)

        # Brightness in visible bands (R+G+B)
        visible_score = (np.clip(red + green + blue, 0.1, 0.8) - 0.1) / 0.7
        score = np.minimum(score, visible_score)

        # Brightness in NIR/RedEdge
        nir_score = (np.clip(nir, 0.15, 0.8) - 0.15) / 0.65    # Might need updating
        score = np.minimum(score, nir_score)

        return score


    def udm_buffer_mask(self, udm_file, analytic_file, buffer_size=3):
        """
        Applies the UDM mask to the analytic file and saves the cleaned image in the output directory.
        A buffer of 3 pixels is applied to the UDM mask to make it slightly larger.
        """
        output_file = os.path.join(self.output_dir,
                                   os.path.basename(analytic_file).replace('.tif', '_udmbuffer_cleaned.tif'))
        if self._skip_if_exists(output_file):
            return output_file
        # Read and scale the analytic data
        analytic_data, out_meta = self._scale_to_reflectance(analytic_file)

        # Open UDM and read mask bands
        with rasterio.open(udm_file) as src_udm:
            udm = src_udm.read() # Shape: (8, height, width)

        # Create a combined mask where any band (except Band 1: clear) indicates unusable data
        unusable_mask = np.zeros_like(udm[0], dtype=bool)  # Initialize mask as False (usable)
        for band_idx in range(1, 7):  # Check bands 2-7 (snow, shadow, haze, cloud, etc.)
            unusable_mask = np.logical_or(unusable_mask, udm[band_idx] == 1)  # Mark pixels as unusable if any band indicates it

        # Apply a buffer to the unusable mask
        buffered_mask = binary_dilation(unusable_mask, iterations=buffer_size)

        # Invert the mask: usable pixels = 1, unusable pixels = 0
        mask = np.where(buffered_mask, 0, 1)

        # Apply the mask to the scaled analytic data
        masked_data = analytic_data * mask[np.newaxis, :, :]

        # Update metadata for the output file
        out_meta.update({'dtype': 'float32'})  # Update data type to float32 for scaled data

        # Save the masked and scaled image
        with rasterio.open(output_file, 'w', **out_meta) as dst:
            dst.write(masked_data.astype('float32'))  # Ensure data is saved as float32

        print(f"Masked and scaled image saved at: {output_file}")
        return output_file


    def udm_mask(self, udm_file, analytic_file):
        """
        Applies the UDM mask to the analytic file and saves the cleaned image in the output directory.
        A buffer of 3 pixels is applied to the UDM mask to make it slightly larger.
        """
        output_file = os.path.join(self.output_dir, os.path.basename(analytic_file).replace('.tif', '_udm_cleaned.tif'))
        if self._skip_if_exists(output_file):
            return output_file

        # Read and scale the analytic data
        analytic_data, out_meta = self._scale_to_reflectance(analytic_file)

        # Open UDM and read mask bands
        with rasterio.open(udm_file) as src_udm:
            udm = src_udm.read() # Shape: (8, height, width)

        # Create a combined mask where any band (except Band 1: clear) indicates unusable data
        unusable_mask = np.zeros_like(udm[0], dtype=bool)  # Initialize mask as False (usable)
        for band_idx in range(1, 7):  # Check bands 2-7 (snow, shadow, haze, cloud, etc.)
            unusable_mask = np.logical_or(unusable_mask, udm[band_idx] == 1)  # Mark pixels as unusable if any band indicates it

        # Invert the mask: usable pixels = 1, unusable pixels = 0
        mask = np.where(unusable_mask, 0, 1)

        # Apply the mask to the scaled analytic data
        masked_data = analytic_data * mask[np.newaxis, :, :]

        # Update metadata for the output file
        out_meta.update({'dtype': 'float32'})  # Update data type to float32 for scaled data

        # Save the masked and scaled image
        with rasterio.open(output_file, 'w', **out_meta) as dst:
            dst.write(masked_data.astype('float32'))  # Ensure data is saved as float32

        print(f"Masked and scaled image saved at: {output_file}")
        return output_file

    def cs_mask(self, analytic_file):
        """
        Applies the custom cloud score mask to the analytic file and saves the masked image
        Args:
             analytic_file (str): Path to the input image file
        Returns:
            output_file (str): Path to the output file, masked image
        """
        output_file = os.path.join(self.output_dir,
                                   os.path.basename(analytic_file).replace('.tif', '_cs_mask_cleaned.tif'))
        if self._skip_if_exists(output_file):
            return output_file

        # Read and scale analytic data
        scaled_data, meta = self._scale_to_reflectance(analytic_file)

        # Compute cloud score
        cloud_score = self._calculate_cloud_score(scaled_data)

        # Define threshold - pixels with score > 0.05 are considered cloudy
        cloud_mask = cloud_score > 0.05

        # Combine Masks (1 = clear, 0 = cloudy or dark)
        final_mask = (~cloud_mask).astype('float32')

        # Apply final mask to scaled data
        masked_data = scaled_data * final_mask

        # Prepare output
        meta.update({'dtype': 'float32'})

        # Save the result
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(masked_data.astype('float32'))

        print(f"Custom cloud-masked image saved to: {output_file}")
        return output_file


    def low_cs_buffer(self, cloud_mask):
        """
        Create a low cloud score buffer (5 pixels)

        Args:
            cloud_mask: Binary cloud mask (1=cloud, 0=clear)
        Returns:
            np.ndarray: Buffered mask
        """
        low_buffer = binary_dilation(cloud_mask, iterations=5)
        return low_buffer


    def high_cs_buffer(self, cloud_mask):
        """
        Create a high cloud score buffer (25 pixels)

        Args:
            cloud_mask: Binary cloud mask (1=cloud, 0=clear)
        Returns:
            np.ndarray: Buffered mask
        """
        high_buffer = binary_dilation(cloud_mask, iterations=25)
        return high_buffer


    def apply_cs_buffer_mask(self, analytic_file, buffer_type="low"):
        """
        Applies the custom cloud score mask with a buffer (low or high) to the analytic file and saves the masked image
        Args:
            analytic_file (str): Path to the input image file
            buffer_type (str): Buffer type to use for masking. Defaults to 'low'.
        Returns:
            output_file (str): Path to the output file, masked image
        """
        output_file = os.path.join(self.output_dir, os.path.basename(analytic_file).replace('.tif',
                                                                                            f'_cs_{buffer_type}_buffer_mask_cleaned.tif'))
        if self._skip_if_exists(output_file):
            return output_file

        # Read and scale analytic data
        scaled_data, meta = self._scale_to_reflectance(analytic_file)

        # Compute cloud score
        cloud_score = self._calculate_cloud_score(scaled_data)

        # Threshold - pixels with score > 0.05 are considered cloudy
        cloud_mask = cloud_score > 0.05

        # Select buffer type
        if buffer_type == "low":
            buffered_cloud_mask = self.low_cs_buffer(cloud_mask)
        elif buffer_type == "high":
            buffered_cloud_mask = self.high_cs_buffer(cloud_mask)
        else:
            raise ValueError(f"Invalid buffer type: {buffer_type}")

        # Combine masks
        final_mask = (~buffered_cloud_mask).astype('float32')

        # Apply mask
        masked_data = scaled_data * final_mask

        # Prepare output
        meta.update({'dtype': 'float32'})

        # Save the masked image
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(masked_data.astype('float32'))

        print(f"Custom cloud masked image with {buffer_type} buffer saved to: {output_file}")
        return output_file

    def combined_mask(self, analytic_file, udm_file, combo_type="udm_cs", buffer_size=25):
        """
        Applies a combination of UDM and CS masking strategies

        Args:
            analytic_file (str): Path to analytic image
            udm_file (str): Path to UDM file
            combo_type (str): Combination type (e.g., "udm_cs_lowbuffer")
            buffer_size (int): Buffer size for udmbuffer

        Returns:
            str: Path to saved masked image
        """
        output_file = os.path.join(self.output_dir,
                                   os.path.basename(analytic_file).replace('.tif', f'_{combo_type}_cleaned.tif'))
        if self._skip_if_exists(output_file):
            return output_file

        scaled_data, meta = self._scale_to_reflectance(analytic_file)

        # Load UDM
        with rasterio.open(udm_file) as src_udm:
            udm_all = src_udm.read()
            unusable_mask = np.zeros_like(udm_all[0], dtype=bool)
            for band_idx in range(1,7):
                unusable_mask |= (udm_all[band_idx] == 1)

        masks = {}

        # Generate UDM masks
        masks["udm"] = np.where(unusable_mask, 0, 1).astype('float32')
        masks["udmbuffer"] = np.where(binary_dilation(unusable_mask, iterations=buffer_size), 0, 1).astype('float32')

        # Compute cloud score
        cloud_score = self._calculate_cloud_score(scaled_data)
        cloud_mask = cloud_score > 0.05

        # Generate CS masks
        masks["cs"] = (~cloud_mask).astype("float32")
        masks["lowcsbuffer"] = (~self.low_cs_buffer(cloud_mask)).astype("float32")
        masks["highcsbuffer"] = (~self.high_cs_buffer(cloud_mask)).astype("float32")

        # Parse combo type
        def parse_combo_type(combo_type):
            valid_parts = set(masks.keys())
            parts = []
            for token in combo_type.split("_"):
                if token in valid_parts:
                    parts.append(token)
                else:
                    raise ValueError(f"Invalid mask type: {token}. Must be one of {list(masks.keys())}")
            return parts

        parts = parse_combo_type(combo_type)

        # Combine all masks
        final_mask = np.ones_like(masks["udm"], dtype=bool)
        for part in parts:
            final_mask = np.logical_and(final_mask, masks[part])

        # Apply mask
        masked_data = scaled_data * final_mask.astype("float32")

        # Save output
        meta.update({'dtype': 'float32'})

        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(masked_data.astype('float32'))

        print(f"Combined mask ({combo_type}) image saved to: {output_file}")
        return output_file
