from .base import CloudClearBase
import rasterio
import numpy as np
from scipy.ndimage import binary_dilation
import os

class RapidEye(CloudClearBase):
    def __init__(self, tmp_dir, output_dir, aoi):
        super().__init__(tmp_dir, output_dir, aoi)
        # Dark pixel threshold - adjust this number between 0.0-1.0 if needed
        # Higher values = more aggressive dark pixel masking
        self.dark_threshold = 0.05  # Default threshold (5% reflectance)

    def _mask_dark_pixels(self, scaled_data):
        """
        Mask pixels based only on the red edge band (B4)
        Args:
            scaled_data: Image data already scaled to reflectance (0-1)
        Returns:
            Binary mask where 1=valid pixels, 0=dark pixels to be masked
        """
        # Use Red Edge band (B4, index 3)
        red_edge_band = scaled_data[3, :, :]

        # Identify pixels below threshold in red edge band
        dark_mask = red_edge_band < self.dark_threshold
        
        # Buffer the mask slightly to avoid edge artifacts
        buffered_mask = binary_dilation(dark_mask, iterations=2)
        
        return ~buffered_mask # Invert so 1=valid, 0=dark

    def _calculate_cloud_score(self, scaled_data):
        """
        Custom cloud scoring using reflectance values (0-1).
        Assumes RapidEye bands are ordered as [B, G, R, RedEdge, NIR].
        """
        blue = scaled_data[0, :, :]
        green = scaled_data[1, :, :]
        red = scaled_data[2, :, :]
        rededge = scaled_data[3, :, :]
        nir = scaled_data[4, :, :]

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


    def udm_mask(self, udm_file, analytic_file):
        """
        Applies the UDM mask to the analytic file and saves the masked image

        Args:
            udm_file (str): Path to the UDM file
            analytic_file (str): Path to the input image file
        Returns:
            output_file (str): Path to the output file, masked image
        """
        # Read and scale analytic data
        scaled_data, meta = self._scale_to_reflectance(analytic_file)

        # Read UDM and generate binary mask
        with rasterio.open(udm_file) as src_udm:
            udm = src_udm.read(1)
            unusable_mask = udm == 2
            # Invert mask: 1 = usable, 0 = masked
            mask = np.where(unusable_mask, 0, 1)

        # Apply the mask to scaled data
        masked_data = scaled_data * mask[np.newaxis, :, :]

        # Prepare output path and metadata
        meta.update({'dtype': 'float32'})
        output_file = os.path.join(str(self.output_dir), os.path.basename(analytic_file).replace('.tif', '_udm_cleaned.tif'))

        # Save result
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(masked_data.astype('float32'))

        print(f"UDM-only masked image saved to: {output_file}")
        return output_file


    def udm_buffer_mask(self, udm_file, analytic_file, buffer_size = 3):
        """
        Applies the buffered UDM mask to the analytic file and saves the masked image
        Args:
            udm_file (str): Path to the UDM file
            analytic_file (str): Path to the input image file
            buffer_size (int): Buffer size to use for masking. Defaults to 3.
        Returns:
            output_file (str): Path to the output file, masked image
        """
        # Read and scale analytic data
        scaled_data, meta = self._scale_to_reflectance(analytic_file)

        # Read UDM file
        with rasterio.open(udm_file) as src_udm:
            udm = src_udm.read(1)

            # Create unusable mask (UDM value of 2 means cloud/shadow/etc.)
            unusable_mask = udm == 2

            # Apply buffer
            buffered_mask = binary_dilation(unusable_mask, iterations=buffer_size)

            # Invert 1 = valid, 0 = masked out
            mask = np.where(buffered_mask, 0, 1)

        # Apply mask to image
        masked_data = scaled_data * mask[np.newaxis, :, :]

        # update metadata
        meta.update({'dtype': 'float32'})
        output_file = os.path.join(str(self.output_dir), os.path.basename(analytic_file).replace('.tif', '_udm_buffer_cleaned.tif'))

        # Save output
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(masked_data.astype('float32'))

        print(f"UDM-buffered masked image saved to: {output_file}")
        return output_file



    def cs_mask(self, analytic_file):
        """
        Applies the custom cloud score mask to the analytic file and saves the masked image
        Args:
             analytic_file (str): Path to the input image file
        Returns:
            output_file (str): Path to the output file, masked image
        """
        # Read and scale analytic data
        scaled_data, meta = self._scale_to_reflectance(analytic_file)

        # Compute cloud score
        cloud_score = self._calculate_cloud_score(scaled_data)

        # Define threshold - pixels with score > 0.05 are considered cloudy
        cloud_mask = cloud_score > 0.05

        # Get dark pixel mask using scaled data
        dark_pixel_mask = self._mask_dark_pixels(scaled_data)

        # Combine Masks (1 = clear, 0 = cloudy or dark)
        final_mask = np.logical_and(~cloud_mask, dark_pixel_mask).astype('float32')

        # Apply final mask to scaled data
        masked_data = scaled_data * final_mask

        # Prepare output
        meta.update({'dtype': 'float32'})
        output_file = os.path.join(str(self.output_dir), os.path.basename(analytic_file).replace('.tif', '_cs_mask_cleaned.tif'))

        # Save the result
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(masked_data.astype('float32'))

        print(f"Custom cloud-masked image saved to: {output_file}")
        return output_file


    def cs_buffer_mask(self, analytic_file, buffer_size = 3):
        """
        Applies the custom cloud score mask with a buffer to the analytic file and saves the masked image
        Args:
            analytic_file (str): Path to the input image file
            buffer_size (int): Buffer size to use for masking. Defaults to 3.
        Returns:
            output_file (str): Path to the output file, masked image
        """
        # Read and scale analytic data
        scaled_data, meta = self._scale_to_reflectance(analytic_file)

        # Compute cloud score
        cloud_score = self._calculate_cloud_score(scaled_data)

        # Threshold - pixels with score > 0.05 are considered cloudy
        cloud_mask = cloud_score > 0.05

        # Buffer cloud mask
        buffered_cloud_mask = binary_dilation(cloud_mask, iterations=buffer_size)

        # Get dark pixel mask using scaled data
        dark_pixel_mask = self._mask_dark_pixels(scaled_data)

        # Combine masks
        final_mask = np.logical_and(~buffered_cloud_mask, dark_pixel_mask).astype('float32')

        # Apply mask
        masked_data = scaled_data * final_mask

        # Prepare output
        meta.update({'dtype': 'float32'})
        output_file = os.path.join(str(self.output_dir), os.path.basename(analytic_file).replace('.tif', '_cs_buffer_mask_cleaned.tif'))

        # Save the masked image
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(masked_data.astype('float32'))

        print(f"Custom cloud masked image with buffer saved to: {output_file}")
        return output_file

    def combined_mask(self, analytic_file, udm_file, combo_type="udm_cs", buffer_size=25):
        """
        Applies a combination of UDM and CS masking strategies
        """
        scaled_data, meta = self._scale_to_reflectance(analytic_file)

        # Load UDM
        with rasterio.open(udm_file) as src_udm:
            udm = src_udm.read(1)
        unusable_mask = udm == 2
        masks = {}

        # Generate UDM masks
        masks["udm"] = np.where(unusable_mask, 0, 1).astype('float32')
        masks["udmbuffer"] = np.where(binary_dilation(unusable_mask, iterations=buffer_size), 0, 1).astype('float32')

        # Compute cloud score
        cloud_score = self._calculate_cloud_score(scaled_data)
        cloud_mask = cloud_score > 0.05
        cloud_mask_buffered = binary_dilation(cloud_mask, iterations=buffer_size)

        # Dark pixel mask
        dark_pixel_mask = self._mask_dark_pixels(scaled_data)

        # Generate CS masks
        cs_valid = np.logical_and(~cloud_mask, dark_pixel_mask)
        cs_buffer_valid = np.logical_and(~cloud_mask_buffered, dark_pixel_mask)

        masks["cs"] = cs_valid.astype('float32')
        masks["csbuffer"] = cs_buffer_valid.astype('float32')

        # Validate combo type
        def parse_combo_type(combo_type):
            valid_parts = {"udm", "udmbuffer", "cs", "csbuffer"}
            parts = []
            for token in combo_type.split("_"):
                if token in valid_parts:
                    parts.append(token)
                else:
                    raise ValueError(f"Invalid mask type: {token}. Must be one of {list(masks.keys())}")
            return parts

        parts = parse_combo_type(combo_type)

        # Combine all masks
        final_mask = np.ones_like(masks["udm"])
        for part in parts:
            final_mask = np.logical_and(final_mask, masks[part])


        # Apply final mask
        masked_data = scaled_data * final_mask

        # Save output
        meta.update({'dtype': 'float32'})

        output_file = os.path.join(
            str(self.output_dir),
            os.path.basename(analytic_file).replace('.tif', f'_{combo_type}_cleaned.tif')
        )

        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(masked_data.astype('float32'))

        print(f"Combined mask ({combo_type}) image saved to: {output_file}")
        return output_file






