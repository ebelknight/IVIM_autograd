#creating a sample mask for demonstration purposes

import numpy as np
import nibabel as nib

# Parameters for the binary mask
dimensions = (110, 110, 10)
side_length = 50

# Create an empty binary mask
binary_mask = np.zeros(dimensions, dtype=np.uint8)

# Calculate the center and the start/end points of the square
center = (dimensions[0] // 2, dimensions[1] // 2)
start_x = center[0] - side_length // 2
end_x = center[0] + side_length // 2
start_y = center[1] - side_length // 2
end_y = center[1] + side_length // 2

# Fill the square region with 1s
binary_mask[start_x:end_x, start_y:end_y, :] = 1

# Convert the numpy array to a NIfTI image
binary_mask_img = nib.Nifti1Image(binary_mask, np.eye(4))

# Save the NIfTI image to a file
binary_mask_file_nii = 'binary_mask.nii'
nib.save(binary_mask_img, binary_mask_file_nii)

print(f"Binary mask saved as {binary_mask_file_nii}")
