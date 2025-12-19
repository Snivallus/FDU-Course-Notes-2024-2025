import nibabel as nib
import numpy as np
import os

# Path to the specific MPRAGE scan file
mpr_file = "./data/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.img"

# Load the MPRAGE scan
img = nib.load(mpr_file)  # Load the .img file
data = img.get_fdata()  # Extract the data as a NumPy array

# Ensure the data is 3D by removing singleton dimensions
data = np.squeeze(data)  # Removes the 4th dimension if it exists

# Choose a slice to segment (e.g., the middle slice along the z-axis)
slice_idx = data.shape[2] // 2  # Middle slice
segmented_data = np.zeros_like(data)
segmented_data[:, :, slice_idx-5:slice_idx+5] = data[:, :, slice_idx-5:slice_idx+5]
segmented_img = nib.Nifti1Image(segmented_data, affine=img.affine, header=img.header)

# Save the segmented slice as an image for inspection
output_dir = "nii_files"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "segmented_brain.nii")
nib.save(segmented_img, output_file)
print(f"Segmented NIfTI file saved to {output_file}")
