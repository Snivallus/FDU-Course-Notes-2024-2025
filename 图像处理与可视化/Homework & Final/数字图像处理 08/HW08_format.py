import nibabel as nib
import os

# Path to the specific MPRAGE scan file
mpr_file = "./data/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.img"

# Load the MPRAGE scan
img = nib.load(mpr_file)  # Load the .img file (header is automatically read from .hdr)
data = img.get_fdata()  # Extract the data as a NumPy array

# Ensure the data is 3D by squeezing any singleton dimensions
data = data.squeeze()  # Removes unnecessary dimensions (if any)

# Create a new NIfTI image using the loaded header and affine
nii_img = nib.Nifti1Image(data, affine=img.affine, header=img.header)

# Create output directory if it doesn't exist
output_dir = "nii_files"
os.makedirs(output_dir, exist_ok=True)

# Save the single MPRAGE scan as a NIfTI file
output_file = os.path.join(output_dir, "OAS1_0001_mpr-1.nii")
nib.save(nii_img, output_file)
print(f"Single MPRAGE NIfTI file saved to {output_file}")
