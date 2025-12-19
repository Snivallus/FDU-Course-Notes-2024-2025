import cv2
import numpy as np
from PIL import Image

def binarize_using_otsu(image_path, save_path='binarized_image.bmp'):
    """
    Convert an image to a binary image using Otsu's thresholding and save it as BMP format.
    
    Parameters:
    image_path: str - Path to the input image
    save_path: str - Path to save the binarized image in BMP format
    """
    # Open the image, convert it to grayscale and then convert to a numpy array
    image = Image.open(image_path).convert('L')  # Convert the image to grayscale ('L' mode)
    image = np.array(image)  # Convert the grayscale image to a numpy array
    
    # Apply Otsu's thresholding to binarize the image
    # The second return value is the threshold value, we don't need it for just binarizing
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save the binary image as BMP format
    cv2.imwrite(save_path, binary_image)
    print(f"Binarized image saved at: {save_path}")

# Example usage
image_path = 'DIP 09.11(a)(noisy_fingerprint).bmp'  # Replace with your actual image path
save_path = 'binarized_noisy_fingerprint.bmp'  # Path to save the binarized image
binarize_using_otsu(image_path, save_path=save_path)
