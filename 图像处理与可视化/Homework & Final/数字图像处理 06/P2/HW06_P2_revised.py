import numpy as np
from PIL import Image
from numba import njit
import matplotlib.pyplot as plt
import cv2

def compute_distance_transform(image: np.array):
    return cv2.distanceTransform(image, cv2.DIST_C, 5).astype(np.uint8)

# Erosion function
def erode(image: np.array, threshold: int):
    """
    Parameters:
    image: Grayscale array of the image
    threshold: Threshold value to determine erosion effect (distance threshold)
    Returns: Grayscale array of the image after erosion
    """
    distance_transform = compute_distance_transform(image)
    image[(distance_transform > 0) & (distance_transform <= threshold)] = 0
    return image

# Dilation function
def dilate(image: np.array, threshold: int):
    """
    Parameters:
    image: Grayscale array of the image
    threshold: Threshold value to determine dilation effect (distance threshold)
    Returns: Grayscale array of the image after dilation
    """
    distance_transform = compute_distance_transform(255 - image)
    image[(distance_transform > 0) & (distance_transform <= threshold)] = 255
    return image

# Closing operation, used for filling holes
def closing(image: np.array, threshold: int):
    """
    Parameters:
    image: Grayscale array of the image
    threshold: Threshold value to determine closing effect (distance threshold);
    Returns: Grayscale array of the image after performing "closing" (dilation followed by erosion)
    """
    image = dilate(image, threshold)  # First dilation
    image = erode(image, threshold)   # Then erosion
    return image

# Opening operation, used for removing noise
def opening(image: np.array, threshold: int):
    """
    Parameters:
    image: Grayscale array of the image
    block_size: Size of the structuring element, measured according to the Chebyshev distance;
    Returns: Grayscale array of the image after performing "opening" (erosion followed by dilation)
    """
    image = erode(image, threshold)   # First erosion
    image = dilate(image, threshold)  # Then dilation
    return image

if __name__ == '__main__':

    # First, import the image and convert it to a numpy array
    option = 2
    if option == 1:
        image_path = 'zmic_fdu_noise.bmp'
        image_name = 'zmic_fdu_noise'
        open_threshold = 3
        close_threshold = 3
        inverse = True
    else:
        image_path = 'DIP 09.11(a)(noisy_fingerprint).bmp'
        image_name = 'DIP 09.11(a)(noisy_fingerprint)'
        open_threshold = 3
        close_threshold = 4
        inverse = False

    # Open the image, convert it to grayscale and then convert to a numpy array
    image = Image.open(image_path).convert('L')  # Convert the image to grayscale ('L' mode)
    image = np.array(image)  # Convert the grayscale image to a numpy array  
    print(np.where((image > 0) & (image < 255)))
    if inverse == True:
        image = 255 - image # Invert the pixel values (background becomes foreground and vice versa)     
    processed_image = np.copy(image)

    if inverse == True:
        # Perform closing operation on the image to fill holes
        processed_image = closing(processed_image, threshold=close_threshold)
        # Perform opening operation on the image to remove noise
        processed_image = opening(processed_image, threshold=open_threshold)
    else:
        # Perform opening operation on the image to remove noise
        processed_image = opening(processed_image, threshold=open_threshold)
        # Perform closing operation on the image to fill holes
        processed_image = closing(processed_image, threshold=close_threshold)

    # Invert the image back to its original form (foreground becomes background and vice versa)
    if inverse == True:
        image = 255 - image
        processed_image = 255 - processed_image

    # Save result
    save_path = f"processed_{image_name}.bmp"
    Image.fromarray(processed_image.astype(np.uint8)).save(save_path, format="BMP")

    # Plot the original and processed images side by side for comparison
    plt.figure(figsize=(12, 6))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')  # Hide axis ticks

    # Plot the processed image
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap='gray')
    plt.title("Processed Image")
    plt.axis('off')  # Hide axis ticks

    # Display the comparison
    plt.tight_layout()
    save_path = f"Comparision-{image_name}.png"
    plt.savefig(save_path)
    plt.show()