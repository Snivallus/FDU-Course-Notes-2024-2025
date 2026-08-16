import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def svd_compress(image, k):
    """
    Compress an image using Singular Value Decomposition (SVD).
    Only the first `k` singular values are kept.

    Parameters:
    - image: 2D numpy array representing the grayscale image.
    - k: The number of singular values to retain for compression.

    Returns:
    - compressed_image: The compressed image as a 2D numpy array.
    - num_singular_values: The number of singular values used.
    - compression_rate: The compression rate for the compressed image.
    """
    # Perform SVD decomposition
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    
    # Keep only the first k singular values
    U_k = U[:, :k]
    S_k = np.diag(S[:k])  # Create a diagonal matrix with the first k singular values
    Vt_k = Vt[:k, :]
    
    # Reconstruct the compressed image using the reduced SVD matrices
    compressed_image = np.dot(U_k, np.dot(S_k, Vt_k))
    
    # Calculate compression rate
    m, n = image.shape
    compression_rate = (k * (m + n + k)) / (m * n)

    # Return the compressed image, number of singular values used, and the compression rate
    return compressed_image, k, compression_rate

def plot_compressed_images(original, compressed_images, k_values, image_name, singular_values, compression_rates):
    """
    Plot the original image and all compressed images in one plot.

    Parameters:
    - original: Original grayscale image.
    - compressed_images: List of compressed images.
    - k_values: List of k-values used for each compression.
    - image_name: Name of the image for title.
    - singular_values: List of the number of singular values used for each k.
    - compression_rates: List of compression rates for each k.
    """
    num_images = len(compressed_images) + 1  # Including the original image
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original Image: {image_name}')
    axes[0].axis('off')

    # Show the number of non-zero singular values and compression rate under the original image
    axes[0].text(0.5, -0.15, f'Non-zero Singular Values: {singular_values}', ha='center', va='top', fontsize=10, transform=axes[0].transAxes)

    # Plot the compressed images for each k value
    for i, (compressed_image, k, comp_rate) in enumerate(zip(compressed_images, k_values, compression_rates)):
        axes[i + 1].imshow(compressed_image, cmap='gray')
        axes[i + 1].set_title(f'k = {k}')
        axes[i + 1].axis('off')
        axes[i + 1].text(0.5, -0.15, f'Compression Rate: {comp_rate:.2f}', ha='center', va='top', fontsize=10, transform=axes[i + 1].transAxes)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Open the image, convert it to grayscale, and then convert it to a numpy array
    image_path = 'DIP_Fig02.41(c)(einstein high contrast).tif'
    image_name = 'Einstein' 
    image = Image.open(image_path).convert('L')  # Convert the image to grayscale ('L' mode)
    image = np.array(image)  # Convert the grayscale image to a numpy array

    # Choose the number of singular values to use for compression
    k_values = [5, 10, 20, 30, 40]  # Example k-values to test
    compressed_images = []
    singular_values_list = []
    compression_rates_list = []

    # Generate compressed images for each k value
    for k in k_values:
        compressed_image, num_singular_values, compression_rate = svd_compress(image, k)
        compressed_images.append(compressed_image)
        singular_values_list.append(num_singular_values)
        compression_rates_list.append(compression_rate)
        print(f"Number of singular values used for k={k}: {num_singular_values}")
        print(f"Compression rate for k={k}: {compression_rate:.2f}")

    # Plot the original image and all compressed images
    plot_compressed_images(image, compressed_images, k_values, image_name, sum(singular_values_list), compression_rates_list)
