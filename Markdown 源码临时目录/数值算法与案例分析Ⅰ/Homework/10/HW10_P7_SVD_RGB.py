import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def svd_compress_color(image, k):
    """
    Compress a color image using Singular Value Decomposition (SVD).
    Only the first `k` singular values are kept for each RGB channel.

    Parameters:
    - image: 3D numpy array representing the color image (height, width, 3).
    - k: The number of singular values to retain for compression.

    Returns:
    - compressed_image: The compressed color image as a 3D numpy array.
    - num_singular_values: The number of non-zero singular values used for each channel.
    - compression_rate: The compression rate for each channel.
    """
    # Get image dimensions
    m, n, _ = image.shape

    # Initialize the compressed image
    compressed_image = np.zeros_like(image)

    # List to store compression rates for each channel
    compression_rates = []
    total_non_zero_singular_values = 0

    # Apply SVD compression on each RGB channel
    for i in range(3):  # Loop through each channel (0 = Red, 1 = Green, 2 = Blue)
        channel = image[:, :, i]
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)

        # Number of non-zero singular values
        num_singular_values = np.sum(S > 1e-10)  # Consider values larger than a small threshold as non-zero

        # Keep only the first k singular values
        U_k = U[:, :k]
        S_k = np.diag(S[:k])  # Create a diagonal matrix with the first k singular values
        Vt_k = Vt[:k, :]

        # Reconstruct the compressed channel using the reduced SVD matrices
        compressed_channel = np.dot(U_k, np.dot(S_k, Vt_k))

        # Store the compressed channel back into the image
        compressed_image[:, :, i] = compressed_channel

        # Update total number of non-zero singular values
        total_non_zero_singular_values += num_singular_values

        # Calculate the compression rate for this channel
        compression_rate = (k * (m + n + k)) / (m * n)
        compression_rates.append(compression_rate)

    # Return the compressed image, number of singular values, and compression rates
    return compressed_image, total_non_zero_singular_values, compression_rates

def plot_compressed_images_color(original, compressed_images, k_values, image_name, singular_values, compression_rates):
    """
    Plot the original image and all compressed images in one plot for color images.
    Also show the number of singular values and compression rates under the original image.

    Parameters:
    - original: Original color image.
    - compressed_images: List of compressed color images.
    - k_values: List of k-values used for each compression.
    - image_name: Name of the image for title.
    - singular_values: List of the number of singular values used for each k.
    - compression_rates: List of compression rates for each k.
    """
    num_images = len(compressed_images) + 1  # Including the original image
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(original)
    axes[0].set_title(f'Original Image: {image_name}')
    axes[0].axis('off')

    # Show the number of non-zero singular values and the compression rate under the original image
    axes[0].text(0.5, -0.15, f'Non-zero Singular Values: {singular_values}', ha='center', va='top', fontsize=10, transform=axes[0].transAxes)

    # Plot the compressed images for each k value
    for i, (compressed_image, k, comp_rate) in enumerate(zip(compressed_images, k_values, compression_rates)):
        axes[i + 1].imshow(compressed_image)
        axes[i + 1].set_title(f'k = {k}')
        axes[i + 1].axis('off')
        axes[i + 1].text(0.5, -0.15, f'Compression Rate: {comp_rate:.2f}', ha='center', va='top', fontsize=10, transform=axes[i + 1].transAxes)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Open the image, convert it to RGB, and then convert it to a numpy array
    image_path = 'DIP_Fig06.46(a)(lenna_original_RGB).tif'
    image_name = 'Lenna' 
    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
    image = np.array(image)  # Convert the image to a numpy array

    # Choose the number of singular values to use for compression
    k_values = [10, 100, 200, 300, 500]  # Example k-values to test
    compressed_images = []
    singular_values_list = []
    compression_rates_list = []

    # Generate compressed images for each k value
    for k in k_values:
        compressed_image, num_singular_values, compression_rates = svd_compress_color(image, k)
        compressed_images.append(compressed_image)
        singular_values_list.append(num_singular_values)
        compression_rates_list.append(np.mean(compression_rates))  # Mean compression rate for all channels
        print(f"Number of singular values used for k={k}: {num_singular_values}")
        print(f"Average compression rate for k={k}: {np.mean(compression_rates):.2f}")

    # Plot the original image and all compressed images
    plot_compressed_images_color(image, compressed_images, k_values, image_name, sum(singular_values_list), compression_rates_list)
