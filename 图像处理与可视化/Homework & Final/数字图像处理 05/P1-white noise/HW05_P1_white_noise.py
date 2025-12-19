import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_histogram(image, num_bins=256):
    """
    Compute the grayscale histogram of an image.
    
    :param image: Grayscale image as a numpy array.
    :param num_bins: Number of bins for the histogram (default is 256).
    :return: The histogram and bin edges.
    """
    # Flatten the image and compute the histogram with the specified number of bins
    histogram, bin_edges = np.histogram(image.ravel(), bins=num_bins, range=[0, num_bins])
    return histogram, bin_edges

def generate_white_noise(image, coefficient=0.1):
    """
    Generate white noise with the same size as the input image.
    
    :param image: The input image (numpy array) to match the size.
    :param coefficient: A coefficient to control the amount of noise to be added to the image.
    :return: The spatial domain representation of the generated noise.
    """
    # Get the image size (height and width)
    height, width = image.shape
    
    # Generate random phase values uniformly distributed between [0, 2pi]
    random_phase = np.random.uniform(0, 2 * np.pi, (height, width))
    
    # The magnitude of the Fourier transform is set to 1 (constant), and the phase is random
    magnitude = np.ones((height, width))
    complex_spectrum = magnitude * np.exp(1j * random_phase)
    
    # Perform inverse Fourier transform to get the noise in the spatial domain
    noise = np.fft.ifft2(complex_spectrum).real

    # 1. Display the original image
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')  # Hide axes for better visualization

    # 2. Display the Fourier transform magnitude spectrum of the original image
    f_transform = np.fft.fftshift(np.fft.fft2(image))  # Apply FFT and shift zero frequency to center
    magnitude_spectrum = np.abs(f_transform)  # Compute the magnitude spectrum
    plt.subplot(3, 3, 2)
    plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')  # Apply log scale for better visibility
    plt.title('Fourier Spectrum of Image')
    plt.axis('off')  # Hide axes

    # 3. Plot the grayscale histogram of the original image
    hist_image, bin_edges = compute_histogram(image)  # Compute histogram
    plt.subplot(3, 3, 3)
    plt.plot(bin_edges[:-1], hist_image)  # Plot histogram
    plt.title('Original Image Histogram')

    # 4. Display the phase spectrum of the noise
    phase_spectrum = np.angle(complex_spectrum)  # Compute the phase of the noise
    plt.subplot(3, 3, 4)
    plt.imshow(phase_spectrum, cmap='gray')  # Display the phase spectrum
    plt.title('Phase Spectrum of Noise')
    plt.axis('off')

    # 5. Show the frequency spectrum of the noise
    plt.subplot(3, 3, 5)
    plt.imshow(np.log(1 + np.abs(complex_spectrum)), cmap='gray')  # Display the phase spectrum
    plt.title('Power Spectrum of Noise')
    plt.axis('off')

    # 6. Plot the grayscale histogram of the noise after normalization
    noise_normalized = 255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))  # Normalize to [0, 255]
    hist_noise, bin_edges = compute_histogram(noise_normalized.astype(np.uint8))  # Compute histogram of normalized noise
    plt.subplot(3, 3, 6)
    plt.plot(bin_edges[:-1], hist_noise)  # Plot histogram
    plt.title('Noise Histogram')

    # 7. Display the noise in the spatial domain (normalized to [0, 255] range)
    plt.subplot(3, 3, 7)
    plt.imshow(noise_normalized.astype(np.uint8), cmap='gray')  # Display noise in spatial domain
    plt.title('Noise in Spatial Domain')
    plt.axis('off')

    # 8. Add the noise to the original image and display the result
    noisy_image = np.clip(image + noise_normalized * coefficient, 0, 255).astype(np.uint8)  # Clip values to [0, 255] range
    plt.subplot(3, 3, 8)
    plt.imshow(noisy_image, cmap='gray')  # Display noisy image
    plt.title(f'Noisy Image (Coeff={coefficient})')
    plt.axis('off')

    # 9. Plot the grayscale histogram of the noisy image
    hist_noisy_image, bin_edges = compute_histogram(noisy_image)  # Compute histogram of noisy image
    plt.subplot(3, 3, 9)
    plt.plot(bin_edges[:-1], hist_noisy_image)  # Plot histogram
    plt.title('Histogram of Noisy Image')
    plt.tight_layout()  # Adjust subplots for better spacing

    # Save the figure containing all the images
    save_path = f"white_noise.png"
    plt.savefig(save_path)
    plt.show()

    # Save the final filtered image
    final_image_path = f"image_with_white_noise.tif"
    Image.fromarray(noisy_image).save(final_image_path, format="TIFF")

    return noise

if __name__ == "__main__":
    # Set seed
    np.random.seed(51)

    # Load the image
    image_path = 'DIP Fig 05.03 (original_pattern).tif'  # Path to the image file
    image_name = 'DIP Fig 05.03 (original_pattern)'  # Image name (for reference)

    # Open the image, convert it to grayscale and then convert to a numpy array
    img = Image.open(image_path).convert('L')  # Convert the image to grayscale ('L' mode)
    image_array = np.array(img)  # Convert the grayscale image to a numpy array

    # Generate white noise with the same size as the image
    noise = generate_white_noise(image_array, coefficient=0.1)
