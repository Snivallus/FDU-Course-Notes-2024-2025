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

def generate_noise(image, noise_type="gaussian", coefficient=0.1, **params):
    """
    Generate different types of noise and add it to the input image.
    
    :param image: The input image (numpy array) to match the size.
    :param noise_type: Type of noise to generate ("gaussian", "rayleigh", "gamma", "uniform", "salt_pepper").
    :param coefficient: A coefficient to control the amount of noise to be added to the image.
    :param params: Additional parameters specific to the noise type.
    :return: The spatial domain representation of the generated noise.
    """
    height, width = image.shape

    if noise_type == "gaussian":
        # Gaussian noise: mean = mu, std deviation = sigma
        mu = params.get("mu", 0)
        sigma = params.get("sigma", 25)
        noise = np.random.normal(mu, sigma, (height, width))
    
    elif noise_type == "rayleigh":
        # Rayleigh noise: scale = b, offset = a
        a = params.get("a", 0)
        b = params.get("b", 25)
        noise = np.random.rayleigh(b, (height, width)) + a

    elif noise_type == "gamma":
        # Gamma noise: shape = alpha, rate = lambda
        alpha = params.get("alpha", 2)
        lambd = params.get("lambda", 1)
        noise = np.random.gamma(alpha, 1/lambd, (height, width))

    elif noise_type == "uniform":
        # Uniform noise: min = a, max = b
        a = params.get("a", 0)
        b = params.get("b", 255)
        noise = np.random.uniform(a, b, (height, width))

    elif noise_type == "salt_pepper":
        # Salt-and-pepper noise: p_salt, p_pepper (density)
        p_salt = params.get("p_salt", 0.01)
        p_pepper = params.get("p_pepper", 0.01)
        
        # Generate a random mask for salt and pepper noise
        mask = np.random.random((height, width))
        image[mask < p_salt] = 255         # salt (white pixels)
        image[mask > (1 - p_pepper)] = 0   # pepper (black pixels)

        noise = 128 * np.ones_like(image).astype(np.uint8)
        noise[mask < p_salt] = 255
        noise[mask > (1 - p_pepper)] = 0
        
        return noise, image

    else:
        raise ValueError("Unsupported noise type. Choose from 'gaussian', 'rayleigh', 'gamma', 'uniform', 'salt_pepper'.")

    # Use noise to he noisy image
    noise_normalized = (255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))).astype(np.uint8)
    image = np.clip(image + noise * coefficient, 0, 255).astype(np.uint8)

    return noise_normalized, image

if __name__ == "__main__":
    # Set seed
    np.random.seed(51)

    # Load the image
    image_path = 'DIP Fig 05.03 (original_pattern).tif'  # Path to the image file
    image_name = 'DIP Fig 05.03 (original_pattern)'  # Image name (for reference)

    # Open the image, convert it to grayscale and then convert to a numpy array
    image = Image.open(image_path).convert('L')  # Convert the image to grayscale ('L' mode)
    image = np.array(image)  # Convert the grayscale image to a numpy array

    # 1. Gaussian Noise
    noise, noisy_image = generate_noise(image, noise_type="gaussian", coefficient=5, mu=3, sigma=1)
    plt.figure(figsize=(20, 16))
    plt.subplot(5, 4, 1)
    plt.imshow(noise, cmap='gray')
    plt.title('Gaussian Noise')
    plt.axis('off')

    plt.subplot(5, 4, 2)
    plt.hist(noise.ravel(), bins=256, color='black', histtype='step')
    plt.title('Gaussian Noise Histogram')

    plt.subplot(5, 4, 3)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Polluted Image (Gaussian)')
    plt.axis('off')

    plt.subplot(5, 4, 4)
    plt.hist(noisy_image.ravel(), bins=256, color='black', histtype='step')
    plt.title('Polluted Image Histogram (Gaussian)')

    # 2. Rayleigh Noise
    noise, noisy_image = generate_noise(image, noise_type="rayleigh", coefficient=0.2, a=0, b=25)
    plt.subplot(5, 4, 5)
    plt.imshow(noise, cmap='gray')
    plt.title('Rayleigh Noise')
    plt.axis('off')

    plt.subplot(5, 4, 6)
    plt.hist(noise.ravel(), bins=256, color='black', histtype='step')
    plt.title('Rayleigh Noise Histogram')

    plt.subplot(5, 4, 7)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Polluted Image (Rayleigh)')
    plt.axis('off')

    plt.subplot(5, 4, 8)
    plt.hist(noisy_image.ravel(), bins=256, color='black', histtype='step')
    plt.title('Polluted Image Histogram (Rayleigh)')

    # 3. Gamma Noise
    noise, noisy_image = generate_noise(image, noise_type="gamma", coefficient=3, alpha=3, lambd=1)
    plt.subplot(5, 4, 9)
    plt.imshow(noise, cmap='gray')
    plt.title('Gamma Noise')
    plt.axis('off')

    plt.subplot(5, 4, 10)
    plt.hist(noise.ravel(), bins=256, color='black', histtype='step')
    plt.title('Gamma Noise Histogram')

    plt.subplot(5, 4, 11)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Polluted Image (Gamma)')
    plt.axis('off')

    plt.subplot(5, 4, 12)
    plt.hist(noisy_image.ravel(), bins=256, color='black', histtype='step')
    plt.title('Polluted Image Histogram (Gamma)')

    # 4. Uniform Noise
    noise, noisy_image = generate_noise(image, noise_type="uniform", coefficient=0.1, a=0, b=255)
    plt.subplot(5, 4, 13)
    plt.imshow(noise, cmap='gray')
    plt.title('Uniform Noise')
    plt.axis('off')

    plt.subplot(5, 4, 14)
    plt.hist(noise.ravel(), bins=256, color='black', histtype='step')
    plt.title('Uniform Noise Histogram')

    plt.subplot(5, 4, 15)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Polluted Image (Uniform)')
    plt.axis('off')

    plt.subplot(5, 4, 16)
    plt.hist(noisy_image.ravel(), bins=256, color='black', histtype='step')
    plt.title('Polluted Image Histogram (Uniform)')

    # 5. Salt-Pepper Noise
    noise, noisy_image = generate_noise(image, noise_type="salt_pepper", p_salt=0.01, p_pepper=0.01)
    plt.subplot(5, 4, 17)
    plt.imshow(noise, cmap='gray')
    plt.title('Salt-Pepper Noise')
    plt.axis('off')

    plt.subplot(5, 4, 18)
    plt.hist(noise.ravel(), bins=256, color='black', histtype='step')
    plt.title('Salt-Pepper Noise Histogram')

    plt.subplot(5, 4, 19)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Polluted Image (Salt-Pepper)')
    plt.axis('off')

    plt.subplot(5, 4, 20)
    plt.hist(noisy_image.ravel(), bins=256, color='black', histtype='step')
    plt.title('Polluted Image Histogram (Salt-Pepper)')
    plt.tight_layout()

    # Save the figure containing all the images
    save_path = f"spatial_noise.png"
    plt.savefig(save_path)
    plt.show()