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

    option = 4
    if option == 1:
        image_path = 'Fig1038(a)(noisy_fingerprint).tif'
        image_name = 'Fig1038(a)(noisy_fingerprint)'
    elif option == 2:
        image_path = 'Fig1045(a)(iceberg).tif'
        image_name = 'Fig1045(a)(iceberg)'
    elif option == 3:
        image_path = 'Fig1016(a)(building_original).tif'
        image_name = 'Fig1016(a)(building_original)'  
    elif option == 4:
        image_path = 'Fig1034(a)(marion_airport).tif'
        image_name = 'Fig1034(a)(marion_airport)'        

    # Open the image, convert it to grayscale and then convert to a numpy array
    image = Image.open(image_path).convert('L')  # Convert the image to grayscale ('L' mode)
    image = np.array(image)  # Convert the grayscale image to a numpy array

    _, noisy_image = generate_noise(image, noise_type="salt_pepper", p_salt=0.005, p_pepper=0.005)
    save_path = f"noisy_{image_name}.tif"
    Image.fromarray(noisy_image).save(save_path, format="TIFF")