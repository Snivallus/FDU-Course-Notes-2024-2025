import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import itertools

def frequency_domain_center_detection(image, image_name, min_distance=25):
    # Plotting setup: Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    
    # Display the original image in the first subplot
    M, N = image.shape
    axs[0, 0].imshow(image, cmap="gray")
    axs[0, 0].set_title(f"Original Image - size: {M}x{N}")
    
    # Padding
    P, Q = 2 * M, 2 * N
    padded_image = np.zeros((P, Q))
    padded_image[:M, :N] = image

    # Step 1: Apply Fourier transform centering by multiplying with (-1)^(x+y)
    # This shifts the zero-frequency component to the center
    centered_image = padded_image * np.fromfunction(lambda x, y: (-1)**(x + y), (P, Q))
    axs[0, 1].imshow(np.clip(centered_image, 0, 255).astype(np.uint8), cmap="gray")
    axs[0, 1].set_title(f"Centered Image - Center: ({N},{M})")

    # Step 2: Perform 2D Fourier Transform on the centered image to obtain the frequency spectrum
    F_uv = np.fft.fft2(centered_image)
    display_spectrum = np.log(1 + np.abs(F_uv))  # Apply log scaling to visualize the spectrum better
    axs[1, 0].imshow(display_spectrum, cmap="gray")
    axs[1, 0].set_title("Original Spectrum")

    # Step 3: Calculate the mean and standard deviation of the spectrum to identify bright spots
    mean_val = np.mean(display_spectrum)
    std_val = np.std(display_spectrum)
    threshold = mean_val + 2 * std_val  # Set threshold for identifying bright spots (based on mean and std)

    # Get coordinates of bright spots where the spectrum exceeds the threshold
    bright_spots = np.argwhere(display_spectrum > threshold)
    highlight_spectrum = np.ones_like(display_spectrum) * 255  # Start with a white background
    for i, (x, y) in enumerate(bright_spots):
        distance_from_center = np.sqrt((x - M)**2 + (y - N)**2)
        if distance_from_center > min_distance:
            highlight_spectrum[x, y] = 0  # Mark bright spots in black
    axs[1, 1].imshow(highlight_spectrum, cmap="gray")
    axs[1, 1].set_title("Highlighted Spectrum")

    # Save the figure containing all the images to a file
    save_path = f"frequency_domain_center_detection_{image_name}.png"
    plt.tight_layout()  # Adjust subplots for better layout
    plt.savefig(save_path)  # Save the figure
    plt.show()  # Display the plot

def frequency_domain_notch(image, centers, cutoffs, image_name, filter_types=[0], window_size=(5,5)):
    # Assert that filter_type is a list of integers, each greater than -2
    assert all(isinstance(filter_type, int) and filter_type >= -2 for filter_type in filter_types), \
        "All elements of filter_type must be integers greater than or equal to -2"
    
    # Plotting setup
    fig, axs = plt.subplots(2, 4, figsize=(20, 16))
    axs[0, 0].imshow(image, cmap="gray")
    axs[0, 0].set_title("Original Image")

    # Step 1: Multiply padded image by (-1)^(x+y) to center the Fourier transform
    M, N = image.shape
    P, Q = 2 * M, 2 * N
    padded_image = np.zeros((P, Q))
    padded_image[:M, :N] = image
    centered_image = padded_image * np.fromfunction(lambda x, y: (-1)**(x + y), (P, Q))
    
    # Display centered image
    axs[0, 1].imshow(np.clip(centered_image, 0, 255).astype(np.uint8), cmap="gray")
    axs[0, 1].set_title(f"Centered Image - Center: ({N},{M})")

    # Step 2: Compute the 2D Fourier Transform of the centered image
    G_uv = np.fft.fft2(centered_image)
    
    # Display original spectrum
    axs[0, 2].imshow(np.log(1 + np.abs(G_uv)), cmap="gray")
    axs[0, 2].set_title("Original Spectrum")

    # Step 3: Create the notch filter with multiple notch centers
    u, v = np.meshgrid(np.arange(0, P), np.arange(0, Q), indexing='ij')

    # Initialize the combined notch filter with ones (multiplicative identity for filters)
    H_uv = np.ones((P, Q))

    # Loop over each "filter_type" to construct the filter
    for i, filter_type in enumerate(filter_types):
        for center, cutoff in zip(centers[i], cutoffs[i]):
            if filter_type == -2:  # Line filter (horizontal or vertical)
                if center[3] == 0:  # Horizontal line
                    y_val = center[0]  # Fixed y-value for the horizontal line
                    x_start, x_end = np.clip(center[1], 0, Q), np.clip(center[2], 0, Q-1)
                    mask_x_range = np.arange(x_start, x_end + 1)
                    mask_y_range = np.arange(np.clip(y_val - cutoff, 0, P), np.clip(y_val + cutoff, 0, P-1) + 1)
                    mask = list(itertools.product(mask_y_range, mask_x_range))
                    mask += [(P - i - 1, Q - j - 1) for i, j in mask]

                    # Convert to numpy arrays for vectorized assignment
                    mask_y, mask_x = zip(*mask)
                    mask_y, mask_x = np.array(mask_y), np.array(mask_x)
                    H_uv[mask_y, mask_x] = 0

                else:  # Vertical line
                    x_val = center[0]  # Fixed x-value for the vertical line
                    y_start, y_end = np.clip(center[1], 0, P), np.clip(center[2], 0, P-1)
                    mask_y_range = np.arange(y_start, y_end + 1)
                    mask_x_range = np.arange(np.clip(x_val - cutoff, 0, Q), np.clip(x_val + cutoff, 0, Q-1) + 1)
                    mask = list(itertools.product(mask_y_range, mask_x_range))
                    mask += [(P - i - 1, Q - j - 1) for i, j in mask]

                    # Convert to numpy arrays for vectorized assignment
                    mask_y, mask_x = zip(*mask)
                    mask_y, mask_x = np.array(mask_y), np.array(mask_x)
                    H_uv[mask_y, mask_x] = 0

            else:
                # Calculate the distances from notch centers
                D1 = np.sqrt((u - center[1])**2 + (v - center[0])**2)
                D2 = np.sqrt((u - (P - center[1]))**2 + (v - (Q - center[0]))**2)

                if filter_type == -1:  # Ideal filter
                    H_uv[D1 <= cutoff] = 0
                    H_uv[D2 <= cutoff] = 0

                elif filter_type == 0:  # Gaussian filter
                    H_k = (1 - np.exp(-D1**2 / (2 * cutoff**2))) * (1 - np.exp(-D2**2 / (2 * cutoff**2)))
                    H_uv *= H_k

                else:  # Butterworth filter (ft now acts as the parameter "n")
                    n = filter_type
                    H_k = (1 - 1 / (1 + (D1 / cutoff)**(2 * n))) * (1 - 1 / (1 + (D2 / cutoff)**(2 * n)))
                    H_uv *= H_k

    # Norch Band Resist filter to Norch Band Pass filter
    H_uv = np.ones_like(H_uv) - H_uv

    # Display combined notch filter
    axs[0, 3].imshow(H_uv, cmap="gray")
    axs[0, 3].set_title("Notch Filter")

    # Step 4: Apply the filter in the frequency domain to extract noise spectrum
    noise_uv = H_uv * G_uv

    # Display filtered spectrum
    axs[1, 0].imshow(np.log(1 + np.abs(noise_uv)), cmap="gray")
    axs[1, 0].set_title("Noise Spectrum")

    # Step 5: Compute the inverse Fourier Transform and remove centering shift
    noise = np.fft.ifft2(noise_uv)
    noise = np.real(noise) * np.fromfunction(lambda x, y: (-1)**(x + y), (P, Q))
    noise = noise[:M, :N]

    # Display noise pattern
    axs[1, 1].imshow(np.clip(noise, 0, 255).astype(np.uint8), cmap="gray")
    axs[1, 1].set_title("Noise pattern")

    # Step 6: Compute the weight function
    w = compute_weight_function(image, noise, window_size=window_size)
    weighted_noise = w * noise

    # Display weighted noise pattern
    axs[1, 2].imshow(np.clip(weighted_noise, 0, 255).astype(np.uint8), cmap="gray")
    axs[1, 2].set_title("Weighted Noise pattern")

    # Step 7: The final image
    f = np.clip(image - weighted_noise, 0, 255).astype(np.uint8)

    # Display the final image
    axs[1, 3].imshow(f, cmap="gray")
    axs[1, 3].set_title("The final image") 

    # Save the figure containing all the images
    save_path = f"frequency_domain_notch_{image_name}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    # Save the final filtered image
    final_image_path = f"final_filtered_image_notch_{image_name}.tif"
    Image.fromarray(f).save(final_image_path, format="TIFF")

    return f

def compute_weight_function(g, eta, window_size=(5, 5)):
    """
    Compute the weight function w(x, y) that minimizes the variance of the filtered image.

    The weight function is calculated by analyzing the correlation between the noisy image (g) and the noise pattern (eta) 
    within a local neighborhood of each pixel.

    :param g: The degraded image g(x, y), which contains both the original image and noise.
    :param eta: The additive noise pattern eta(x, y), which has been obtained after filtering.
    :param window_size: The height and width of the local neighborhood (should be two odd integers).
    :return: The weight function w(x, y), which will be used to filter the image and minimize the noise.
    """
    # Create an empty array for the weight function w
    w = np.ones_like(g).astype(np.float64)
    M, N = g.shape
    radius_x = window_size[0] // 2
    radius_y = window_size[1] // 2

    # Iterate over each pixel in the image
    for x in range(radius_x, M-radius_x):
        for y in range(radius_y, N-radius_y):

            g_window = g[x-radius_x:x+radius_x+1, y-radius_y:y+radius_y+1]
            eta_window = eta[x-radius_x:x+radius_x+1, y-radius_y:y+radius_y+1]
            
            # Calculate the means
            bar_g = np.mean(g_window)
            bar_eta = np.mean(eta_window)
            bar_eta2 = np.mean(eta_window * eta_window)
            bar_g_eta = np.mean(g_window * eta_window)
            w[x, y] = (bar_g_eta - bar_g * bar_eta) / (bar_eta2 - bar_eta * bar_eta + 1e-3)

    return w

# Define filter type mappings
filter_type_dict = {
    "line filter": -2,
    "ideal filter": -1,
    "Gaussian filter": 0,
    "Butterworth filter (n=1)": 1,
    "Butterworth filter (n=2)": 2,
    "Butterworth filter (n=3)": 3,
    "Butterworth filter (n=4)": 4,
    "Butterworth filter (n=5)": 5,
    # Add more Butterworth filter orders if needed
}

if __name__ == "__main__":
    # Initialize the option to select which image and filter settings to load
    option = 1

    if option == 1:
        # Load 'shepp_logan.png' and set parameters specific to this image
        image_path = 'shepp_logan.png'
        image_name = 'shepp_logan'
        min_distance = 75  # Minimum distance between detected centers

        # Define circular filter centers across the full x and y sets
        centers = [[1119, 0, 624, 1],
                   [1119, 804, 1428, 1]]
        cutoffs = [313] * len(centers)  # Set 30-pixel cutoff for all centers_2 entries
        filter_type = filter_type_dict.get("line filter", -2)

        # Group centers, cutoffs, and filter types for use in frequency domain function
        centers = [centers]
        cutoffs = [cutoffs]
        filter_types = [filter_type]

        # Set window size
        window_size = (9, 9)

    else:
        # Load 'DIP Fig 04.65(a)(cassini).tif' and set line filter settings
        image_path = 'DIP Fig 05.20(a)(NASA_Mariner6_Mars).tif'
        image_name = 'DIP Fig 05.20(a)(NASA_Mariner6_Mars)'
        min_distance = 25  # Set minimum distance for center detection

        # Define two vertical line filters with specific x and y ranges
        centers = [[[453, 449], 
                    [336, 446],
                    [250, 415],
                    [289, 254],
                    [338, 256],
                    [330, 233],
                    [377, 234],
                    [383, 257],
                    [427, 236],
                    [420, 214],
                    [372, 212],
                    [325, 211],
                    [276, 208],
                    [461, 17 ],
                    [291, 336],
                    [290, 294],
                    [276, 167],
                    [270, 146],
                    [320, 200]]]
        cutoffs = [[5] * len(centers[0])]
        filter_types = [filter_type_dict.get("Butterworth filter (n=4)", 4)]

        # Set window size
        window_size = (7, 7)

    # Load and convert the image to grayscale format
    img = Image.open(image_path).convert('L')
    image_array = np.array(img)  # Convert image to a numpy array for processing

    # Run the center detection function in frequency domain with specified distance
    # frequency_domain_center_detection(image_array, image_name, min_distance)

    # Apply the frequency domain notch filtering with specified centers and cutoffs
    filter_image = frequency_domain_notch(image_array, centers, cutoffs, image_name, filter_types, window_size)