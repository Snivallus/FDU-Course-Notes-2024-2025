import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def frequency_domain_sharpen(image, image_name):

    # Plotting setup
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs[0, 0].imshow(image, cmap="gray")
    axs[0, 0].set_title("Original Image")

    # Step 1: Zero-padding the image to size P x Q where P = 2M, Q = 2N
    M, N = image.shape
    P, Q = 2 * M, 2 * N
    image_normalized = (image.astype(np.float64) - np.min(image)) / (np.max(image) - np.min(image))
    padded_image = np.zeros((P, Q)).astype(np.float64)
    padded_image[:M, :N] = image_normalized

    # Display padded image
    axs[0, 1].imshow(np.clip(255 * padded_image, 0, 255).astype(np.uint8), cmap="gray")
    axs[0, 1].set_title("Padded Image")

    # Step 2: Multiply padded image by (-1)^(x+y) to center the Fourier transform
    centered_image = padded_image * np.fromfunction(lambda x, y: (-1)**(x + y), (P, Q))
    
    # Display centered image
    axs[0, 2].imshow(np.clip(255 * centered_image, 0, 255).astype(np.uint8), cmap="gray")
    axs[0, 2].set_title("Centered Image (Shifted)")

    # Step 3: Compute the 2D Fourier Transform of the centered image
    F_uv = np.fft.fft2(centered_image)
    
    # Display original spectrum
    axs[0, 3].imshow(np.log(1 + np.abs(F_uv)), cmap="gray")
    axs[0, 3].set_title("Original Spectrum")

    # Step 4: Create the Laplace filter
    u, v = np.meshgrid(np.arange(-P//2, P//2), np.arange(-Q//2, Q//2), indexing='ij')
    D = np.sqrt(u**2 + v**2)
    H_uv = -4 * np.pi**2 * D ** 2 # Laplace filter

    # Display Laplace filter
    axs[1, 0].imshow(np.log(1 + np.abs(H_uv)), cmap="gray")
    axs[1, 0].set_title("Laplace Filter")

    # Apply the filter in the frequency domain
    Laplace_uv = H_uv * F_uv

    # Display Laplace filter
    axs[1, 1].imshow(np.log(1 + np.abs(Laplace_uv)), cmap="gray")
    axs[1, 1].set_title("Laplace Filtered Spectrum")

    Laplace_filtered = np.fft.ifft2(Laplace_uv)
    Laplace_filtered = np.real(Laplace_filtered) * np.fromfunction(lambda x, y: (-1)**(x + y), (P, Q))
    Laplace_filtered = Laplace_filtered / np.max(np.abs(Laplace_filtered))
    Laplace_filtered = Laplace_filtered[:M, :N]

    # Display Laplace filter
    axs[1, 2].imshow(np.clip(255 * Laplace_filtered, 0, 255).astype(np.uint8), cmap="gray")
    axs[1, 2].set_title("Laplace Filtered Image")

    # Step 6: Extract the top-left M x N portion to obtain the final result
    g = image - Laplace_filtered * (np.max(image) - np.min(image))
    g = np.clip(g, 0, 255).astype(np.uint8)

    # Display final filtered image
    axs[1, 3].imshow(g, cmap="gray")
    axs[1, 3].set_title("Final Filtered Image")

    # Save the figure containing all the images
    save_path = f"frequency_domain_sharpen_{image_name}.png"
    plt.tight_layout()
    plt.savefig(save_path)

    # Save the final filtered image
    final_image_path = f"final_filtered_image_{image_name}.tif"
    Image.fromarray(g).save(final_image_path, format="TIFF")

    plt.show()

    return g

if __name__ == "__main__":
    # Load the image
    option = 2
    if option == 1:
        image_path = 'DIP Fig 04.58(a)(blurry_moon).tif'
        image_name = 'DIP Fig 04.58(a)(blurry_moon)'
    else:
        image_path = 'DIP Fig 04.27(a)(woman).tif'
        image_name = 'DIP Fig 04.27(a)(woman)'

    img = Image.open(image_path).convert('L')  # Convert it in grayscale
    image_array = np.array(img)

    # Apply the frequency domain lowpass filter with a chosen cutoff frequency
    filtered_image = frequency_domain_sharpen(image_array, image_name=image_name)
