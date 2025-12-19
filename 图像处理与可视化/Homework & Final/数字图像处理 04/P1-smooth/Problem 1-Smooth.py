import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def frequency_domain_smooth(image, cutoff, image_name):

    # Plotting setup
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs[0, 0].imshow(image, cmap="gray")
    axs[0, 0].set_title("Original Image")

    # Step 1: Zero-padding the image to size P x Q where P = 2M, Q = 2N
    M, N = image.shape
    P, Q = 2 * M, 2 * N
    padded_image = np.zeros((P, Q))
    padded_image[:M, :N] = image

    # Display padded image
    axs[0, 1].imshow(padded_image, cmap="gray")
    axs[0, 1].set_title("Padded Image")

    # Step 2: Multiply padded image by (-1)^(x+y) to center the Fourier transform
    centered_image = padded_image * np.fromfunction(lambda x, y: (-1)**(x + y), (P, Q))
    
    # Display centered image
    axs[0, 2].imshow(np.clip(centered_image, 0, 255).astype(np.uint8), cmap="gray")
    axs[0, 2].set_title("Centered Image (Shifted)")

    # Step 3: Compute the 2D Fourier Transform of the centered image
    F_uv = np.fft.fft2(centered_image)
    
    # Display original spectrum
    axs[0, 3].imshow(np.log(1 + np.abs(F_uv)), cmap="gray")
    axs[0, 3].set_title("Original Spectrum")

    # Step 4: Create the Gaussian low-pass filter with correct shape
    u, v = np.meshgrid(np.arange(-P//2, P//2), np.arange(-Q//2, Q//2), indexing='ij')
    D = np.sqrt(u**2 + v**2)
    H_uv = np.exp(-D**2 / (2 * cutoff**2))  # Gaussian low-pass filter

    # Display Gaussian filter
    axs[1, 0].imshow(H_uv, cmap="gray")
    axs[1, 0].set_title("Gaussian Low-Pass Filter")

    # Apply the filter in the frequency domain
    G_uv = H_uv * F_uv

    # Display the filter spectrum
    axs[1, 1].imshow(np.log(1 + np.abs(G_uv)), cmap="gray")
    axs[1, 1].set_title("Filtered Spectrum")

    # Step 5: Compute the inverse Fourier Transform and remove centering shift
    g_padded = np.fft.ifft2(G_uv)
    g_padded = np.real(g_padded) * np.fromfunction(lambda x, y: (-1)**(x + y), (P, Q))
    g_padded = np.clip(g_padded, 0, 255).astype(np.uint8)

    # Display g_padded
    axs[1, 2].imshow(g_padded, cmap="gray")
    axs[1, 2].set_title("Padded Filtered Image")

    # Step 6: Extract the top-left M x N portion to obtain the final result
    g = g_padded[:M, :N]

    # Display final filtered image
    axs[1, 3].imshow(g, cmap="gray")
    axs[1, 3].set_title("Final Filtered Image")

    # Save the figure containing all the images
    save_path = f"frequency_domain_smooth_cutoff_{cutoff}_{image_name}.png"
    plt.tight_layout()
    plt.savefig(save_path)

    # Save the final filtered image
    final_image_path = f"final_filtered_image_cutoff_{cutoff}_{image_name}.tif"
    Image.fromarray(g).save(final_image_path, format="TIFF")

    plt.show()

    return g

if __name__ == "__main__":
    # Load the image
    option = 1
    if option == 1:
        image_path = 'DIP Fig 04.29(a)(blown_ic).tif'
        cutoff = 45
        image_name = 'DIP Fig 04.29(a)(blown_ic)'
    elif option == 2:
        image_path = 'DIP Fig 04.41(a)(characters_test_pattern).tif'
        cutoff = 45
        image_name = 'DIP Fig 04.41(a)(characters_test_pattern)'
    elif option == 3:
        image_path = 'DIP Fig 04.58(a)(blurry_moon).tif'
        cutoff = 60
        image_name = 'DIP Fig 04.58(a)(blurry_moon)'
    else:
        image_path = 'DIP Fig 04.27(a)(woman).tif'
        cutoff = 60
        image_name = 'DIP Fig 04.27(a)(woman)'

    img = Image.open(image_path).convert('L')  # Convert it in grayscale
    image_array = np.array(img)

    # Apply the frequency domain lowpass filter with a chosen cutoff frequency
    filtered_image = frequency_domain_smooth(image_array, cutoff=cutoff, image_name=image_name)
