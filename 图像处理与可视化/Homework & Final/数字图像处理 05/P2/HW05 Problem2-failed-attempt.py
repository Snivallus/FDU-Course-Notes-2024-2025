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

    # Step 1: Apply Fourier transform centering by multiplying with (-1)^(x+y)
    # This shifts the zero-frequency component to the center
    centered_image = image * np.fromfunction(lambda x, y: (-1)**(x + y), (M, N))
    axs[0, 1].imshow(np.clip(centered_image, 0, 255).astype(np.uint8), cmap="gray")
    axs[0, 1].set_title(f"Centered Image - Center: ({N // 2},{M // 2})")

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
        distance_from_center = np.sqrt((x - M // 2)**2 + (y - N // 2)**2)
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
    centered_image = image * np.fromfunction(lambda x, y: (-1)**(x + y), (M, N))
    
    # Display centered image
    axs[0, 1].imshow(np.clip(centered_image, 0, 255).astype(np.uint8), cmap="gray")
    axs[0, 1].set_title(f"Centered Image - Center: ({N // 2},{M // 2})")

    # Step 2: Compute the 2D Fourier Transform of the centered image
    G_uv = np.fft.fft2(centered_image)
    
    # Display original spectrum
    axs[0, 2].imshow(np.log(1 + np.abs(G_uv)), cmap="gray")
    axs[0, 2].set_title("Original Spectrum")

    # Step 3: Create the notch filter with multiple notch centers
    u, v = np.meshgrid(np.arange(0, M), np.arange(0, N), indexing='ij')

    # Initialize the combined notch filter with ones (multiplicative identity for filters)
    H_uv = np.ones((M, N))

    # Loop over each "filter_type" to construct the filter
    for i, filter_type in enumerate(filter_types):
        for center, cutoff in zip(centers[i], cutoffs[i]):
            if filter_type == -2:  # Line filter (horizontal or vertical)
                if center[3] == 0:  # Horizontal line
                    y_val = center[0]  # Fixed y-value for the horizontal line
                    x_start, x_end = center[1], center[2]  # x-range for the horizontal line
                    mask_x_range = np.arange(x_start, x_end + 1)
                    mask_y_range = np.arange(np.clip(y_val - cutoff, 0, M), np.clip(y_val + cutoff, 0, M) + 1)
                    mask = list(itertools.product(mask_y_range, mask_x_range))
                    mask += [(M - i - 1, N - j - 1) for i, j in mask]

                    # Convert to numpy arrays for vectorized assignment
                    mask_y, mask_x = zip(*mask)
                    mask_y, mask_x = np.array(mask_y), np.array(mask_x)
                    H_uv[mask_y, mask_x] = 0

                else:  # Vertical line
                    x_val = center[0]  # Fixed x-value for the vertical line
                    y_start, y_end = center[1], center[2]  # y-range for the vertical line
                    mask_y_range = np.arange(y_start, y_end + 1)
                    mask_x_range = np.arange(np.clip(x_val - cutoff, 0, N), np.clip(x_val + cutoff, 0, N) + 1)
                    mask = list(itertools.product(mask_y_range, mask_x_range))
                    mask += [(M - i - 1, N - j - 1) for i, j in mask]

                    # Convert to numpy arrays for vectorized assignment
                    mask_y, mask_x = zip(*mask)
                    mask_y, mask_x = np.array(mask_y), np.array(mask_x)
                    H_uv[mask_y, mask_x] = 0

            else:
                # Calculate the distances from notch centers
                D1 = np.sqrt((u - center[1])**2 + (v - center[0])**2)
                D2 = np.sqrt((u - (M - center[1]))**2 + (v - (N - center[0]))**2)

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
    noise = np.real(noise) * np.fromfunction(lambda x, y: (-1)**(x + y), (M, N))

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
    # Compute local histograms for g and eta
    eta_histograms = compute_local_histograms(eta, window_size=window_size)
    g_histograms = compute_local_histograms(g, window_size=window_size)
    
    # Create an empty array for the weight function w
    w = np.ones_like(g).astype(np.float64)
    M, N = g.shape
    mn = window_size[0] * window_size[1]

    # Iterate over each pixel in the image
    for x in range(M):
        for y in range(N):
            # Extract the local histograms for g and eta at (x, y)
            g_local_hist = g_histograms[x, y]
            eta_local_hist = eta_histograms[x, y]

            # Get the non-zero frequencies (values)
            g_nonzero_indices = np.nonzero(g_local_hist)[0]
            eta_nonzero_indices = np.nonzero(eta_local_hist)[0]
            g_nonzero = g_local_hist[g_nonzero_indices]  # Frequencies for the corresponding gray levels
            eta_nonzero = eta_local_hist[eta_nonzero_indices]  # Frequencies for the corresponding noise levels
            
            # Calculate the weighted means
            bar_g = np.dot(g_nonzero_indices, g_nonzero) / mn
            bar_eta = np.dot(eta_nonzero_indices, eta_nonzero) / mn

            # Calculate the weighted means for eta^2
            bar_eta2 = np.dot(eta_nonzero_indices ** 2, eta_nonzero) / mn

            # Calculate the weighted covariance (g_eta)
            bar_g_eta = 0
            for g_val, eta_val in zip(g_nonzero_indices, eta_nonzero_indices):
                bar_g_eta += g_val * eta_val * g_local_hist[g_val] * eta_local_hist[eta_val]
            
            bar_g_eta /= mn
            
            # Compute the weight function w(x, y)
            denominator = bar_eta2 - bar_eta ** 2
            if abs(denominator) > 1e-6:  # Prevent division by zero
                w[x, y] = (bar_g_eta - bar_g * bar_eta) / denominator
            else:
                w[x, y] = 1  # If the denominator is zero, set w to zero (or use some other default value)

    return w

def compute_histogram(image, num_bins=256):
    """
    计算图像的灰度直方图。
    
    :param image: 灰度图像的 numpy 数组
    :param num_bins: 直方图的 bins 数量
    :return: 直方图和 bins 边缘
    """
    histogram, bin_edges = np.histogram(image.ravel(), bins=num_bins, range=[0, num_bins])
    return histogram, bin_edges

def update_histogram(old_hist, new_col=None, remove_col=None, num_bins=256):
    """
    使用增量更新直方图.
    :param old_hist: 当前的直方图.
    :param new_col: 要加入的新的列像素 (可以为 None).
    :param remove_col: 要移除的列像素 (可以为 None).
    :param num_bins: 直方图的 bins 数量.
    :return: 更新后的直方图.
    """
    if new_col is None:
        return old_hist - np.bincount(remove_col, minlength=num_bins)
    elif remove_col is None:
        return old_hist + np.bincount(new_col, minlength=num_bins)
    else:
        return old_hist - np.bincount(remove_col, minlength=num_bins) + np.bincount(new_col, minlength=num_bins)

def compute_local_histograms(image, window_size=(9, 9), num_bins=256):
    """
    计算图像的局部直方图.
    :param image: 输入的灰度图像.
    :param window_size: 邻域窗口的尺寸 (height, width).
    :param num_bins: 直方图的 bins 数量.
    :return: 所有局部直方图的列表.
    """
    h, w = image.shape
    win_h, win_w = window_size
    half_win_h = win_h  // 2  # 使用整数除法，得到窗口半径
    half_win_w = win_w  // 2
    
    # 初始化局部直方图列表
    local_histograms = np.zeros((h, w, num_bins), dtype=np.uint8)

    # 移动完整窗口
    for i in range(h):
        if i == 0: 
            # 计算第一个完整窗口的直方图
            local_histograms[0, 0, :], _ = compute_histogram(image[0:half_win_h+1,0:half_win_w+1], num_bins=num_bins)
        else: 
        # 更新直方图 (垂直移动)
            if i <= half_win_h: # 首
                new_row = image[i+half_win_h, 0:half_win_w+1]
                remove_row = None
            elif i >= h - half_win_h: # 尾
                new_row = None
                remove_row = image[i-half_win_h-1, 0:half_win_w+1]
            else: # 中间部分
                new_row = image[i+half_win_h, 0:half_win_w+1]
                remove_row = image[i-half_win_h-1, 0:half_win_w+1]
            # 更新直方图 (垂直移动)
            local_histograms[i, 0, :] = update_histogram(local_histograms[i-1, 0, :], new_row, remove_row, num_bins=num_bins)

        i_safe_lower = max(i-half_win_h,0)
        i_safe_upper = min(i+half_win_h+1,h)

        for j in range(1, w):
            # 更新直方图 (水平移动)
            if j <= half_win_w: # 首
                new_col = image[i_safe_lower:i_safe_upper, j+half_win_w]
                remove_col = None
            elif j >= w - half_win_w: # 尾
                new_col = None
                remove_col = image[i_safe_lower:i_safe_upper, j-half_win_w-1]
            else: # 中间部分
                new_col = image[i_safe_lower:i_safe_upper, j+half_win_w]
                remove_col = image[i_safe_lower:i_safe_upper, j-half_win_w-1]
            
            # 更新直方图 (水平移动)
            local_histograms[i, j, :] = update_histogram(local_histograms[i, j-1, :], new_col, remove_col, num_bins=num_bins)

    return local_histograms

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
    option = 4

    if option == 1:
        # Load 'shepp_logan.png' and set parameters specific to this image
        image_path = 'shepp_logan.png'
        image_name = 'shepp_logan'
        min_distance = 75  # Minimum distance between detected centers

        # Define x-coordinates for line filters
        centers_x = [413, 497, 531, 581, 612, 698]
        
        # Define two sets of y-coordinates for line filters
        centers_y1 = [17, 101, 132, 186, 218, 298]
        centers_y2 = [416, 498, 533, 615, 700]
        
        # Concatenate y-coordinates for general use in 2D product combinations
        centers_y = centers_y1 + centers_y2

        # Define horizontal line filters in each y-set with x bounds
        centers_1 = list(itertools.product(centers_y1, [[centers_x[0], centers_x[-1], 0]]))
        centers_1.extend(itertools.product(centers_y2, [[centers_x[0], centers_x[-1], 0]]))

        # Define vertical line filters in each x-set with y bounds
        centers_1.extend(itertools.product(centers_x, [[centers_y1[0], centers_y1[-1], 1]]))
        centers_1.extend(itertools.product(centers_x, [[centers_y2[0], centers_y2[-1], 1]]))
        
        # Flatten tuple list for centers to format (y, x_start, x_end, orientation)
        centers_1 = [(x, *y) for x, y in centers_1]
        cutoffs_1 = [10] * len(centers_1)  # Set a 20-pixel cutoff for all centers_1 entries
        
        # Retrieve the filter type code for line filter from dictionary
        filter_type_1 = filter_type_dict.get("line filter", -2)

        # Define circular filter centers across the full x and y sets
        centers_2 = list(itertools.product(centers_x, centers_y))
        cutoffs_2 = [15] * len(centers_2)  # Set 30-pixel cutoff for all centers_2 entries
        
        # Retrieve the filter type code for ideal filter from dictionary
        filter_type_2 = filter_type_dict.get("ideal filter", -1)

        # Group centers, cutoffs, and filter types for use in frequency domain function
        centers = [centers_1, centers_2]
        cutoffs = [cutoffs_1, cutoffs_2]
        filter_types = [filter_type_1, filter_type_2]

        # Set window size
        window_size = (9, 9)

    elif option == 2:
        # Load 'DIP Fig 04.64(a)(car_75DPI_Moire).tif' and set filter settings
        image_path = 'DIP Fig 04.64(a)(car_75DPI_Moire).tif'
        image_name = 'DIP Fig 04.64(a)(car_75DPI_Moire)'
        min_distance = 25  # Set minimum distance between detected centers

        # Define specific coordinates and cutoff values for Butterworth filters
        centers = [[[111, 81], 
                    [113, 161], 
                    [111, 39], 
                    [113, 202]]]
        cutoffs = [[9, 9, 9, 9]]  # Cutoffs correspond to each center in centers

        # Use dictionary to get the filter type code for Butterworth (defaulting to 0)
        filter_types = [filter_type_dict.get("Butterworth filter (n=4)", 4)]

        # Set window size
        window_size = (5, 5)

    elif option == 3:
        # Load 'DIP Fig 04.65(a)(cassini).tif' and set line filter settings
        image_path = 'DIP Fig 04.65(a)(cassini).tif'
        image_name = 'DIP Fig 04.65(a)(cassini)'
        min_distance = 25  # Set minimum distance for center detection

        # Define two vertical line filters with specific x and y ranges
        centers = [[[337, 0, 330, 1], [337, 344, 674, 1]]]
        cutoffs = [[1] * len(centers)]  # Set 10-pixel cutoff for each center
        
        # Retrieve the filter type code for line filter from dictionary
        filter_types = [filter_type_dict.get("line filter", -2)]

        # Set window size
        window_size = (7, 7)

    else:
        # Load 'DIP Fig 04.65(a)(cassini).tif' and set line filter settings
        image_path = 'DIP Fig 05.20(a)(NASA_Mariner6_Mars).tif'
        image_name = 'DIP Fig 05.20(a)(NASA_Mariner6_Mars)'
        min_distance = 25  # Set minimum distance for center detection

        # Define two vertical line filters with specific x and y ranges
        centers = [[[337, 0, 330, 1], [337, 344, 674, 1]]]
        cutoffs = [[1] * len(centers)]  # Set 10-pixel cutoff for each center
        
        # Retrieve the filter type code for line filter from dictionary
        filter_types = [filter_type_dict.get("line filter", -2)]

        # Set window size
        window_size = (7, 7)

    # Load and convert the image to grayscale format
    img = Image.open(image_path).convert('L')
    image_array = np.array(img)  # Convert image to a numpy array for processing

    # Run the center detection function in frequency domain with specified distance
    # frequency_domain_center_detection(image_array, image_name, min_distance)

    # Apply the frequency domain notch filtering with specified centers and cutoffs
    filter_image = frequency_domain_notch(image_array, centers, cutoffs, image_name, filter_types, window_size)