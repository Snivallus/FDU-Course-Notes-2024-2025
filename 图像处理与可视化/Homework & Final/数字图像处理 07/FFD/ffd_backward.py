import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg') # 可交互
import matplotlib.pyplot as plt
matplotlib.rc("font",family='YouYuan') # 显示中文字体
from PIL import Image

def interpolate(image, indices, method="bilinear"):
    """
    Interpolation function to calculate grayscale values at given indices in the image.

    Parameters:
        image (np.ndarray): Input image as a 2D array.
        indices (np.ndarray): Image-like 3D array where each element is the original (x, y) index.
        method (str): Interpolation method ("nearest", "bilinear", or "bicubic").

    Returns:
        np.ndarray: Interpolated grayscale values in the shape of the input indices array.
    """
    original_height, original_width = image.shape

    # Extract x and y coordinates from the indices array
    x, y = indices[..., 0].ravel(), indices[..., 1].ravel()

    if method == "nearest":
        x_nearest = np.clip(np.round(x).astype(int), 0, original_height - 1)
        y_nearest = np.clip(np.round(y).astype(int), 0, original_width - 1)
        result = image[x_nearest, y_nearest]

    elif method == "bilinear":
        x0 = np.clip(np.floor(x).astype(int), 0, original_height - 1)
        y0 = np.clip(np.floor(y).astype(int), 0, original_width - 1)
        x1 = np.clip(x0 + 1, 0, original_height - 1)
        y1 = np.clip(y0 + 1, 0, original_width - 1)

        dx = x - x0
        dy = y - y0

        I11 = image[x0, y0]
        I12 = image[x0, y1]
        I21 = image[x1, y0]
        I22 = image[x1, y1]

        result = (
            I11 * (1 - dx) * (1 - dy) +
            I12 * (1 - dx) * dy +
            I21 * dx * (1 - dy) +
            I22 * dx * dy
        )

    elif method == "bicubic":
        def cubic_weight(t):
            abs_t = np.abs(t)
            return np.where(
                abs_t <= 1,
                1.5 * abs_t**3 - 2.5 * abs_t**2 + 1,
                np.where(
                    (1 < abs_t) & (abs_t < 2),
                    -0.5 * abs_t**3 + 2.5 * abs_t**2 - 4 * abs_t + 2,
                    0
                )
            )

        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)

        result = np.zeros(len(x))
        for m in range(-1, 3):
            for n in range(-1, 3):
                x_idx = np.clip(x0 + m, 0, original_height - 1)
                y_idx = np.clip(y0 + n, 0, original_width - 1)
                weights = cubic_weight(x - (x0 + m)) * cubic_weight(y - (y0 + n))
                result += image[x_idx, y_idx] * weights

        result = np.clip(result, 0, 255)

    else:
        raise ValueError("Invalid interpolation method. Choose 'nearest', 'bilinear', or 'bicubic'.")

    # Reshape the result to match the shape of indices array
    return result.reshape(indices.shape[:2])

# Function for cubic B-spline basis
def beta(u):
    """
    Compute the B-spline basis functions for u, where u ∈ [0, 1).
    """
    beta = np.zeros(4)
    beta[0] = (1 - u)**3 / 6
    beta[1] = (3 * u**3 - 6 * u**2 + 4) / 6
    beta[2] = (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6
    beta[3] = u**3 / 6
    return beta

def inversely_transform_image_ffd(image, n_x, n_y, shift_dict, interpolation_method="bilinear"):
    """
    Perform inverse FFD to transform the image based on control point displacements.
    """
    # Initialize control points
    def return_no_shift():
        return np.zeros(2)
    
    # Use defaultdict to store the shifts of control points (default value is zero displacement)
    control_points = defaultdict(lambda: np.zeros(2))  # Default is [0, 0] for control points
    for (i, j), (delta_y, delta_x) in shift_dict.items():
        control_points[(i, j)] = np.array([delta_y, delta_x])  # Store displacement for control points
    
    # Get image dimensions
    height, width = image.shape
    ly = height / n_y  # Grid cell height
    lx = width / n_x   # Grid cell width
    
    # Prepare a mesh grid for source coordinates
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    source_coords = np.stack((y.ravel(), x.ravel()), axis=-1)
    source_coords = source_coords.reshape((height, width, 2)).astype(np.float64)

    for x_idx in range(width):  # Iterate over all pixels in the image
        ix = np.floor(x_idx / lx) - 1
        u = (x_idx / lx) - np.floor(x_idx / lx)
        beta_u = beta(u)

        for y_idx in range(height):
            iy = np.floor(y_idx / ly) - 1
            v = (y_idx / ly) - np.floor(y_idx / ly)
            beta_v = beta(v)

            # Sum over the neighboring control points using the B-spline weights
            for l in range(4):
                for k in range(4):
                    source_coords[y_idx, x_idx, :] += beta_u[l] * beta_v[k] * control_points[(ix + l, iy + k)]

    # Perform interpolation on the transformed coordinates
    transformed_image = interpolate(image, source_coords, method=interpolation_method)
    return transformed_image

if __name__ == "__main__":
    # Image loading and preprocessing
    image_path = 'grid.png'
    image_name = 'grid'
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = np.array(image)

    # Initialize the shift dictionary for control points
    n_x = 20
    n_y = 20
    height, width = image.shape
    lx = width / n_x
    ly = height / n_y

    # Define control points shift
    shift_dict = dict()
    shift_dict[(7, 7)] = np.array([15, 15])
    shift_dict[(7, 10)] = np.array([0, -15])
    shift_dict[(10, 7)] = np.array([-15, 0])
    shift_dict[(7, 13)] = np.array([-15, 15])
    shift_dict[(13, 7)] = np.array([15, -15])
    shift_dict[(13, 10)] = np.array([0, 15])
    shift_dict[(10, 13)] = np.array([15, 0])
    shift_dict[(13, 13)] = np.array([-15, -15])

    # Apply the inverse transformation to the image
    transformed_image = inversely_transform_image_ffd(image, n_x, n_y, shift_dict, interpolation_method="bilinear")

    # Plot the original and processed images side by side for comparison
    plt.figure(figsize=(12, 6))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image") 
    plt.axis('off')  # Hide axis ticks

    # Plot the transformed image
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image, cmap='gray')
    plt.title("Transformed Image")
    plt.axis('off')  # Hide axis ticks

    # Display the comparison
    plt.tight_layout()
    save_path = f"Comparision-{image_name}.png"
    plt.savefig(save_path)
    plt.show()