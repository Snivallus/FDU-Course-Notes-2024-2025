import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_affine_transformation(transformation_types, params=None):
    """
    Apply a series of affine transformations (scaling, rotation, translation, shear_vertical, shear_horizontal)
    by multiplying their corresponding transformation matrices together.

    Parameters:
    - transformation_types: A list of transformation types (e.g., ['scaling', 'rotation', 'translation']).
    - params: A list of parameter dictionaries, each corresponding to a transformation in `transformation_types`.
        Example: 
        [
            {'cx': 2, 'cy': 2},       # for scaling
            {'theta': 45},            # for rotation
            {'tx': 10, 'ty': 20},     # for translation
            {'sv': 0.5},              # for shear_vertical
            {'sh': 0.5}               # for shear_horizontal
        ]

    Returns:
    - A: The combined affine transformation matrix as a result of all the transformations.
    """

    # Initialize the result matrix as the identity matrix
    A_total = np.eye(3)

    # Process each transformation type and apply the corresponding transformation matrix
    for transformation_type, param in zip(transformation_types, params):
        if transformation_type == 'scaling':
            cx = param.get('cx', 1)
            cy = param.get('cy', 1)
            A = np.array([
                [cx, 0, 0],
                [0, cy, 0],
                [0, 0, 1]
            ])
        elif transformation_type == 'rotation':
            theta = np.radians(param.get('theta', 0))  # Convert to radians
            A = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        elif transformation_type == 'translation':
            tx = param.get('tx', 0)
            ty = param.get('ty', 0)
            A = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])
        elif transformation_type == 'shear_vertical':
            sv = param.get('sv', 0)
            A = np.array([
                [1, sv, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        elif transformation_type == 'shear_horizontal':
            sh = param.get('sh', 0)
            A = np.array([
                [1, 0, 0],
                [sh, 1, 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError(f"Unknown transformation type: {transformation_type}")

        # Apply the transformation matrix by multiplying with the cumulative result matrix
        A_total = np.dot(A_total, A)

    return A_total

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

def inversely_transform_image_local(image, region, transformation, output_shape=None, p=1.5, interpolation_method="bilinear"):

    if output_shape is None:
        output_shape = image.shape

    height, width = output_shape
    # the boundary of image
    source_coords = np.zeros((height, width, 2))
    for y in range(height):
        source_coords[y, 0, :] = [y, 0]
        source_coords[y, width-1, :] = [y, width-1]
    for x in range(width):
        source_coords[0, x, :] = [0, x]
        source_coords[height-1, x, :] = [height-1, x]

    A_inv = np.linalg.inv(transformation)
    y, x = np.meshgrid(np.arange(1, height-1), np.arange(1, width-1), indexing="ij")
    region_coords = np.stack((y.ravel(), x.ravel(), np.ones_like(x.ravel())), axis=-1)
    region_coords = region_coords @ A_inv.T
    region_coords = region_coords[..., :2]
    region_coords = region_coords.reshape((height-2, width-2, 2))

    y_min, y_max = region[0]
    x_min, x_max = region[1]
    mask = (
        (region_coords[..., 0] >= x_min) & (region_coords[..., 0] <= x_max) &
        (region_coords[..., 1] >= y_min) & (region_coords[..., 1] <= y_max)
    )
    valid_indices = np.argwhere(mask)
    for idx in valid_indices:
        i, j = idx
        source_coords[i+1, j+1, :] = region_coords[i, j, :]

    invalid_indices = np.argwhere(~mask)
    for idx in invalid_indices:
        i, j = idx
        i = i + 1
        j = j + 1
        top_distance = i
        bottom_distance = height - 1 - i
        left_distance = j
        right_distance = width - 1 - j
        x_project = np.clip(j, x_min + 1, x_max + 1)
        y_project = np.clip(i, y_min + 1, y_max + 1)
        region_distance = np.linalg.norm(np.array([
            np.maximum(1, abs(y_project - i)),
            np.maximum(1, abs(x_project - j))
        ]))
        distance = np.array([top_distance, bottom_distance, left_distance, right_distance, region_distance])
        weight = 1 / (distance ** p)
        weight = weight / np.sum(weight)
        source_coords[i, j] = (weight[0] + weight[1] + weight[2] + weight[3]) * np.array([i, j]
                              ) + weight[4] * region_coords[i-1, j-1]

    # Interpolate pixel values from the source image
    transformed_image = interpolate(image, source_coords, method=interpolation_method)

    return transformed_image

if __name__ == "__main__":
    image_path = 'grid.png'
    image_name = 'grid'
    image = Image.open(image_path).convert('L')  # Convert the image to grayscale ('L' mode)
    image = np.array(image)  # Convert the grayscale image to a numpy array  

    # Affine transformation type ('scaling', 'rotation', 'translation', 'shear_vertical', 'shear_horizontal')
    # Example: 
        # [
        #     {'cx': 2, 'cy': 2},       # for scaling
        #     {'theta': 45},            # for rotation
        #     {'tx': 10, 'ty': 20},     # for translation
        #     {'sv': 0.5},              # for shear_vertical
        #     {'sh': 0.5}               # for shear_horizontal
        # ]
    height, width = image.shape
    transformation_types = ['translation', 'rotation', 'translation']
    params = [{'tx': 0.5 * height, 'ty': 0.5 * width}, 
              {'theta': 30}, 
              {'tx': -0.5 * height, 'ty': -0.5 * width}]
    # Get the combined affine transformation matrix
    A = apply_affine_transformation(transformation_types, params)
    region = [[358, 665], [358, 665]]
    interpolation_method = "bicubic"

    # Output shape is the same as the input image
    output_shape = np.array(image.shape)

    # Apply the inverse transformation to the image
    transformed_image = inversely_transform_image_local(image, region=region, 
                                                        transformation=A, output_shape=output_shape, 
                                                        p=1.5, interpolation_method=interpolation_method)

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