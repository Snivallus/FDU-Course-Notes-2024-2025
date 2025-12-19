import numpy as np

def apply_affine_transformation(transformation_type, params=None):
    """
    Calculate an affine transformation of a specific type and coefficients

    Parameters:
    - transformation_type: Type of transformation ('scaling', 'rotation', 'translation', 'shear_vertical', 'shear_horizontal').
    - params: Additional parameters required for specific transformations:
        * For 'scaling': {'cx': float, 'cy': float}
        * For 'rotation': {'theta': float (in degrees)}
        * For 'translation': {'tx': float, 'ty': float}
        * For 'shear_vertical': {'sv': float}
        * For 'shear_horizontal': {'sh': float}

    Returns:
    - A: transform matrix
    """
    # Define the transformation matrix based on the type
    if transformation_type == 'scaling':
        cx = params.get('cx', 1)
        cy = params.get('cy', 1)
        A = np.array([
            [cx, 0, 0],
            [0, cy, 0],
            [0, 0, 1]
        ])
    elif transformation_type == 'rotation':
        theta = np.radians(params.get('theta', 0))  # Convert to radians
        A = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    elif transformation_type == 'translation':
        tx = params.get('tx', 0)
        ty = params.get('ty', 0)
        A = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])
    elif transformation_type == 'shear_vertical':
        sv = params.get('sv', 0)
        A = np.array([
            [1, sv, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    elif transformation_type == 'shear_horizontal':
        sh = params.get('sh', 0)
        A = np.array([
            [1, 0, 0],
            [sh, 1, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unknown transformation type: {transformation_type}")

    return A

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

def inversely_transform_image(image, transformation_type, params=None, output_shape=None, interpolation_method="bilinear"):
    """
    Inversely transform an image using a specified affine transformation type and coefficients.

    Parameters:
        image (np.ndarray): Input image as a 2D array.
        transformation_type (str): Affine transformation type ('scaling', 'rotation', 'translation', 'shear_vertical', 'shear_horizontal').
        params (dict): Parameters for the affine transformation.
        output_shape (tuple): Shape of the output image (height, width). If None, use the input image's shape.
        interpolation_method (str): Interpolation method for pixel values ('nearest', 'bilinear', 'bicubic').

    Returns:
        np.ndarray: Transformed image with the specified output shape.
    """
    if output_shape is None:
        output_shape = image.shape

    # Create the affine transformation matrix
    A = apply_affine_transformation(transformation_type, params)

    # Compute the inverse of the transformation matrix
    A_inv = np.linalg.inv(A)

    # Generate the coordinate grid for the target image
    height, width = output_shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    target_coords = np.stack((y.ravel(), x.ravel(), np.ones_like(x.ravel())), axis=-1)
    print("target_coords")
    print(target_coords)

    # Apply the inverse transformation to map target coordinates back to source coordinates
    source_coords = target_coords @ A_inv.T
    source_coords = source_coords[..., :2] # Normalize homogeneous coordinates
    print("source_coords")
    print(source_coords)

    # Reshape the source coordinates to match the target shape
    source_indices = source_coords.reshape((height, width, 2))
    print("indices")
    print(source_indices[0, ...])

    # Interpolate pixel values from the source image
    transformed_image = interpolate(image, source_indices, method=interpolation_method)

    return transformed_image

if __name__ == "__main__":
    # Example grayscale image
    image = np.array([
        [100, 150, 200],
        [50, 100, 150],
        [0, 50, 100],
        [0, 25, 50]
    ])

    # Scaling transformation (inverse of scaling down by 2x in x and y)
    params = {'cx': 2, 'cy': 2}  # Scaling factor to reverse
    transformed_image = inversely_transform_image(
        image, transformation_type='scaling', params=params, output_shape=(4, 3), interpolation_method="bilinear"
    )

    print("Original Image:")
    print(image)
    print("Transformed Image:")
    print(transformed_image)