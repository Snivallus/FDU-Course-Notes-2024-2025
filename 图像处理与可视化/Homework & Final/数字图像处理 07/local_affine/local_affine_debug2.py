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

def calculate_weights(image_shape, region, p=1.5):
    """
    Calculate weights for each pixel in the image based on distances to the region and the image boundaries.

    Parameters:
    - image_shape (tuple): Shape of the image (height, width).
    - region (tuple): Region specified as ((y_min, y_max), (x_min, x_max)).
    - p (float): Power parameter for weight calculation.

    Returns:
    - tuple: A 2D array of shape (height, width, 5), where the last dimension represents weights,
             and a 2D array of shape (height * width, 2) representing source coordinates.
    """
    height, width = image_shape
    y_min, y_max = region[0]
    x_min, x_max = region[1]

    # Create coordinate grids for the entire image
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Compute Chebyshev distances to the region
    region_distance = np.maximum(
        np.maximum(y_min - y_coords, 0) + np.maximum(y_coords - y_max, 0),
        np.maximum(x_min - x_coords, 0) + np.maximum(x_coords - x_max, 0)
    )

    # Compute distances to the image boundaries
    top_distance = y_coords
    bottom_distance = height - 1 - y_coords
    left_distance = x_coords
    right_distance = width - 1 - x_coords

    # Combine all distances into a single array
    distances = np.stack(
        (top_distance, bottom_distance, left_distance, right_distance, region_distance), axis=-1
    ).astype(np.float64)
    distances[distances == 0] = np.inf

    # Calculate weights using the inverse distance formula
    weights = 1 / (distances ** p)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if np.any(distances[i, j] == np.inf):
                inf_index = np.argmax(distances[i, j] == np.inf)  # Get the index where distance is np.inf
                weights[i, j] = np.zeros(weights.shape[2])  # Set all weights to 0
                weights[i, j, inf_index] = 1  # Set the weight corresponding to np.inf to 1

    # Normalize weights so their sum equals 1 at each pixel
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights /= row_sums  # Normalize the weights

    # Initialize the source coordinates array
    source_coords = np.zeros((height, width, 2))
    source_coords[..., 0] += weights[..., 0] * x_coords + weights[..., 1] * x_coords + weights[..., 3] * (width - 1)
    source_coords[..., 1] += weights[..., 2] * y_coords + weights[..., 3] * y_coords + weights[..., 1] * (height - 1)

    return weights, source_coords

def inversely_transform_image_local(image, region, transformation, output_shape=None, p=1.5, interpolation_method="bilinear"):
    """
    Inversely transform an image using local affine transformations.
    
    Parameters:
    - image (np.ndarray): Input image as a 2D array.
    - region: the control region.
    - transformation : Affine transformations (3x3 matrices) for the region.
    - output_shape (tuple): Shape of the output image (height, width). If None, use the input image's shape.
    - p (float): Power parameter for weight calculation.
    - interpolation_method (str): Interpolation method for pixel values ('nearest', 'bilinear', 'bicubic').
    
    Returns:
    - np.ndarray: Transformed image with the specified output shape.
    """
    if output_shape is None:
        output_shape = image.shape

    # Calculate the weights for each target coordinate based on the distances to the control points
    y_min, y_max = region[0]
    x_min, x_max = region[1]
    region_height = y_max - y_min + 1
    region_width = x_max - x_min + 1
    y, x = np.meshgrid(np.arange(y_min, y_max+1), np.arange(x_min, x_max+1), indexing="ij")
    target_coords = np.stack((y.ravel(), x.ravel(), np.ones_like(x.ravel())), axis=-1)  # Homogeneous coordinates

    # Apply transformations to the region
    A_inv = np.linalg.inv(transformation)  # Compute the inverse of the transformation matrix
    transformed_coords = target_coords @ A_inv.T
    transformed_coords = transformed_coords[..., :2]
    transformed_coords = transformed_coords.reshape((region_height, region_width, 2))

    # height, width = output_shape
    # source_coords = np.zeros((height, width, 2))
    # source_coords[y_min:y_max+1, x_min:x_max+1] = transformed_coords

    # Calculate weights for pixels outside the region and not on the image boundary
    weights, source_coords = calculate_weights(image_shape=output_shape, region=region, p=p)
    source_coords = source_coords.reshape((height, width, 2))

    # use the fifth column of weights to sum the projection of (x,y) in region into source_coords
    for i in range(height):
        for j in range(width):
            project_x = int(np.clip(j - x_min, 0, region_width - 1))
            project_y = int(np.clip(i - y_min, 0, region_height - 1))
            source_coords[i,j] += weights[i, j, 4] * transformed_coords[project_x, project_y]

    source_coords[y_min:y_max+1, x_min:x_max+1] = transformed_coords

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
              {'theta': 45}, 
              {'tx': -0.5 * height, 'ty': -0.5 * width}]
    # Get the combined affine transformation matrix
    A = apply_affine_transformation(transformation_types, params)
    # region = [[356, 665], [356, 665]]
    region = [[256, 765], [256, 765]]
    interpolation_method="bicubic"

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