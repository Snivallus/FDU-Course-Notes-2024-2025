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

def calculate_weights(target_coords, control_points, p=1.5):
    """
    Calculate weights for each region based on distances in a vectorized manner.
    
    Parameters:
    - target_coords: A 2D array of shape (N, 2) representing the target coordinates (x, y).
    - control_points: List of control points, each as a tuple (center_x, center_y).
    - p: Power parameter for the distance-based weighting.
    
    Returns:
    - weights: A 2D array of shape (N, M), where N is the number of target coordinates 
               and M is the number of control points. Each element is the weight of a target 
               point with respect to a control point.
    """
    # Convert control_points to a NumPy array for easier manipulation
    control_points = np.array(control_points)

    # Calculate distances between each target coordinate and each control point
    # target_coords: (N, 2), control_points: (M, 2), result: (N, M)
    distances = np.linalg.norm(target_coords[:, np.newaxis] - control_points, axis=2, ord=np.inf)
    
    # TODO: If a line of distances (shape [1, M]) has zero value, set the zero value = np.inf
    distances[distances == 0] = np.inf  # Replace zero distances with np.inf
    
    # Calculate the weights using the inverse distance raised to the power of p
    weights = 1 / (distances ** p)
    
    # TODO: If a line of distances (shape [1, M]) has np.inf value, then set the weights into [0, 0, 0, ..., 1, 0, ..., 0]
    # where the only 1 is the position where np.inf is.
    # For each row in weights, check if there are any inf values and adjust accordingly.
    for i in range(weights.shape[0]):
        if np.any(distances[i] == np.inf):
            inf_index = np.argmax(distances[i] == np.inf)  # Get the index where distance is np.inf
            weights[i] = np.zeros(weights.shape[1])  # Set all weights to 0
            weights[i, inf_index] = 1  # Set the weight corresponding to np.inf to 1
    
    # TODO: Normalize the weights
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights /= row_sums  # Normalize the weights

    return weights

def inversely_transform_image_local(image, control_points, transformations, output_shape=None, p=1.5, interpolation_method="bilinear"):
    """
    Inversely transform an image using local affine transformations.
    
    Parameters:
    - image (np.ndarray): Input image as a 2D array.
    - control_points (list): List of control points (x, y) coordinates.
    - transformations (list): List of affine transformations (3x3 matrices) for each region.
    - output_shape (tuple): Shape of the output image (height, width). If None, use the input image's shape.
    - p (float): Power parameter for weight calculation.
    - interpolation_method (str): Interpolation method for pixel values ('nearest', 'bilinear', 'bicubic').
    
    Returns:
    - np.ndarray: Transformed image with the specified output shape.
    """
    if output_shape is None:
        output_shape = image.shape

    height, width = output_shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    target_coords = np.stack((y.ravel(), x.ravel()), axis=-1)  # Homogeneous coordinates

    # Initialize the source coordinates array
    source_coords = np.zeros((target_coords.shape[0], 2))

    # Calculate the weights for each target coordinate based on the distances to the control points
    weights = calculate_weights(target_coords, control_points, p=p)

    # Apply transformations to control points to get their corresponding positions in the original image
    centers = []
    for i in range(len(control_points)):  # Loop over control points

        center = np.array([control_points[i][1], control_points[i][0], 1])  # Convert (x, y) to (row, col) format
        A = transformations[i]  # Get the transformation matrix for this control point
        A_inv = np.linalg.inv(A)  # Compute the inverse of the transformation matrix
        transformed_center = np.dot(A_inv, center.T)  # Apply the inverse transformation (matrix multiplication)
        centers.append(transformed_center[:2])  # Append the transformed center (x, y) without homogeneous coordinate

    # Calculate weighted sum of centers to determine the source coordinates for each target pixel
    for i in range(len(centers)):
        source_coords += weights[:, i, np.newaxis] * np.array(centers[i])

    # Reshape the source coordinates to match the target shape
    source_indices = source_coords.reshape((height, width, 2))

    # Interpolate pixel values from the source image
    transformed_image = interpolate(image, source_indices, method=interpolation_method)

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
    interpolation_method="bicubic"

    # Output shape is the same as the input image
    output_shape = np.array(image.shape)

    # Image boundary control points (constant step size along the image boundary)
    image_boundary_points = []
    for i in range(0, width, 50):
        image_boundary_points.append(np.array([0, i]))  # Top row
        image_boundary_points.append(np.array([height-1, i]))  # Bottom row
    for i in range(0, height, 50):
        image_boundary_points.append(np.array([i, 0]))  # Left column
        image_boundary_points.append(np.array([i, width-1]))  # Right column

    # Boundary points of region [356:665, 356:665]
    region_boundary_points = []
    for i in range(380, 640, 60):
        region_boundary_points.append(np.array([380, i]))  # Top side of the region
        region_boundary_points.append(np.array([640, i]))  # Bottom side of the region
    for i in range(380, 640, 60):
        region_boundary_points.append(np.array([i, 380]))  # Left side of the region
        region_boundary_points.append(np.array([i, 640]))  # Right side of the region

    # Control points and transformations
    control_points = image_boundary_points + region_boundary_points
    transformations = [np.eye(3)] * len(image_boundary_points) + [A] * len(region_boundary_points)

    # Apply the inverse transformation to the image
    transformed_image = inversely_transform_image_local(image, control_points=control_points, 
                                                        transformations=transformations, output_shape=output_shape, 
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