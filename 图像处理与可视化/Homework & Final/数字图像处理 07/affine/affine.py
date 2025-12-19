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

# Example usage
if __name__ == "__main__":
    # Scaling by 0.5 in x and 0.5 in y
    print("Scaling:\n", apply_affine_transformation('scaling', {'cx': 0.5, 'cy': 0.5}))

    # Rotation by 45 degrees
    print("Rotation:\n", apply_affine_transformation('rotation', {'theta': 45}))

    # Translation by (1, -1)
    print("Translation:\n", apply_affine_transformation('translation', {'tx': 1, 'ty': -1}))

    # Vertical shear with sv = 0.5
    print("Shear Vertical:\n", apply_affine_transformation('shear_vertical', {'sv': 0.5}))

    # Horizontal shear with sh = 0.5
    print("Shear Horizontal:\n", apply_affine_transformation('shear_horizontal', {'sh': 0.5}))
