import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compute_histogram(image, num_bins=256):
    """
    Compute the grayscale histogram of an image.
    
    :param image: Grayscale image as a numpy array.
    :param num_bins: Number of bins for the histogram (default is 256).
    :return: The histogram and bin edges.
    """
    # Flatten the image and compute the histogram with the specified number of bins
    histogram, bin_edges = np.histogram(image.ravel(), bins=num_bins, range=[0, num_bins])
    return histogram, bin_edges

def k_means_grayscale(image, K, initial_means=None, max_iter=100, tolerance=1e-5):
    """
    K-Means algorithm for clustering grayscale values in an image.
    
    Args:
        image (numpy.ndarray): 2D array representing the grayscale image.
        K (int): Number of clusters (i.e., number of means).
        initial_means (numpy.ndarray): Optional initial means (K values), if None, they are chosen randomly.
        max_iters (int): Maximum number of iterations to run the algorithm.
        tol (float): Convergence threshold based on the change in means.
        
    Returns:
        segmented_image (numpy.ndarray): 2D array representing the clustered image.
        means (numpy.ndarray): Final cluster means.
        num_iter: number of iterations
    """
    # Flatten the image to 1D array of grayscale values
    flat_image = image.flatten()
    
    # Initialize means (either given or randomly chosen)
    if initial_means is None:
        means = np.random.choice(flat_image, K, replace=False)
    else:
        means = initial_means
    
    # Store the previous means to check for convergence
    prev_means = np.zeros_like(means)
    num_iter = max_iter
    
    # Initialize cluster assignments
    labels = np.zeros(flat_image.shape, dtype=int)
    
    for iteration in range(max_iter):
        # Step 1: Assign each pixel to the nearest cluster
        for i, z in enumerate(flat_image):
            distances = np.abs(z - means)
            labels[i] = np.argmin(distances)
        
        # Step 2: Update the means based on the assigned clusters
        new_means = np.array([flat_image[labels == k].mean() if np.sum(labels == k) > 0 else means[k]
                              for k in range(K)])
        
        change = np.linalg.norm(new_means - prev_means)
        print(f"The {iteration + 1}-th iteration: {change}")

        # Check for convergence (if the change in means is below the tolerance)
        if change < tolerance:
            num_iter = iteration + 1
            print(f"Converged after {num_iter} iterations.")
            break
        
        prev_means = new_means
        
        # Update the cluster means
        means = new_means
    
    # Step 3: Create the segmented image based on the final labels
    segmented_image = np.reshape(means[labels], image.shape).astype(np.uint8)
    
    return segmented_image, means, num_iter

if __name__ == "__main__":
    
    option = 4
    if option == 1:
        image_path = 'Fig1045(a)(iceberg).tif'
        image_name = 'Fig1045(a)(iceberg)'
        K = 3
        initial_means = [60, 120, 180]
        max_iter = 10
        tolerance = 1e-5
    elif option == 2:
        image_path = 'noisy_Fig1045(a)(iceberg).tif'
        image_name = 'noisy_Fig1045(a)(iceberg)'
        K = 3
        initial_means = [60, 120, 180]
        max_iter = 10
        tolerance = 1e-5
    elif option == 3:
        image_path = 'Fig1016(a)(building_original).tif'
        image_name = 'Fig1016(a)(building_original)'
        K = 4
        initial_means = [60, 120, 180, 240]
        max_iter = 20
        tolerance = 1e-5    
    elif option == 4:
        image_path = 'noisy_Fig1016(a)(building_original).tif'
        image_name = 'noisy_Fig1016(a)(building_original)'
        K = 4
        initial_means = [60, 120, 180, 240]
        max_iter = 20
        tolerance = 1e-5

    # Load and convert the image to grayscale format
    image = Image.open(image_path).convert('L')
    image = np.array(image)  # Convert image to a numpy array for processing
    
    # Compute histogram of the original image
    original_histogram, bin_edges = compute_histogram(image)

    # Run K-means algorithm on the grayscale image
    segmented_image, final_means, num_iter = k_means_grayscale(image, 
                                K=K, initial_means=initial_means, max_iter=max_iter, tolerance=tolerance)
    
    # Compute histogram of the segmented image
    segmented_histogram, _ = compute_histogram(segmented_image)

    # Display the results
    print("Final means (cluster centers):", final_means)
    
    plt.figure(figsize=(12, 8))
    
    # 1. Display original image
    plt.subplot(2, 2, 1)
    plt.title(f"Original Image: {image_name}")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    # 2. Display histogram of original image
    plt.subplot(2, 2, 2)
    plt.title("Original Image Histogram")
    plt.plot(bin_edges[:-1], original_histogram, color='black')
    plt.xlabel("Grayscale value")
    plt.ylabel("Frequency")
    
    # 3. Display segmented image
    plt.subplot(2, 2, 3)
    plt.title(f"Segmented Image (K-means: {num_iter} iterations)")
    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')
    
    # 4. Display histogram of segmented image
    plt.subplot(2, 2, 4)
    plt.title("Segmented Image Histogram")
    plt.plot(bin_edges[:-1], segmented_histogram, color='black')
    for i, mean in enumerate(final_means):
        plt.axvline(mean, linestyle='--', label=f"Mean {i+1}: {mean:.2f}")
    plt.xlabel("Grayscale value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    save_path = f"{K}-means-{image_name}.png"
    plt.savefig(save_path)
    plt.show()