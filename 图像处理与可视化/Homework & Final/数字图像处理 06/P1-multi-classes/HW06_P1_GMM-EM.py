import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import multivariate_normal

def compute_histogram(image, num_bins=256):
    """
    Compute the grayscale histogram of an image.
    
    :param image: Grayscale image as a numpy array.
    :param num_bins: Number of bins for the histogram (default is 256).
    :return: The histogram and bin edges.
    """
    histogram, bin_edges = np.histogram(image.ravel(), bins=num_bins, range=[0, num_bins])
    return histogram, bin_edges

def gmm_em_grayscale(image, K, initial_means=None, max_iter=1000, tolerance=1e-5):
    """
    Gaussian Mixture Model (GMM) EM algorithm for clustering grayscale values in an image.
    
    Args:
        image (numpy.ndarray): 2D array representing the grayscale image.
        K (int): Number of clusters (i.e., number of Gaussian components).
        initial_means (numpy.ndarray): Optional initial means (K values), if None, they are chosen randomly.
        max_iters (int): Maximum number of iterations to run the algorithm.
        tolerance (float): Convergence threshold based on the change in log-likelihood.
        
    Returns:
        segmented_image (numpy.ndarray): 2D array representing the clustered image.
        means (numpy.ndarray): Final cluster means.
        covariances (numpy.ndarray): Final cluster covariance matrices.
        alphas (numpy.ndarray): Final mixture component weights.
        num_iter: number of iterations
    """
    # Flatten the image to 1D array of grayscale values
    flat_image = image.flatten()
    n = flat_image.shape[0]
    
    # Initialize means, covariances, and mixture coefficients
    if initial_means is None:
        means = np.random.choice(flat_image, K, replace=False)
    else:
        means = np.array(initial_means)
    
    # Initialize weights (alphas) and covariance matrices (Sigma)
    alphas = np.ones(K) / K  # Uniform initial weights
    covariances = np.array([np.var(flat_image)] * K)  # Initial covariances set to the variance of the image
    
    # Initialize responsibilities (gamma values)
    gamma = np.zeros((n, K))
    num_iter = max_iter
    
    prev_means = np.copy(means)
    prev_covariances = np.copy(covariances)
    prev_alphas = np.copy(alphas)
    
    for iteration in range(max_iter):
        # E-step: Compute responsibilities (gamma values)
        for k in range(K):
            # Calculate the Gaussian probability density function for each pixel
            pdf = multivariate_normal.pdf(flat_image, mean=means[k], cov=covariances[k])
            gamma[:, k] = alphas[k] * pdf
        
        # Normalize the responsibilities (soft assignment of pixels to clusters)
        gamma = gamma / np.sum(gamma, axis=1)[:, np.newaxis]
        
        # M-step: Update the model parameters (means, covariances, alphas)
        for k in range(K):
            # Update the mean for each cluster
            means[k] = np.sum(gamma[:, k] * flat_image) / np.sum(gamma[:, k])
            
            # Update the covariance for each cluster
            covariances[k] = np.sum(gamma[:, k] * (flat_image - means[k])**2) / np.sum(gamma[:, k])
            
            # Update the weight (alpha) for each cluster
            alphas[k] = np.sum(gamma[:, k]) / n
        
        # Convergence check based on the variation of means, covariances, and alphas
        mean_change = np.linalg.norm(means - prev_means)
        cov_change = np.linalg.norm(covariances - prev_covariances)
        alpha_change = np.linalg.norm(alphas - prev_alphas)
        
        # Combine changes to determine convergence
        total_change = mean_change + cov_change + alpha_change
        total_change = total_change / (np.linalg.norm(means) + np.linalg.norm(covariances) + np.linalg.norm(alphas))
        print(f"The {iteration+1}-th iteration: {total_change}")

        if total_change < tolerance:
            num_iter = iteration + 1
            print(f"Converged after {num_iter} iterations.")
            break
        
        # Update previous values for next iteration
        prev_means = np.copy(means)
        prev_covariances = np.copy(covariances)
        prev_alphas = np.copy(alphas)

    # Assign the final cluster labels based on the highest responsibility
    labels = np.argmax(gamma, axis=1)
    
    # Step 3: Create the segmented image based on the final labels
    segmented_image = np.reshape(means[labels], image.shape).astype(np.uint8)
    
    return segmented_image, means, covariances, alphas, num_iter

if __name__ == "__main__":

    option = 4
    if option == 1:
        image_path = 'Fig1045(a)(iceberg).tif'
        image_name = 'Fig1045(a)(iceberg)'
        K = 3
        initial_means = [28.74370715, 131.95402687, 221.96702829] # K-means result
        max_iter = 1000
        tolerance = 1e-5
    elif option == 2:
        image_path = 'noisy_Fig1045(a)(iceberg).tif'
        image_name = 'noisy_Fig1045(a)(iceberg)'
        K = 3
        initial_means = [28.74370715, 131.95402687, 221.96702829] # K-means result
        max_iter = 1000
        tolerance = 1e-5
    elif option == 3:
        image_path = 'Fig1016(a)(building_original).tif'
        image_name = 'Fig1016(a)(building_original)'
        K = 4
        initial_means = [47.21778267, 101.72042789, 166.50053224, 234.75427634] # K-means result
        max_iter = 1000
        tolerance = 1e-5 
    elif option == 4:
        image_path = 'noisy_Fig1016(a)(building_original).tif'
        image_name = 'noisy_Fig1016(a)(building_original)'
        K = 4
        initial_means = [47.21778267, 101.72042789, 166.50053224, 234.75427634] # K-means result
        max_iter = 1000
        tolerance = 1e-5
        
    # Load and convert the image to grayscale format
    image = Image.open(image_path).convert('L')
    image = np.array(image)  # Convert image to a numpy array for processing
    
    # Compute histogram of the original image
    original_histogram, bin_edges = compute_histogram(image)

    # Run GMM-EM algorithm on the grayscale image
    segmented_image, final_means, final_covariances, final_alphas, num_iter = gmm_em_grayscale(image, 
                                        K=K, initial_means=initial_means, max_iter=max_iter, tolerance=tolerance)
    
    # Compute histogram of the segmented image
    segmented_histogram, _ = compute_histogram(segmented_image)

    # Display the results
    print("Final means (cluster centers):", final_means)
    print("Final covariances:", final_covariances)
    print("Final alphas (weights):", final_alphas)
    
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
    plt.title(f"Segmented Image (GMM-EM: {num_iter} iterations)")
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
    
    # Save the results to file
    save_path = f"{K}-GMM-{image_name}.png"
    plt.savefig(save_path)
    plt.show()