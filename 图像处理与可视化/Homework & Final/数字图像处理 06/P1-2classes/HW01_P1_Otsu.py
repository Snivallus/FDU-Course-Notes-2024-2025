import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_histogram(image, num_bins=256):
    """
    计算图像的灰度直方图。
    
    :param image: 灰度图像的 numpy 数组
    :param num_bins: 直方图的 bins 数量
    :return: 直方图和 bins 边缘
    """
    histogram, bin_edges = np.histogram(image.ravel(), bins=num_bins, range=[0, num_bins])
    return histogram, bin_edges

def compute_cumulative_probabilities_and_weighted_sum(histogram):
    """
    计算累计概率和累积灰度加权和。

    :param histogram: 图像的灰度直方图
    :return: 累计概率 P_k 和累积灰度加权和 S_k
    """
    total_pixels = np.sum(histogram)
    probabilities = histogram / total_pixels
    cumulative_probabilities = np.cumsum(probabilities)
    cumulative_weighted_sum = np.cumsum(np.arange(len(histogram)) * probabilities)
    
    return cumulative_probabilities, cumulative_weighted_sum

def otsu_global_thresholding(image):
    """
    Otsu 算法的实现。

    :param image: 输入的灰度图像
    :return: 二值化后的图像、Otsu 阈值以及可分离性测度
    """
    histogram, _ = compute_histogram(image)
    cumulative_probabilities, cumulative_weighted_sum = compute_cumulative_probabilities_and_weighted_sum(histogram)

    # 计算全局均值
    mu_global = cumulative_weighted_sum[-1]

    # 计算全局方差
    sigma_global_squared = np.sum((np.arange(len(histogram)) - mu_global) ** 2 * (histogram / np.sum(histogram)))
    print(f"Global mean is {mu_global} and global variance is {sigma_global_squared}")
    
    # 类间方差的向量化计算
    P = cumulative_probabilities
    S = cumulative_weighted_sum
    
    # 避免除以 0 或无效计算的情况，先屏蔽掉 P_k 为 0 和 1 的值
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_between_class_squared = np.where(
            (P > 0) & (P < 1),
            (P * mu_global - S) ** 2 / (P * (1 - P)),
            0
        )
    
    # 寻找最大类间方差
    max_variance = np.max(sigma_between_class_squared)
    best_thresholds = np.where(sigma_between_class_squared == max_variance)[0]  # 找到所有最大类间方差的阈值
    
    # 如果存在多个最大类间方差的阈值，取平均值
    best_threshold = np.mean(best_thresholds).astype(np.uint8)
    
    print(f"Max variance {max_variance} is reached at thresholds {best_thresholds}, average threshold: {best_threshold}")
    # 生成二值化图像
    binary_image = (image > best_threshold).astype(np.uint8)

    # 计算最好阈值的可分离性测度
    separability_measure = max_variance / sigma_global_squared if sigma_global_squared > 0 else 0

    return binary_image, best_threshold, separability_measure

def plot_image_and_histogram(image, binary_image, threshold, image_name):
    """
    绘制原始图像和二值化后的图像及其直方图。

    :param image: 原始灰度图像的 2D numpy 数组
    :param binary_image: 二值化后的图像的 2D numpy 数组
    :param threshold: 用于二值化的阈值
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 绘制原始图像
    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 绘制原始图像的直方图
    histogram, bin_edges = compute_histogram(image)
    axes[0, 1].bar(bin_edges[:-1], histogram, width=1, color='gray')
    axes[0, 1].set_title('Original Histogram')
    axes[0, 1].set_xlabel('Gray Level')
    axes[0, 1].set_ylabel('Frequency')

    # 添加阈值直线
    axes[0, 1].axvline(x=threshold, color='red', linestyle='--', label='Threshold = {}'.format(int(threshold)))
    axes[0, 1].legend()

    # 绘制二值化后的图像
    axes[1, 0].imshow(binary_image, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Binary Image')
    axes[1, 0].axis('off')

    # 绘制二值化后的图像的直方图
    binary_histogram, _ = compute_histogram(binary_image, num_bins=2)
    axes[1, 1].bar(np.arange(2), binary_histogram, width=0.1, color='gray')
    axes[1, 1].set_title('Binary Histogram')
    axes[1, 1].set_xlabel('Gray Level')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    save_path = f"Otsu_{image_name}.png"
    plt.savefig(save_path)
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 加载灰度图像
    option = 2 # 更改图像选择
    if option == 1:
        image_name = 'Fig1038(a)(noisy_fingerprint).tif'
    else:
        image_name = 'noisy_Fig1038(a)(noisy_fingerprint).tif'

    image = Image.open(image_name).convert('L')
    image_array = np.array(image)
    
    # 应用 Otsu 阈值算法
    binary_image, threshold, separability_measure = otsu_global_thresholding(image_array)

    # 输出 Ostu 阈值的可分离性测度
    print(f"separability measure: {separability_measure}")

    # 保存二值化分割后的图像
    # Image.fromarray(binary_image * 255).save('Otsu_Binary_separated_' + image_name)

    # 绘制结果
    plot_image_and_histogram(image_array, binary_image, threshold, image_name)
