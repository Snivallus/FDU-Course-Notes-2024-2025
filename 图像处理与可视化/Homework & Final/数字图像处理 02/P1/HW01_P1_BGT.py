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

def basic_global_thresholding(image, epsilon=1e-5):
    """
    基础全局阈值算法的实现。

    :param image: 输入的灰度图像
    :param epsilon: 迭代停止的阈值
    :return: 二值化后的图像以及阈值
    """
    histogram, _ = compute_histogram(image)
    cumulative_probabilities, cumulative_weighted_sum = compute_cumulative_probabilities_and_weighted_sum(histogram)

    # 初始阈值
    tau = cumulative_weighted_sum[-1]
    
    while True:
        # 计算 G1 和 G2 的平均灰度值
        lower_bound = int(np.floor(tau))
        if lower_bound >= len(histogram) - 1:
            lower_bound = len(histogram) - 1

        if lower_bound < 0:
            lower_bound = 0

        P1 = cumulative_probabilities[lower_bound]
        P2 = 1 - P1

        if P1 > 0:
            mu1 = cumulative_weighted_sum[lower_bound] / P1
        else:
            mu1 = 0

        if P2 > 0:
            mu2 = (cumulative_weighted_sum[-1] - cumulative_weighted_sum[lower_bound]) / P2
        else:
            mu2 = 0

        # 更新阈值
        new_tau = 0.5 * (mu1 + mu2)

        # 检查是否收敛
        if abs(new_tau - tau) < epsilon:
            break
        
        tau = new_tau

    # 生成二值化图像
    binary_image = (image > tau).astype(np.uint8)
    return binary_image, np.floor(tau).astype(int)

def plot_image_and_histogram(image, binary_image, threshold):
    """
    绘制原始图像和二值化后的图像及其直方图。

    :param image: 原始灰度图像的 2D numpy 数组
    :param binary_image: 二值化后的图像的 2D numpy 数组
    :param threshold: 用于二值化的阈值
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 绘制原始图像
    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 绘制原始图像的直方图
    histogram, bin_edges = compute_histogram(image)
    axes[1, 0].bar(bin_edges[:-1], histogram, width=1, color='gray')
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Gray Level')
    axes[1, 0].set_ylabel('Frequency')
    
    # 添加阈值直线
    axes[1, 0].axvline(x=threshold, color='red', linestyle='--', label='Threshold = {}'.format(int(threshold)))
    axes[1, 0].legend()

    # 绘制二值化后的图像
    axes[0, 1].imshow(binary_image, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Binary Image')
    axes[0, 1].axis('off')

    # 绘制二值化后的图像的直方图
    binary_histogram, _ = compute_histogram(binary_image, num_bins=2)
    axes[1, 1].bar(np.arange(2), binary_histogram, width=0.1, color='gray')
    axes[1, 1].set_title('Binary Histogram')
    axes[1, 1].set_xlabel('Gray Level')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 加载灰度图像
    option = 3 # 更改图像选择
    if option == 1:
        image_name = 'DIP 10.38(a) (noisy_fingerprint).tif'
    elif option == 2:
        image_name = 'DIP 10.39(a) (polymersomes).tif'
    else:
        image_name = 'DIP 2.22 (face).tif'
    image = Image.open(image_name).convert('L')
    image_array = np.array(image)
    
    # 应用基础全局阈值算法
    binary_result, threshold = basic_global_thresholding(image_array)
        
    # 保存二值化分割后的图像
    Image.fromarray(binary_result * 255).save('BGT_Binary_separated_'+ str(image_name))

    # 绘制结果
    plot_image_and_histogram(image_array, binary_result, threshold)
