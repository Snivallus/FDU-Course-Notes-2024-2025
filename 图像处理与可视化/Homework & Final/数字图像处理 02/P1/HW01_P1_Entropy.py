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

def compute_cumulative_probabilities_and_entropy(histogram):
    """
    计算累计概率和累积信息熵
    
    :param histogram: 图像的灰度直方图
    :return: 累计概率 P_k 和累积信息熵
    """
    total_pixels = np.sum(histogram)
    probabilities = histogram / total_pixels

    # 累计概率
    cumulative_probabilities = np.cumsum(probabilities)
    
    # 累积信息熵，跳过 p = 0 的项, 因为极限值 0 * log(0) 定义为 0
    cumulative_entropy = -np.cumsum(np.where(probabilities > 0, probabilities * np.log(probabilities), 0))

    return cumulative_probabilities, cumulative_entropy

def max_entropy_global_thresholding(image):
    """
    最大熵分割算法的实现。
    
    :param image: 输入的灰度图像
    :return: 二值化后的图像、最大熵对应的阈值
    """
    # 计算灰度直方图
    histogram, _ = compute_histogram(image)
    
    # 计算累计概率和累积信息熵
    cumulative_probabilities, cumulative_entropy = compute_cumulative_probabilities_and_entropy(histogram)
    
    # 计算 Shannon 信息熵 H
    with np.errstate(divide='ignore', invalid='ignore'):
        H_1 = np.where(cumulative_probabilities > 0, 
                       np.log(cumulative_probabilities) + cumulative_entropy / cumulative_probabilities, 
                       0)
        H_2 = np.where(1 - cumulative_probabilities > 0, 
                       np.log(1 - cumulative_probabilities) + (cumulative_entropy[-1] - cumulative_entropy) / (1 - cumulative_probabilities), 
                       0)
    
    H = H_1 + H_2

    # 寻找最大熵的所有阈值
    max_entropy = np.max(H)
    best_thresholds = np.where(H == max_entropy)[0]  # 找到所有最大熵对应的阈值
    
    # 如果存在多个最大熵阈值，取平均值
    best_threshold = np.mean(best_thresholds).astype(np.uint8)
    
    print(f"Max entropy {max_entropy} is reached at thresholds {best_thresholds}, average threshold: {best_threshold}")
    
    # 生成二值化图像
    binary_image = (image > best_threshold).astype(np.uint8)

    return binary_image, best_threshold

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
    option = 2 # 更改图像选择
    if option == 1:
        image_name = 'DIP 10.38(a) (noisy_fingerprint).tif'
    elif option == 2:
        image_name = 'DIP 10.39(a) (polymersomes).tif'
    else:
        image_name = 'DIP 2.22 (face).tif'
    image = Image.open(image_name).convert('L')
    image_array = np.array(image)
    
    # 应用最大熵分割算法
    binary_result, threshold = max_entropy_global_thresholding(image_array)

    # 保存二值化分割后的图像
    Image.fromarray(binary_result * 255).save('MaxEntropy_Binary_' + image_name)

    # 绘制结果
    plot_image_and_histogram(image_array, binary_result, threshold)
