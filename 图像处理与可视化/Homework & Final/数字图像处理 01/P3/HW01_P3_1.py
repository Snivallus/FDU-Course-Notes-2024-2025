import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compute_histogram(image, num_bins=256):
    """
    计算图像的灰度直方图
    :param image: 灰度图像的 numpy 数组
    :param num_bins: 直方图的 bins 数量
    :return: 直方图和 bins 边缘
    """
    histogram, bin_edges = np.histogram(image.ravel(), bins=num_bins, range=[0, num_bins])
    return histogram, bin_edges

def compute_cdf(histogram):
    """
    计算累积分布函数 (CDF)
    :param histogram: 图像的灰度直方图
    :return: 累积分布函数
    """
    cdf = np.cumsum(histogram)  # 累加直方图的每个值
    cdf_normalized = cdf / cdf[-1]  # 归一化
    return cdf_normalized

def histogram_equalization(image, num_bins=256):
    """
    对图像进行均衡化
    :param image: 输入灰度图像的 numpy 数组
    :return: 均衡化后的图像
    """
    # 计算灰度直方图
    histogram, bin_edges = compute_histogram(image, num_bins=num_bins)
    
    # 计算累积分布函数 (CDF)
    cdf_normalized = compute_cdf(histogram)
    
    # 计算均衡化映射
    equalized_map = (num_bins - 1) * cdf_normalized
    equalized_map = np.round(equalized_map).astype(np.uint8)
    
    # 应用均衡化映射到图像
    equalized_image = equalized_map[image]
    return histogram, bin_edges, equalized_image

def plot_image_and_histogram(image, histogram, bin_edges, equalized_image, num_bins=256):
    """
    绘制原始图像、均衡化后的图像以及它们的灰度直方图
    :param image: 原始灰度图像的 2D numpy 数组
    :param equalized_image: 均衡化后的图像的 2D numpy 数组
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 绘制原始图像
    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=num_bins-1)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 绘制原始图像的直方图
    axes[1, 0].bar(bin_edges[:-1], histogram, width=1, color='gray')
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Gray Level')
    axes[1, 0].set_ylabel('Frequency')
    
    # 绘制均衡化后的图像
    axes[0, 1].imshow(equalized_image, cmap='gray', vmin=0, vmax=num_bins-1)
    axes[0, 1].set_title('Equalized Image')
    axes[0, 1].axis('off')
    
    # 绘制均衡化后的图像的直方图
    equalized_histogram, _ = compute_histogram(equalized_image)
    axes[1, 1].bar(bin_edges[:-1], equalized_histogram, width=1, color='gray')
    axes[1, 1].set_title('Equalized Histogram')
    axes[1, 1].set_xlabel('Gray Level')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def main():
    # 加载灰度图像
    image_name = 'DIP 3.26.tif'
    image = Image.open(image_name).convert('L')
    image_array = np.array(image)
    
    # 进行图像均衡化
    num_bins = 256
    histogram, bin_edges, equalized_image = histogram_equalization(image_array, num_bins=num_bins)

    # 保存均衡后的图像
    Image.fromarray(equalized_image).save('Global_equalized_'+ str(image_name))
    
    # 绘制原始图像和均衡化后的图像及其直方图
    plot_image_and_histogram(image_array, histogram, bin_edges, equalized_image)

if __name__ == "__main__":
    main()
