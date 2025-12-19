import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_histogram(image, num_bins=256):
    """
    计算图像的灰度直方图
    :param image: 灰度图像的 numpy 数组
    :param num_bins: 直方图的 bins 数量
    :return: 直方图和 bins 边缘
    """
    histogram, bin_edges = np.histogram(image.ravel(), bins=num_bins, range=[0, num_bins])
    return histogram, bin_edges

def update_histogram(old_hist, new_col=None, remove_col=None, num_bins=256):
    """
    使用增量更新直方图.
    :param old_hist: 当前的直方图.
    :param new_col: 要加入的新的列像素 (可以为 None).
    :param remove_col: 要移除的列像素 (可以为 None).
    :param num_bins: 直方图的 bins 数量.
    :return: 更新后的直方图.
    """
    if new_col is None:
        return old_hist - np.bincount(remove_col, minlength=num_bins)
    elif remove_col is None:
        return old_hist + np.bincount(new_col, minlength=num_bins)
    else:
        return old_hist - np.bincount(remove_col, minlength=num_bins) + np.bincount(new_col, minlength=num_bins)

def compute_local_histograms(image, window_size=(9, 9), num_bins=256):
    """
    计算图像的局部直方图.
    :param image: 输入的灰度图像.
    :param window_size: 邻域窗口的尺寸 (height, width).
    :param num_bins: 直方图的 bins 数量.
    :return: 所有局部直方图的列表.
    """
    h, w = image.shape
    win_h, win_w = window_size
    half_win_h = win_h  // 2  # 使用整数除法，得到窗口半径
    half_win_w = win_w  // 2
    
    # 初始化局部直方图列表
    local_histograms = np.zeros((h, w, num_bins), dtype=np.uint8)

    # 移动完整窗口
    for i in range(h):
        if i == 0: 
            # 计算第一个完整窗口的直方图
            local_histograms[0, 0, :], _ = compute_histogram(image[0:half_win_h+1,0:half_win_w+1], num_bins=num_bins)
        else: 
        # 更新直方图 (垂直移动)
            if i <= half_win_h: # 首
                new_row = image[i+half_win_h, 0:half_win_w+1]
                remove_row = None
            elif i >= h - half_win_h: # 尾
                new_row = None
                remove_row = image[i-half_win_h-1, 0:half_win_w+1]
            else: # 中间部分
                new_row = image[i+half_win_h, 0:half_win_w+1]
                remove_row = image[i-half_win_h-1, 0:half_win_w+1]
            # 更新直方图 (垂直移动)
            local_histograms[i, 0, :] = update_histogram(local_histograms[i-1, 0, :], new_row, remove_row, num_bins=num_bins)

        i_safe_lower = max(i-half_win_h,0)
        i_safe_upper = min(i+half_win_h+1,h)

        for j in range(1, w):
            # 更新直方图 (水平移动)
            if j <= half_win_w: # 首
                new_col = image[i_safe_lower:i_safe_upper, j+half_win_w]
                remove_col = None
            elif j >= w - half_win_w: # 尾
                new_col = None
                remove_col = image[i_safe_lower:i_safe_upper, j-half_win_w-1]
            else: # 中间部分
                new_col = image[i_safe_lower:i_safe_upper, j+half_win_w]
                remove_col = image[i_safe_lower:i_safe_upper, j-half_win_w-1]
            
            # 更新直方图 (水平移动)
            local_histograms[i, j, :] = update_histogram(local_histograms[i, j-1, :], new_col, remove_col, num_bins=num_bins)

    return local_histograms

def compute_cdf(histogram):
    """
    计算累积分布函数 (CDF).
    :param histogram: 图像的灰度直方图.
    :return: 累积分布函数.
    """
    cdf = np.cumsum(histogram)
    cdf_normalized = cdf / cdf[-1]  # 归一化到 [0, 1]
    return cdf_normalized

def local_histogram_equalization(image, local_histograms, num_bins=256):
    """
    对图像进行局部均衡化.
    :param image: 输入的灰度图像.
    :param local_histograms: 所有局部直方图的列表.
    :param num_bins: 直方图的 bins 数量.
    :return: 局部均衡化后的图像.
    """
    h, w = image.shape

    # 初始化均衡化后的图像
    equalized_image = np.zeros_like(image, dtype=np.uint8)

    # 遍历像素点，参照局部直方图应用均衡化映射
    for i in range(h):
        for j in range(w):
            # 使用局部直方图
            current_hist = local_histograms[i, j, :]
            
            # 计算累积分布函数 (CDF)
            current_cdf = compute_cdf(current_hist)
            equalized_map = (num_bins - 1) * current_cdf
            equalized_map = np.round(equalized_map).astype(np.uint8)
            
            # 应用均衡化映射到当前点
            equalized_image[i, j] = equalized_map[image[i, j]]
    
    return equalized_image

def plot_image_and_histogram(image, equalized_image, num_bins=256):
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
    histogram, bin_edges = compute_histogram(image, num_bins=num_bins)
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
    
    # 定义窗口大小
    window_size = (3, 3)
    
    # 计算局部直方图
    num_bins = 256
    local_histograms = compute_local_histograms(image_array, window_size=window_size, num_bins=num_bins)

    # 进行局部均衡化
    equalized_image = local_histogram_equalization(image_array, local_histograms, num_bins=num_bins)
    
    # 保存均衡后的图像
    Image.fromarray(equalized_image).save('Local_equalized_'+ str(image_name))

    # 绘制原始图像和均衡化后的图像及其直方图
    plot_image_and_histogram(image_array, equalized_image, num_bins=num_bins)

if __name__ == "__main__":
    main()
