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

def otsu_local_thresholding(image, window_size=(9, 9), num_bins=256):
    """
    Otsu 局部阈值算法的实现。

    :param image: 输入的灰度图像
    :param window_size: 局部窗口大小
    :param num_bins: 直方图的 bins 数量
    :return: 二值化后的图像和局部阈值矩阵
    """
    h, w = image.shape

    # 初始化二值化的图像和阈值矩阵
    binary_image = np.zeros_like(image, dtype=np.uint8)
    thresholds = np.zeros((h, w), dtype=np.uint8)

    # 计算每个局部窗口的直方图
    local_histograms = compute_local_histograms(image, window_size=window_size, num_bins=num_bins)

    # 遍历图像中的每个像素
    for i in range(h):
        for j in range(w):
            # 使用该像素位置的局部直方图
            current_hist = local_histograms[i, j]

            # 计算累积概率和累积加权和
            cumulative_probabilities, cumulative_weighted_sum = compute_cumulative_probabilities_and_weighted_sum(current_hist)

            # 全局均值
            mu_global = cumulative_weighted_sum[-1]

            # 类间方差的向量化计算
            P = cumulative_probabilities
            S = cumulative_weighted_sum

            with np.errstate(divide='ignore', invalid='ignore'):
                sigma_between_class_squared = np.where(
                    (P > 0) & (P < 1),
                    (P * mu_global - S) ** 2 / (P * (1 - P)),
                    0
                )

            # 寻找最大类间方差对应的阈值
            best_threshold = np.argmax(sigma_between_class_squared)

            # 记录最佳阈值
            thresholds[i, j] = best_threshold

            # 应用阈值进行二值化
            binary_image[i, j] = (image[i, j] > best_threshold).astype(np.uint8)

    return binary_image, thresholds

def plot_image_and_histogram(image, binary_image):
    """
    绘制原始图像和二值化后的图像及其直方图。

    :param image: 原始灰度图像的 2D numpy 数组
    :param binary_image: 二值化后的图像的 2D numpy 数组
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
    option = 4 # 更改图像选择
    if option == 1:
        image_name = 'DIP 10.38(a) (noisy_fingerprint).tif'
    elif option == 2:
        image_name = 'DIP 10.39(a) (polymersomes).tif'
    elif option == 3:
        image_name = 'DIP 2.22 (face).tif'
    elif option == 4:
        image_name = 'DIP 10.49(a)(spot_shaded_text_image).tif'
    else:
        image_name = 'DIP 10.43(a) (yeast_USC).tif'
    image = Image.open(image_name).convert('L')
    image_array = np.array(image)
    
    # 定义窗口大小
    window_size = (40, 40)

    # 应用 Otsu 局部阈值算法
    binary_image, thresholds = otsu_local_thresholding(image_array, window_size=window_size)

    # 保存二值化分割后的图像
    Image.fromarray(binary_image * 255).save('Otsu_Binary_separated_' + image_name)

    # 绘制结果
    plot_image_and_histogram(image_array, binary_image)
