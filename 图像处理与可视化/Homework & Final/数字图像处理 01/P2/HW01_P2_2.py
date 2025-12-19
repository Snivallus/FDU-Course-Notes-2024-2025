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

# 随机绘制某些窗口的直方图
def plot_random_local_histograms(local_histograms, num_samples=9):
    """
    随机选择一些窗口并绘制它们的直方图
    :param local_histograms: 局部直方图数组，形状为 (height, width, num_bins)
    :param num_samples: 要绘制的随机直方图数量
    """
    height, width, num_bins = local_histograms.shape
    
    # 确定子图的行数和列数
    num_rows = int(np.sqrt(num_samples))
    num_cols = int(np.ceil(num_samples / num_rows))
    
    # 创建子图
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axes = axes.flatten()  # 展平为一维数组，方便迭代
    
    for i in range(num_samples):
        # 随机选择 i, j 索引
        row_idx = np.random.randint(0, height)
        col_idx = np.random.randint(0, width)

        # 提取该位置的局部直方图
        hist = local_histograms[row_idx, col_idx, :]

        # 绘制直方图
        ax = axes[i]  # 选择当前的子图
        ax.bar(range(len(hist)), hist, color='gray')
        ax.set_title(f'Local Histogram at ({row_idx}, {col_idx})')
        ax.set_xlabel('Gray Level')
        ax.set_ylabel('Frequency')
    
    # 调整子图间距
    plt.tight_layout()
    plt.show()

def main():
    # 加载灰度图像
    image = Image.open('DIP 3.10 (pollen).tif').convert('L')
    image_array = np.array(image)
    
    # 定义窗口大小
    window_size = (9, 9)

    # 定义直方图的 bins 数量
    num_bins = 256
    
    # 计算局部直方图
    local_histograms = compute_local_histograms(image_array, window_size=window_size, num_bins=num_bins)

    # 随机绘制某些窗口的直方图
    num_samples = 9
    plot_random_local_histograms(local_histograms, num_samples)

if __name__ == "__main__":
    main()