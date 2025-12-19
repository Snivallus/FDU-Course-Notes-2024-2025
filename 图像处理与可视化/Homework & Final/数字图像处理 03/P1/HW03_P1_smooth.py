import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def smooth_filter(n, type="box", sigma=1, scale=1):
    """生成平滑滤波器核。

    参数：
        n (int): 核的大小，必须是奇数。
        type (str): 滤波器类型，可以是 "box" 或 "gaussian"。
        sigma (float): 高斯滤波器的标准差，仅在类型为 "gaussian" 时使用。
        scale (float): 高斯核的缩放因子，仅在类型为 "gaussian" 时使用。

    返回：
        tuple: 包含一维权重 w1, w2 和二维滤波器核。
    """
    assert n % 2 == 1, "n 必须是奇数"

    if type == "box":
        # 盒式滤波器
        w1 = np.ones(n).reshape(-1, 1) / n
        w2 = w1.copy()
        kernel = np.outer(w1, w2)  # 计算外积，形成二维核
    else:
        if n > np.ceil(6 * np.sqrt(sigma)):
            print("Warning: A width of ceil(6 * sigma)^2 may be too large.")
            print("For Gaussian kernels, we generally use a size of ⌈6σ⌉ × ⌈6σ⌉.")
            print("This is because the value of the Gaussian function becomes negligible for r ≥ 3σ.")
            print("Therefore, a Gaussian kernel of size ⌈6σ⌉ × ⌈6σ⌉ has a similar effect to a larger-sized kernel.")
            print("Since we typically work with odd-sized kernels, we select the smallest odd number greater than 6σ.")
            print("(For example, when σ = 7, we use the smallest odd number greater than 6σ = 42, which is 43.)")

        # Gauss 滤波器
        w1 = np.zeros(n).reshape(-1, 1)  # 初始化一维 Gauss 核
        for x in range(n):
            x_coord = x - (n - 1) / 2  # 中心化坐标
            w1[x] = np.exp(-(x_coord**2) / (2 * sigma**2))
        kernel = np.outer(w1, w1)
        w1 = np.sqrt(scale) * w1 / np.sqrt(np.sum(kernel))
        w2 = w1.copy()
        kernel = kernel / np.sum(kernel)  # 计算外积，形成二维核

    return w1, w2, kernel

def apply_and_plot(image_array, filter_sizes, filter_type, sigma=None):
    """应用平滑滤波器并绘制结果图像。"""
    plt.figure(figsize=(15, 10))
    plt.subplot(1, len(filter_sizes) + 1, 1)
    plt.title("Original Image")
    plt.imshow(image_array, cmap='gray')
    plt.axis('off')

    for i, n in enumerate(filter_sizes):
        if filter_type == "box":
            w1, w2, _ = smooth_filter(n, type="box")
        else:
            # 确保 sigma 是一个向量，并引用相应的值
            current_sigma = sigma[i] if sigma is not None else 1
            w1, w2, _ = smooth_filter(n, type="gaussian", sigma=current_sigma)

        # 应用卷积
        intermediate_image = convolve(image_array, w1, mode='constant', cval=0)
        smoothed_image = convolve(intermediate_image, w2.T, mode='constant', cval=0)

        plt.subplot(1, len(filter_sizes) + 1, i + 2)
        if filter_type == "box":
            plt.title(f"{filter_type.capitalize()} Filter (n={n})")
        else:
            plt.title(f"{filter_type.capitalize()} Filter (n={n}, sigma={current_sigma})")
        plt.imshow(smoothed_image, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 读取图像
    image_path = 'Fig 03.33(a) (test_pattern_blurring_orig).tif'
    img = Image.open(image_path).convert('L')  # 转换为灰度图像
    image_array = np.array(img)

    # 盒式核平滑
    box_filter_sizes = [3, 11, 21]
    apply_and_plot(image_array, box_filter_sizes, filter_type="box")

    # 高斯核平滑
    gaussian_filter_sizes = [21, 43, 85]
    gaussian_sigmas = [3.5, 7, 7]
    apply_and_plot(image_array, gaussian_filter_sizes, filter_type="gaussian", sigma=gaussian_sigmas)