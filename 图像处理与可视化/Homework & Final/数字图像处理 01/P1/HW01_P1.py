import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 定义 Piecewise_Linear_Transformation 函数
def piecewise_linear_transformation(r, r1, r2, s1, s2, L=256):
    """
    对给定的灰度值 r 应用分段线性对比度拉伸变换.
    
    :param r: 输入的灰度值，要求是 [0, L-1] 范围内的整数
    :param r1: 第一个输入灰度分段阈值，要求满足 0 <= r1 < r2 < L-1
    :param r2: 第二个输入灰度分段阈值
    :param s1: 第一个输出灰度值，映射到 r1
    :param s2: 第二个输出灰度值，映射到 r2
    :param L: 灰度级别, 默认值为 256, 表示输入图像的灰度范围是 [0, 255]
    
    :return: 对应的输出灰度值 s, 若输入灰度值 r 无效，则返回 None
    """
    
    # 检查输入灰度值 r 是否在 [0, L-1] 范围内
    if not (0 <= r <= L - 1):
        print(f"Invalid input intensity value r = {r}, should be an integer within [0,{L-1}]")
        return None
    
    # 第一段: 对应 0 <= r <= r1 的灰度值线性映射
    if 0 <= r <= r1:
        return (s1 / r1) * r
    
    # 第二段: 对应 r1 < r <= r2 的灰度值线性映射
    elif r1 < r <= r2:
        return s1 + ((s2 - s1) / (r2 - r1)) * (r - r1)
    
    # 第三段: 对应 r2 < r <= L-1 的灰度值线性映射
    elif r2 < r <= L - 1:
        return s2 + ((L - 1 - s2) / (L - 1 - r2)) * (r - r2)
    
    # 默认情况下不会进入此分支，但为了安全性也处理一下无效输入
    else:
        print(f"Unexpected input value r = {r}")
        return None

# 应用分段线性变换到整个图像
def apply_contrast_stretching(image, r1, r2, s1, s2, L=256):
    """
    将分段线性变换应用到输入的灰度图像.
    :param image: 输入灰度图像的二维numpy数组
    :param r1, r2, s1, s2: 变换中的阈值和输出值
    :param L: 灰度值的级数，默认为 256
    :return: 经过对比度拉伸的图像
    """
    # 获取图像的尺寸
    height, width = image.shape
    print(f"height = {height}, width = {width}")

    # 创建新的图像数组用于存储输出图像
    stretched_image = np.zeros_like(image, dtype=np.uint8)
    
    # 对每个像素应用分段线性变换
    for i in range(height):
        for j in range(width):
            stretched_image[i, j] = piecewise_linear_transformation(image[i, j], r1, r2, s1, s2, L)

    return stretched_image

# 读取图像，应用对比度拉伸，并显示结果
def in_and_out(image_name, a1=3/8, a2=5/8, b1=1/8, b2=7/8):
    """
    读取图像，应用分段线性对比度拉伸，并显示和保存结果。
    :param image_name: 输入图像的文件名
    :param a1, a2: 输入灰度值比例（默认为图像灰度范围的 3/8 到 5/8 之间）
    :param b1, b2: 输出灰度值比例（默认为图像灰度范围的 1/8 到 7/8 之间）
    """
    
    # 读取灰度图像
    input_image = Image.open(image_name).convert("L")
    image_array = np.array(input_image)
    
    # 获取灰度值的级数 L
    L = 256  # 假设输入图像是8位灰度图像，灰度级别为256
    
    # 定义对比度拉伸参数
    r1 = max(a1 * L, 0)  # 确保 r1 不小于 0
    r2 = min(a2 * L, L - 1)  # 确保 r2 不超过 L-1
    s1 = max(b1 * L, 0)  # 确保 s1 不小于 0
    s2 = min(b2 * L, L - 1)  # 确保 s2 不超过 L-1
    print(f"[r1, r2] = [{r1}, {r2}], [s1, s2] = [{s1}, {s2}]")
    
    # 应用分段线性对比度拉伸
    stretched_image = apply_contrast_stretching(image_array, r1, r2, s1, s2, L)
    
    # 转换为PIL图像
    output_image = Image.fromarray(stretched_image)

    # 显示原始图像和拉伸后的图像
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(input_image, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Contrast Stretched Image")
    plt.imshow(output_image, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    
    plt.show()

    # 保存拉伸后的图像
    output_image.save('Contrast_Streched_'+ str(image_name))

# 主函数
def main():
    """
    主函数，用于运行对比度拉伸算法。
    """
    # 调用 in_and_out 函数处理图像
    in_and_out("DIP 3.10 (pollen).tif")

if __name__ == "__main__":
    main()