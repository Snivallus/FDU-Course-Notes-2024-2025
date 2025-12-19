import numpy as np
from PIL import Image

def nearest_neighbor_interpolation(image, scale_factor):
    """
    对图像进行最邻近插值放大

    :param image: 输入的灰度图像 (numpy数组)
    :param scale_factor: 图像缩放的倍数
    :return: 放大后的图像 (numpy数组)
    """
    h, w = image.shape  # 原图像尺寸
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)  # 新图像尺寸

    # 创建目标图像
    new_image = np.zeros((new_h, new_w), dtype=np.uint8)

    # 生成新的像素坐标
    x_new = np.arange(new_h) / scale_factor
    y_new = np.arange(new_w) / scale_factor

    # 计算对应的原图像坐标
    orig_x = np.clip(np.floor(x_new).astype(int), 0, h - 1)
    orig_y = np.clip(np.floor(y_new).astype(int), 0, w - 1)

    # 使用广播机制将原图像的像素值赋给新图像
    new_image = image[orig_x[:, None], orig_y]

    return new_image

# 主函数
if __name__ == "__main__":
    # 读取图像
    image_path = 'downsampled_image.tif'
    img = Image.open(image_path).convert('L')  # 转换为灰度图像
    image_array = np.array(img)

    # 设置缩放倍数 N
    N = 9

    # 进行最邻近插值
    resized_image = nearest_neighbor_interpolation(image_array, N)

    # 保存结果
    output_image = Image.fromarray(resized_image)
    output_image.save('resized_image_nearest_neighbor.png')
    output_image.show()
