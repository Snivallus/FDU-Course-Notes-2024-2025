import numpy as np
from PIL import Image

def bilinear_interpolation(image, scale_factor):
    """
    对图像进行双线性插值放大

    :param image: 输入的灰度图像 (numpy数组)
    :param scale_factor: 图像缩放的倍数
    :return: 放大后的图像 (numpy数组)
    """
    h, w = image.shape  # 原图像尺寸
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)  # 新图像尺寸
    
    # 创建目标图像
    new_image = np.zeros((new_h, new_w), dtype=np.uint8)

    # 生成新图像中像素的浮点坐标
    x_new = np.arange(new_h) / scale_factor
    y_new = np.arange(new_w) / scale_factor

    # 获取整数部分和小数部分
    x1 = np.floor(x_new).astype(int)
    y1 = np.floor(y_new).astype(int)
    x2 = np.clip(x1 + 1, 0, h - 1)
    y2 = np.clip(y1 + 1, 0, w - 1)

    # 计算小数部分
    dx = x_new - x1
    dy = y_new - y1

    # 使用广播机制计算双线性插值
    I11 = image[x1[:, None], y1[None, :]]  # 左上角像素
    I12 = image[x1[:, None], y2[None, :]]  # 右上角像素
    I21 = image[x2[:, None], y1[None, :]]  # 左下角像素
    I22 = image[x2[:, None], y2[None, :]]  # 右下角像素

    # 计算插值结果
    new_image = ((I11 * (1 - dx)[:, None] * (1 - dy)[None, :] +
                  I12 * (1 - dx)[:, None] * dy[None, :] +
                  I21 * dx[:, None] * (1 - dy)[None, :] +
                  I22 * dx[:, None] * dy[None, :])).astype(np.uint8)

    return new_image

# 主函数
if __name__ == "__main__":
    # 读取图像
    image_path = 'downsampled_image.tif'
    img = Image.open(image_path).convert('L')  # 转换为灰度图像
    image_array = np.array(img)

    # 设置缩放倍数 N
    N = 9

    # 进行双线性插值
    resized_image = bilinear_interpolation(image_array, N)

    # 保存结果
    output_image = Image.fromarray(resized_image)
    output_image.save('resized_image_bilinear.png')
    output_image.show()
