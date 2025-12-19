import numpy as np
from PIL import Image


def cubic_kernel(x):
    """
    双三次插值核函数
    """
    abs_x = np.abs(x)
    abs_x2 = abs_x ** 2
    abs_x3 = abs_x ** 3
    
    result = np.where(
        abs_x <= 1,
        (1.5 * abs_x3 - 2.5 * abs_x2 + 1),
        np.where(
            (abs_x > 1) & (abs_x <= 2),
            (-0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2),
            0
        )
    )
    
    return result

def bicubic_interpolation(image, scale_factor):
    """
    对图像进行双三次插值放大
    
    :param image: 输入的灰度图像 (numpy数组)
    :param scale_factor: 图像缩放的倍数
    :return: 放大后的图像 (numpy数组)
    """
    h, w = image.shape  # 原图像尺寸
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)  # 新图像尺寸
    
    # 创建目标图像，初始化为float64类型
    new_image = np.zeros((new_h, new_w), dtype=np.float64)
    
    # 生成新图像中像素的浮点坐标
    x_new = np.arange(new_h) / scale_factor
    y_new = np.arange(new_w) / scale_factor

    # 获取整数部分
    x_floor = np.floor(x_new).astype(int)
    y_floor = np.floor(y_new).astype(int)
    
    # 确保坐标不越界
    x_floor = np.clip(x_floor, 1, h - 3)
    y_floor = np.clip(y_floor, 1, w - 3)

    # 计算小数部分
    dx = x_new - x_floor
    dy = y_new - y_floor
    
    # 双三次插值
    for i in range(-1, 3):
        for j in range(-1, 3):
            # 获取插值权重
            weight_x = cubic_kernel(dx - i)
            weight_y = cubic_kernel(dy - j)
            
            # 获取原图像中的像素点
            patch = image[(x_floor + i)[:, None], (y_floor + j)[None, :]]

            # 对所有像素点进行加权求和
            new_image += (weight_x[:, None] * weight_y[None, :]) * patch

    # 将插值结果剪裁到合法范围并转换为uint8
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    
    return new_image

if __name__ == "__main__":
    # 读取图像
    image_path = 'downsampled_image.tif'
    img = Image.open(image_path).convert('L')  # 转换为灰度图像
    image_array = np.array(img)

    # 设置缩放倍数 N
    N = 9

    # 进行双三方插值
    resized_image = bicubic_interpolation(image_array, N)

    # 保存结果
    output_image = Image.fromarray(resized_image)
    output_image.save('resized_image_bicubic.png')
    output_image.show()