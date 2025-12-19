import numpy as np
from PIL import Image

def downsample_image(image, scale_factor):
    """
    将图像降采样，降低分辨率。
    
    :param image: 输入的灰度图像 (numpy数组)
    :param scale_factor: 缩小的倍数
    :return: 降采样后的图像 (numpy数组)
    """
    h, w = image.shape
    # 计算降采样后的尺寸
    new_h, new_w = int(h / scale_factor), int(w / scale_factor)
    
    # 创建降采样后的图像
    downsampled_image = np.zeros((new_h, new_w), dtype=np.uint8)
    
    for i in range(new_h):
        for j in range(new_w):
            # 对于下采样，简单选择最近邻的像素
            downsampled_image[i, j] = image[i * scale_factor, j * scale_factor]
    
    return downsampled_image

# 主函数
if __name__ == "__main__":
    # 读取原图像
    image_path = 'DIP 2.20(a) (chronometer 3600x2808).tif'  # 输入图像路径
    img = Image.open(image_path).convert('L')  # 转换为灰度图像
    image_array = np.array(img)
    Image.fromarray(image_array).save('DIP 2.20(a) (chronometer 3600x2808).png')

    # 设置缩小倍数 N
    scale_factor = 9

    # 进行降采样
    downsampled_image = downsample_image(image_array, scale_factor)

    # 保存降采样后的图像
    downsampled_image_pil = Image.fromarray(downsampled_image)
    downsampled_image_pil.save('downsampled_image.tif')
    downsampled_image_pil.show()
