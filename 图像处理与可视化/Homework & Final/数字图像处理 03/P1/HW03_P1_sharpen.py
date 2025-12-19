import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def laplace_sharpening(image_array):
    # 定义 Laplace 核
    laplace_kernel = np.array([[0, 1, 0],
                               [1,-4, 1],
                               [0, 1, 0]], dtype=np.float32)

    # 应用 Laplace 核
    laplace_filtered  = convolve(image_array.astype(float), laplace_kernel, mode='reflect')
    
    # Laplace 核具有负中心系数, 故原图像应当减去拉伸后的滤波图像得到锐化图像
    sharpened_image = image_array.astype(float) - laplace_filtered

    # 确保像素值在有效范围内
    laplace_filtered= np.clip(laplace_filtered, 0, 255).astype(np.uint8)
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

    return laplace_filtered, sharpened_image

if __name__ == "__main__":
    # 读取图像
    image_path = 'Fig 03.38(a)(blurry_moon).tif'
    img = Image.open(image_path).convert('L')  # 转换为灰度图像
    image_array = np.array(img)

    # 应用锐化
    laplace_filtered, sharpened_image = laplace_sharpening(image_array)

    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image_array, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Filtered Image')
    plt.imshow(laplace_filtered, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Sharpened Image')
    plt.imshow(sharpened_image, cmap='gray')
    plt.axis('off')

    plt.show()