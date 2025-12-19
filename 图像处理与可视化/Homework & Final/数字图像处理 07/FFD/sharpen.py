import cv2
import numpy as np

# 读取图像
image_path = 'grid.png'  # 替换为你的图像路径
image = cv2.imread(image_path)

# 定义锐化滤波器
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

# 应用锐化滤波器
sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)

# 保存锐化后的图像
cv2.imwrite('sharpened_image.jpg', sharpened_image)