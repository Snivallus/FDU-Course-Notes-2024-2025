import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_joint_histogram(image_array, num_bins=256):
    """
    计算 n 维联合直方图。
    
    :param image_array: n 维灰度图像的 numpy 数组
    :param num_bins: 直方图的 bins 数量
    :return: n 维联合直方图和每个维度的 bins 边缘
    """
    # 检查图像是否为 n 维
    if image_array.ndim < 2:
        raise ValueError("The gray-level dimension of input picture must be higher than 2")

    # 将 n 维图像展平为二维数组，其中每一行是一个像素的 n 维向量
    reshaped_array = image_array.reshape(-1, image_array.shape[-1])

    # 计算 n 维联合直方图
    joint_hist, edges = np.histogramdd(
        reshaped_array, 
        bins=[num_bins] * image_array.shape[2], 
        range=[[0, num_bins]] * image_array.shape[2]
    )
    
    return joint_hist, edges

def plot_joint_histogram(joint_hist, edges):
    """
    绘制联合直方图。只绘制二维联合直方图。
    
    :param joint_hist: n 维联合直方图的 numpy 数组
    :param edges: 每个维度的 bins 边缘
    """
    if joint_hist.ndim == 2:
        xedges = np.array(edges[0])
        yedges = np.array(edges[1])

        # 绘制 3D 条形图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        xpos, ypos = np.meshgrid(xedges[:-1] + (xedges[1] - xedges[0]) / 2,
                                yedges[:-1] + (yedges[1] - yedges[0]) / 2, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        # Construct arrays with the dimensions for the bars.
        dx = dy = (xedges[1] - xedges[0]) * 0.8  # 增大柱形图的底面积
        dz = joint_hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
        ax.set_title('3D Joint Histogram of 2D Test Data')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Frequency')
        plt.tight_layout()
        plt.savefig('3d_joint_histogram.png', dpi=150)
        plt.show()

        # 绘制 2D 彩色联合直方图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        H = ax.hist2d(xedges[:-1], yedges[:-1], bins=[xedges, yedges], weights=joint_hist.ravel(), cmap='viridis')
        fig.colorbar(H[3], ax=ax)
        ax.set_title('2D Joint Histogram of 2D Test Data')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        plt.tight_layout()
        plt.savefig('2d_joint_histogram.png', dpi=150)
        plt.show()
    else:
        print(f"Cannot plot {joint_hist.ndim}-dimensional joint histogram. Only 2D histograms can be plotted.")

def main():
    # 加载两幅图像并将它们堆叠为一个 3D 数组
    image1 = Image.open('Contrast_Streched_DIP 3.10 (pollen).tif').convert('L')
    image2 = Image.open('DIP 3.10 (pollen).tif').convert('L')
    
    # 转换为 numpy 数组
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    
    # 检查图像是否具有相同的尺寸
    if image1_array.shape != image2_array.shape:
        raise ValueError("The two pictures must have the same shape!")
    
    # 堆叠图像形成 3D 数组
    stacked_images = np.stack((image1_array, image2_array), axis=-1)
    
    # 打印图像的维度
    print(f"Stacked image dimensions: {stacked_images.shape}")
    
    # 计算联合直方图
    joint_hist, edges = compute_joint_histogram(stacked_images, num_bins=256)
    
    # 绘制联合直方图
    plot_joint_histogram(joint_hist, edges)

if __name__ == "__main__":
    main()
