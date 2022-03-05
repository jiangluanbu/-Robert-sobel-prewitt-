import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import MyCV

#中文显示工具函数
def set_chinese():
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    set_chinese()

    gray_imager = np.asarray(Image.open(r'./1.jpeg').convert('L'))

    robert = [[[0, 4], [-4, 0]], [[4, 0], [0, -4]]]
    sobel = [[[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], [[-2, -4, -2], [0, 0, 0], [2, 4, 2]]]
             # [[0, 2, 4], [-2, 0, 2], [-4, -2, 0]], [[-4, -2, 0], [-2, 0, -2], [0, -2, 4]]
    prewitt = [[[-4, 0, 4], [-4, 0, 4], [-4, 0, 4]], [[-4, -4, -4], [0, 0, 0], [4, 4, 4]]]
               # [[0, 4, 4], [-4, 0, 4], [-4, -4, 0]], [[-4, -4, 0], [-4, 0, 4], [0, 4, 4]]
    robert_img = MyCV.suanzi(gray_imager, robert, 1)
    sobel_img = MyCV.suanzi(gray_imager, sobel, 1)
    prewitt_img = MyCV.suanzi(gray_imager, prewitt, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax1.set_title('原图')
    ax1.imshow(gray_imager, cmap='gray', vmin=0, vmax=255)

    ax1 = fig.add_subplot(234)
    ax1.set_title('罗伯特变换')
    ax1.imshow(robert_img, cmap='gray', vmin=0, vmax=255)

    ax1 = fig.add_subplot(235)
    ax1.set_title('Sobel变换')
    ax1.imshow(sobel_img, cmap='gray', vmin=0, vmax=255)

    ax1 = fig.add_subplot(236)
    ax1.set_title('prewitt变换')
    ax1.imshow(prewitt_img, cmap='gray', vmin=0, vmax=255)

    plt.show()