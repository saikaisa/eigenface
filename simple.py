import numpy as np
from numpy import *
from PIL import Image
import os
from matplotlib import pyplot as plt


def loadImageSet(path):
    # 匹配指定路径下的所有以 .pgm 结尾的文件，并将文件路径存储在 filenames 列表中
    filenames = []
    for file in os.listdir(path):
        if file.endswith('.pgm'):
            filenames.append(os.path.join(path, file))
    filenames.sort()
    # 使用 PIL 库中的 Image.open 函数打开每个文件
    # 并将图像转换为灰度（convert 函数，参数 L 表示灰度）
    # 然后调整图像大小为(98, 116)像素（resize 函数）
    # 将每个处理过的图像对象存储在 images 列表中。
    images = [Image.open(fn).convert('L').resize((98, 116)) for fn in filenames]
    # 先将每个 PIL 图像对象 img 转换为 NumPy 数组 (np.array(img))
    # 然后将每个数组转换为一维向量 (np.array(img).flatten())
    # 最后将所有向量存储在一个矩阵中 (np.asarray([np.array(img).flatten() for img in images]))
    faces = np.asarray([np.array(img).flatten() for img in images])
    # 转置，每一列代表一张人脸
    faces = faces.T

    return faces


def compute_eigenfaces(faces, ratio=0.8):
    # 计算平均脸，对每行求均值
    mean_face = np.mean(faces, axis=1)
    # 由于求完均值后，mean_face 变成了一维数组，所以需要 reshape 成列向量
    # mean_face.shape[0] 为行数，1 为列数
    mean_face = mean_face.reshape(mean_face.shape[0], 1)
    diff_faces = faces - mean_face  # 中心化

    # 协方差衡量的是不同像素位置间的相关性
    # 这里先不计算协方差矩阵 (diff_faces * diff_faces^T)，而是求 L(diff_faces^T * diff_faces) 的特征向量
    # 因为如果每张图片的像素很大，那么协方差矩阵会非常大，计算量也会非常大
    L = np.dot(diff_faces.T, diff_faces)
    # 计算 L 的特征值和特征向量，注意 L 的特征向量和原协方差矩阵的 特征值 相同
    eigen_vals, eigen_vects_L = linalg.eig(L)
    # 再转换成原协方差矩阵的特征向量
    eigen_vects = np.mat(np.dot(diff_faces, eigen_vects_L))
    # 对特征值进行从大到小的排序，返回索引值
    eigen_vals_index = argsort(-eigen_vals)

    # 使用 ratio 作为阈值选择特征向量
    selected_vects_index = []
    eigen_vals_sum = eigen_vals.sum()
    for i in range(len(eigen_vals_index)):
        selected_vals = eigen_vals[eigen_vals_index[:i]]
        ratio_sum = selected_vals.sum() / eigen_vals_sum
        if ratio_sum >= ratio:
            selected_vects_index = eigen_vals_index[:i]
            break

    # 选取特征向量，这里的特征向量是列向量，为特征脸
    eigenfaces = eigen_vects[::, selected_vects_index]
    return eigenfaces, mean_face





if __name__ == '__main__':
    # 获取图片矩阵，每一行代表一张图片
    faces = loadImageSet('img/')
    # 返回特征脸和平均脸
    eigenfaces, meanface = compute_eigenfaces(faces, ratio=0.8)
    # 显示平均脸


