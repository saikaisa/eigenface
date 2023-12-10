import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import fetch_lfw_people
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import glob
import cv2


# 调整图片大小,并且调整为灰度
def resize_images(images, new_size):
    resized_images = [cv2.resize(img, new_size, interpolation=cv2.INTER_AREA) for img in images]
    return resized_images


# 计算平均,特征脸
def compute_eigenfaces(images, num_components):
    # 将图片拉成一维度
    X = np.array([img.flatten() for img in images], dtype=np.float64)
    mean_face = np.mean(images, axis=0)

    # 中心化
    X -= mean_face.flatten()

    # svd分解
    U, S, Vt = np.linalg.svd(X, full_matrices=False)  # U：特征向量矩阵，S：对角矩阵，包含了奇异值，Vt：另一个特征向量矩阵的转置

    # 提取特征脸,使用右奇异矩阵
    eigenfaces = Vt[:num_components, :]
    return mean_face, eigenfaces


# 绘制图像
def plot_faces(faces, titles, num_rows, num_cols):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# 从lfw数据库获取人脸数据, 每个人至少 min_faces_per_person 张图片
images = fetch_lfw_people(min_faces_per_person=100, resize=0.4).images

images = resize_images(images, (100, 100))

# 计算平均脸和特征脸
mean_face, eigenfaces = compute_eigenfaces(images, num_components=15)

# 绘制平均脸和前15个特征脸
faces_to_plot = [mean_face] + [eigenface.reshape(mean_face.shape) for eigenface in eigenfaces]
titles = ['Mean Face'] + [f'Eigenface {i + 1}' for i in range(15)]
plot_faces(faces_to_plot, titles, num_rows=4, num_cols=4)