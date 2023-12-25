import numpy as np
from numpy import *
from PIL import Image
import os
from matplotlib import pyplot as plt

data_path = '.\\CroppedYale'    # 图片路径
train_img_ref = 'ImgRef/TrainImage.txt'  # 存储训练集图片名称后缀
test_img_ref = 'ImgRef/TestImage.txt'    # 存储测试集图片名称后缀
image_size = (192, 168)     # 图片大小
train_img_per_person = 5    # 每个人的训练图片数
test_img_per_person = 2     # 每个人的测试图片数
ratio = 0.92    # 特征脸选取阈值，在 0~1 之间


def loadImageSet(path):
    print("=================================================")
    print("Loading images...")

    # 分别存放 待训练人脸 和 待预测人脸
    train_faces = []
    test_faces = []

    people_dirs = os.listdir(path)  # 存储 yaleBxx 目录名
    train_img_name = np.loadtxt(train_img_ref, dtype=str, comments=None)  # 读取训练图片名称后缀文件
    test_img_name = np.loadtxt(test_img_ref, dtype=str, comments=None)  # 读取测试图片名称后缀文件

    # 遍历每个目录
    for i in range(len(people_dirs)):
        people_path = os.path.join(path, people_dirs[i])
        train_count = 0
        test_count = 0

        for j in range(len(train_img_name)):
            train_img_path = os.path.join(people_path, f'{people_dirs[i]}{train_img_name[j]}')
            # 检测文件是否存在
            if not os.path.exists(train_img_path):
                print(f'Failed to load train image: {train_img_path}')
                continue
            # 使用 PIL 库中的 Image.open 函数打开每个文件
            # 并将图像转换为灰度（convert 函数，参数 L 表示灰度）
            # 然后调整图像大小为 image_size 像素（resize 函数）
            # 将每个处理过的图像对象存储在 faces 列表中。
            # ！！！注意：这里要是 resize 会出错，平均脸没法正常显示！！！
            train_img = Image.open(train_img_path).convert('L')
            # 将每个 PIL 图像对象 img 转换为 numpy 数组，并转换为一维向量
            train_img_array = np.array(train_img).flatten()
            # 将其存储在 faces 矩阵中
            train_faces.append(train_img_array)
            train_count += 1

        for k in range(len(test_img_name)):
            test_img_path = os.path.join(people_path, f'{people_dirs[i]}{test_img_name[k]}')
            if not os.path.exists(test_img_path):
                print(f'Failed to load test image: {test_img_path}')
                continue
            test_img = Image.open(test_img_path).convert('L')
            test_img_array = np.array(test_img).flatten()
            test_faces.append(test_img_array)
            test_count += 1

        # print(f'Read train img counts in directory {people_dirs[i]}: {train_count}')
        # print(f'Read test img counts in directory {people_dirs[i]}: {test_count}')

    print('Load successfully.')
    print("=================================================")

    # 将列表转换成 numpy 数组，并且转置，每一列代表一张人脸
    train_faces = np.array(train_faces).T
    test_faces = np.array(test_faces).T

    return train_faces, test_faces


def compute_eigenfaces(faces, ratio=0.8):
    print("Calculating eigenfaces...")
    eigenface_num = 0

    # 计算平均脸，对每行求均值
    mean_face = np.mean(faces, axis=1)

    # 由于求完均值后，mean_face 变成了一维数组，所以需要 reshape 成列向量
    # mean_face.shape[0] 为行数，1 为列数
    mean_face = mean_face.reshape(mean_face.shape[0], 1)

    # 中心化
    diff_faces = faces - mean_face

    # 协方差衡量的是不同像素位置间的相关性
    # 这里先不计算协方差矩阵 (diff_faces * diff_faces^T)，而是求 L(diff_faces^T * diff_faces) 的特征向量
    # 因为如果每张图片的像素很大，那么协方差矩阵会非常大，计算量也会非常大
    L = np.dot(diff_faces.T, diff_faces)
    # 计算 L 的特征值和特征向量，注意 L 和原协方差矩阵的 特征值 相同
    eigen_vals, eigen_vects_L = linalg.eig(L)
    # 再转换成原协方差矩阵的特征向量
    eigen_vects = np.mat(np.dot(diff_faces, eigen_vects_L))
    # 对特征值进行从大到小的排序，返回索引值
    eigen_vals_index = argsort(-eigen_vals)

    # 使用 ratio 作为阈值选择特征向量
    selected_vects_index = []
    eigen_vals_sum = eigen_vals.sum()
    for i in range(1, len(eigen_vals_index) + 1):
        selected_vals = eigen_vals[eigen_vals_index[:i]]
        ratio_sum = selected_vals.sum() / eigen_vals_sum
        if ratio_sum >= ratio:
            selected_vects_index = eigen_vals_index[:i]
            eigenface_num = i
            break

    # 选取出最有代表性的特征向量，这里的特征向量是列向量，为特征脸
    eigenfaces = eigen_vects[::, selected_vects_index]
    print("Calculated successfully.")
    print(f"特征脸 (eigenface) 数量：{eigenface_num}")
    print("=================================================")

    return eigenfaces, mean_face, diff_faces, eigenface_num


def predict(testfaces, eigenfaces, meanface, diff_faces):
    print("Predicting...")
    right_count = 0
    train_eigen = np.dot(eigenfaces.T, diff_faces)  # 将训练脸集投影到特征空间

    # 遍历测试脸
    for i in range(testfaces.shape[1]):
        testface = testfaces[:, i].reshape((-1, 1))  # 一列列地抽出测试集中的图片
        test_diff_face = testface - meanface
        test_eigen_image = np.dot(eigenfaces.T, test_diff_face)   # 将测试脸投影到特征空间
        distance_list = []

        # 遍历训练脸
        for j in range(train_eigen.shape[1]):
            train_eigen_image = train_eigen[:, j]
            # 计算在特征脸空间中，测试脸与各个训练脸间的欧氏距离
            distance = np.linalg.norm(test_eigen_image - train_eigen_image)
            distance_list.append(distance)

        index = np.argmin(distance_list)
        res = index // train_img_per_person + 1
        answer = i // test_img_per_person + 1
        if res == answer:
            right_count += 1
        print(f"第 {i+1:02d} 张测试脸"
              f"（来自 {i//test_img_per_person+1:02} 号）"
              f"最贴近的人是：{res:02} 号")

    correct_rate = right_count / testfaces.shape[1]
    print("Prediction finished.")
    print(f"正确率：{correct_rate:.3%}")
    print("=================================================")


if __name__ == '__main__':
    # 获取图片矩阵，每一行代表一张图片
    train_faces, test_faces = loadImageSet(data_path)

    # 返回特征脸，平均脸，中心化后的训练脸，特征脸数量
    eigenfaces, mean_face, diff_faces, eigenface_num = compute_eigenfaces(train_faces, ratio)

    # 显示平均脸
    plt.imshow(mean_face.reshape(image_size), cmap='gray')
    plt.axis('off')  # 关闭坐标轴
    plt.show()

    # 显示特征脸
    columns = int(np.ceil(np.sqrt(eigenface_num)))
    rows = int(np.ceil(eigenface_num / columns))
    fig = plt.figure()
    for i in range(eigenface_num):
        ax = fig.add_subplot(rows, columns, i+1)  # 确定子图位置
        ax.imshow(np.array(eigenfaces[:, i-1]).reshape(image_size), cmap='gray')
        ax.axis('off')
    plt.show()

    # 进行预测
    predict(test_faces, eigenfaces, mean_face, diff_faces)
