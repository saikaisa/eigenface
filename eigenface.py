import numpy as np
from numpy import *
from numpy import linalg as la
from PIL import Image
import glob
from matplotlib import pyplot as plt


def loadImageSet(add):
    filenames = glob.glob('face/pgm/*.pgm') # 匹配指定路径下的所有以 .pgm 结尾的文件，并将文件路径存储在 filenames 列表中
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

    return faces


def recogInitVector(selecthr=0.8):
    # 步骤1：加载人脸图像数据，获取包含所有图像的矩阵
    FaceMat = loadImageSet('face/yalefaces/')
    print('-----------FaceMat.shape--------')
    print(FaceMat.shape)

    # 步骤2：计算FaceMat的平均图像
    avgImg = mean(FaceMat, 0)

    # 步骤3：计算平均图像（avgImg）与所有图像数据（FaceMat）之间的差异
    diffTrain = FaceMat - avgImg
    covMat = np.asmatrix(diffTrain) * np.asmatrix(diffTrain.T)
    eigvals, eigVects = linalg.eig(covMat)  # la.linalg.eig(np.mat(covMat))

    # 步骤4：计算协方差矩阵的特征向量（由于协方差矩阵可能会导致内存错误）
    eigSortIndex = argsort(-eigvals)
    for i in range(shape(FaceMat)[1]):
        if (eigvals[eigSortIndex[:i]] / eigvals.sum()).sum() >= selecthr:
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = diffTrain.T * eigVects[:, eigSortIndex]  # covVects是协方差矩阵的特征向量

    # avgImg 是平均图像，covVects是协方差矩阵的特征向量，diffTrain是偏差矩阵
    return avgImg, covVects, diffTrain


def judgeFace(judgeImg, FaceVector, avgImg, diffTrain):
    # 判断
    diff = judgeImg - avgImg
    weiVec = FaceVector.T * diff.T
    res = 0
    resVal = inf

    # ==============================================================================

    # plt.imshow(avgImg.reshape(98, 116))

    # plt.show()

    # ==============================================================================

    for i in range(15):
        TrainVec = (diffTrain[i] * FaceVector).T
        if (array(weiVec - TrainVec) ** 2).sum() < resVal:
            res = i
            resVal = (array(weiVec - TrainVec) ** 2).sum()
    return res + 1


if __name__ == '__main__':
    avgImg, FaceVector, diffTrain = recogInitVector(selecthr=0.8)
    nameList = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
    characteristic = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'rightlight', 'sad', 'sleepy',
                      'surprised', 'wink']
    for c in characteristic:
        count = 0
        for i in range(len(nameList)):
            # 这是我们要识别的未知人脸图像。我们将其与相应的训练人脸进行比较，以计算准确率。
            loadname = 'face/yalefaces/subject' + nameList[i] + '.' + c + '.pgm'
            judgeImg = Image.open(loadname).convert('L').resize((98, 116))
            # print(loadname)
            if judgeFace(mat(judgeImg).flatten(), FaceVector, avgImg, diffTrain) == int(nameList[i]):
                count += 1
        print('准确率：%s：%f' % (c, float(count) / len(nameList)))  # 计算准确率