import numpy as np
from numpy import *
from PIL import Image
import os
from matplotlib import pyplot as plt

data_path = '.\\CroppedYale'    
train_img_ref = 'ImgRef/TrainImage.txt'  
test_img_ref = 'ImgRef/TestImage.txt'    
image_size = (192, 168)     
train_img_per_person = 5    
test_img_per_person = 2     
ratio = 0.92    


def loadImageSet(path):
    print("=================================================")
    print("Loading images...")

    train_faces = []
    test_faces = []

    people_dirs = os.listdir(path)  
    train_img_name = np.loadtxt(train_img_ref, dtype=str, comments=None)  
    test_img_name = np.loadtxt(test_img_ref, dtype=str, comments=None)
    
    for i in range(len(people_dirs)):
        people_path = os.path.join(path, people_dirs[i])
        train_count = 0
        test_count = 0

        for j in range(len(train_img_name)):
            train_img_path = os.path.join(people_path, f'{people_dirs[i]}{train_img_name[j]}')
            
            if not os.path.exists(train_img_path):
                print(f'Failed to load train image: {train_img_path}')
                continue
            
            train_img = Image.open(train_img_path).convert('L')
            train_img_array = np.array(train_img).flatten()
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

    print('Load successfully.')
    print("=================================================")
    
    train_faces = np.array(train_faces).T
    test_faces = np.array(test_faces).T

    return train_faces, test_faces


def compute_eigenfaces(faces, ratio=0.8):
    print("Calculating eigenfaces...")
    eigenface_num = 0

    mean_face = np.mean(faces, axis=1)
    mean_face = mean_face.reshape(mean_face.shape[0], 1)
    diff_faces = faces - mean_face

    L = np.dot(diff_faces.T, diff_faces)
    eigen_vals, eigen_vects_L = linalg.eig(L)
    eigen_vects = np.mat(np.dot(diff_faces, eigen_vects_L))
    eigen_vals_index = argsort(-eigen_vals)

    selected_vects_index = []
    eigen_vals_sum = eigen_vals.sum()
    for i in range(1, len(eigen_vals_index) + 1):
        selected_vals = eigen_vals[eigen_vals_index[:i]]
        ratio_sum = selected_vals.sum() / eigen_vals_sum
        if ratio_sum >= ratio:
            selected_vects_index = eigen_vals_index[:i]
            eigenface_num = i
            break

    eigenfaces = eigen_vects[::, selected_vects_index]
    print("Calculated successfully.")
    print(f"特征脸 (eigenface) 数量：{eigenface_num}")
    print("=================================================")

    return eigenfaces, mean_face, diff_faces, eigenface_num


def predict(testfaces, eigenfaces, meanface, diff_faces):
    print("Predicting...")
    right_count = 0
    train_eigen = np.dot(eigenfaces.T, diff_faces)  

    for i in range(testfaces.shape[1]):
        testface = testfaces[:, i].reshape((-1, 1))  
        test_diff_face = testface - meanface
        test_eigen_image = np.dot(eigenfaces.T, test_diff_face)   
        distance_list = []
        
        for j in range(train_eigen.shape[1]):
            train_eigen_image = train_eigen[:, j]
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
    train_faces, test_faces = loadImageSet(data_path)
    eigenfaces, mean_face, diff_faces, eigenface_num = compute_eigenfaces(train_faces, ratio)

    plt.imshow(mean_face.reshape(image_size), cmap='gray')
    plt.axis('off')  
    plt.show()
    
    columns = int(np.ceil(np.sqrt(eigenface_num)))
    rows = int(np.ceil(eigenface_num / columns))
    fig = plt.figure()
    for i in range(eigenface_num):
        ax = fig.add_subplot(rows, columns, i+1)  
        ax.imshow(np.array(eigenfaces[:, i-1]).reshape(image_size), cmap='gray')
        ax.axis('off')
    plt.show()
    
    predict(test_faces, eigenfaces, mean_face, diff_faces)
