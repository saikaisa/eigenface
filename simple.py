import cv2
import numpy as np

images=[]
images.append(cv2.imread("a1.png",0))
images.append(cv2.imread("a2.png",0))
images.append(cv2.imread("b1.png",0))
images.append(cv2.imread("b2.png",0))

# 四张图片分别对应的标签
labels = [0,0,1,1]
# recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.train(images,np.array(labels))

predict_img = cv2.imread("a5.png",0)
label,confidence = recognizer.predict(predict_img)
print("label:",label)
print("confidence:",confidence)