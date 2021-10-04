import os
import cv2

img = cv2.imread("../levindata/05/02_k.png",cv2.IMREAD_GRAYSCALE)
img = (img - img.min()) /(img.max() - img.min())
for col in range(0,img.shape[0]):
    for row in range(0,img.shape[1]):
        if img[col,row] != 0.0:
            img[col,row] = img[col,row] * 255
cv2.imwrite("022_k.png",img)