import cv2
import numpy as np

def get_patches(x,overlap):
    if x.max() > 1:
        x = (x - x.min()) / (x.max() - x.min())
    w = int(x.shape[1]/2)
    h = int(x.shape[0]/2)
    p0 = x[0:h,0:w+overlap,:]
    p1 = x[h-overlap:,0:w,:]
    p2 = x[0:h+overlap,w:,:]
    p3 = x[h:,w-overlap:,:]
    return [p0,p1,p2,p3]

x = cv2.imread("02_x.png")
x = cv2.resize(x,(256,256))
print(x.shape)
a = get_patches(x,25)

cv2.imwrite("01.png",a[0]*255)
cv2.imwrite("02.png",a[1] * 255)
cv2.imwrite("03.png",a[2] * 255)
cv2.imwrite("04.png",a[3] * 255)
