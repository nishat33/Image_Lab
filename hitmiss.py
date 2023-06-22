# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:06:07 2023

@author: Administrator
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt

img=cv2.imread("hitmiss.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("input",img)

r, img = cv2.threshold(img, 100, 200 ,cv2.THRESH_BINARY) #4

rate=50

kernel1=np.array(([1,1,1],
                  [1,1,1],
                  [1,1,1]),np.uint8)

kernel2=np.array(([0,1,1],
                  [0,0,1],
                  [0,0,1]),np.uint8)

kernel3=np.array(([0,0,0],
                  [1,1,0],
                  [1,0,0]),np.uint8)

kernel4=np.array(([1,1,1],
                  [0,1,0],
                  [0,1,0]),np.uint8)


K= cv2.resize(kernel1, None, fx=rate, fy=rate , interpolation=cv2.INTER_NEAREST)
k2= cv2.resize(kernel2, None, fx=rate, fy=rate , interpolation=cv2.INTER_NEAREST)
k3= cv2.resize(kernel3, None, fx=rate, fy=rate , interpolation=cv2.INTER_NEAREST)
k4= cv2.resize(kernel4, None, fx=rate, fy=rate , interpolation=cv2.INTER_NEAREST)

K=K*255
k2=k2*255
k3=k3*255
k4=k4*255

def hit_miss(img,K,kernel1):
    img_c=cv2.bitwise_not(img)
    a=cv2.erode(img,kernel1,1)
    kernel2=K-kernel1
    b=cv2.erode(img_c,kernel2,1)
    
    out=cv2.bitwise_and(a,b)
    return out

out1=cv2.dilate(hit_miss(img, K, k2),K,1)
out2=cv2.dilate(hit_miss(img, K, k3),K,1)
out3=cv2.dilate(hit_miss(img, K, k4),K,1)

cv2.imshow("First",out1)
cv2.imshow("Second",out2)
cv2.imshow("Third",out3)
    







cv2.waitKey(0)
cv2.destroyAllWindows()
