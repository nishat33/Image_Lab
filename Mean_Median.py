# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 02:19:08 2023

@author: Administrator
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


img=cv2.imread("noise.png",cv2.IMREAD_GRAYSCALE)
cv2.imshow("input",img)
output=np.zeros_like(img, np.uint8)
kernel=np.ones((3,3),np.float32)
total=kernel.sum()
print(total)
kw,kh=kernel.shape
p_h=(kh-1)//2
p_w=(kw-1)//2
img2=cv2.copyMakeBorder(img,p_w,p_w,p_h,p_h,cv2.BORDER_WRAP)

for i in range ( img2.shape[0]-kernel.shape[0]):
    for j in range( img2.shape[1]-kernel.shape[1]):
        sum=0
        for k in range(kernel.shape[0]):
            for l in range(kernel.shape[1]):
                sum+=img2[i+k][j+l]*kernel[k][l]
        output[i][j]=sum/total
cv2.imshow("output of Mean filter", output)

output=np.zeros_like(img, np.uint8)

for i in range ( img2.shape[0]-kernel.shape[0]):
    for j in range( img2.shape[1]-kernel.shape[1]):
        array=[]
        for k in range(kernel.shape[0]):
            for l in range(kernel.shape[1]):
                a=img2[i+k][j+l]*kernel[k][l]
                array.append(a)
        array.sort()
        val=len(array)//2
        output[i][j]=array[val]
cv2.imshow("output of Median filter", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
        