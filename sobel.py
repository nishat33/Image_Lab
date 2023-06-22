# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 01:50:37 2023

@author: Administrator
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread("moon.png",cv2.IMREAD_GRAYSCALE)

kernel=np.array(([-1,0,1],
                 [-1,0,1],
                 [-1,0,1]),np.float32)

kernel2=np.array(([-1,-1,-1],
                 [0,0,0],
                 [1,1,1]),np.float32)

kw,kh=kernel.shape

p_h=(kh-1)//2
p_w=(kw-1)//2

img2=cv2.copyMakeBorder(img,p_w,p_w,p_h,p_h,cv2.BORDER_WRAP)

img_w,img_h=img2.shape

cv2.imshow("input",img2)

kernel=np.rot90(kernel,2)
kernel2=np.rot90(kernel2,2)

print(kernel)
output=np.zeros_like(img,np.float32)
output2=np.zeros_like(img,np.float32)

for i in range(0,img_w-kw):
    for j in range(0, img_h-kh):
        temp=0
        temp2=0
        for k in range(0,kw):
            for l in range(0, kh):
                temp+=img2[i+k][j+l]*kernel[k][l]
                temp2+=img2[i+k][j+l]*kernel2[k][l]
                
        output[i][j]=temp
        output2[i][j]=temp2


cv2.normalize(output,None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("output",output)


cv2.normalize(output2,None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("output2",output2)



cv2.waitKey(0)
cv2.destroyAllWindows()