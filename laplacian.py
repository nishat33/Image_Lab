# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 00:04:32 2023

@author: Administrator
"""
import cv2 
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread('moon.png',cv2.IMREAD_GRAYSCALE)
kernel=np.array(([-1,-1,-1],
                 [-1,8,-1],
                 [-1,-1,-1]),np.float32)
kernel=np.rot90(kernel,2)

kh=kernel.shape[0]
kw=kernel.shape[1]
padding=(kh-1)//2

center_h=kh//2
center_w=kw//2

image2=cv2.copyMakeBorder(img, padding,padding, padding, padding, cv2.BORDER_WRAP)

cv2.imshow("input", image2)
img_width=image2.shape[0]
img_height=image2.shape[1]

output=np.zeros_like(img,np.float32)
  
print(img_height,img_width)
for i in range(0, img_width-kw):
    for j in range(0, img_height-kh):
        sum=0
        for k in range(0,kw):
            for l in range(0,kh):
                a=image2[i+k][j+l]*kernel[k][l]
                sum+=a
        output.itemset((i,j),sum)

cv2.normalize(output,None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("output",output)




cv2.waitKey(0)
cv2.destroyAllWindows()
