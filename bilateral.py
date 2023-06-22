# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 01:47:15 2023

@author: Administrator
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def Gaussian_spatial(size,sigma):
    center=size//2
    c=1/(2*np.pi*pow(sigma, 2))
    kernel=np.zeros((size,size),np.float32)
    for i in range(-center,center+1):
        for j in range(-center,center+1):
            num=-(i*i+j*j)/(pow(sigma,2))
            num=np.exp(num)
            num=c*num
            kernel[center+i][center+j]=num
    return kernel

img=cv2.imread("noise.png",cv2.IMREAD_GRAYSCALE)

size=5
padding=size//2
center=padding

kw=size
kh=size
img2=cv2.copyMakeBorder(img,padding,padding,padding,padding,cv2.BORDER_WRAP)

img_w,img_h=img2.shape

print(img_w,img_h)

sigma=10
center=size//2
c=1/(2*np.pi*pow(sigma, 2))
kernel=np.zeros((size,size),np.float32)

output=np.zeros_like(img,np.uint8)

print(output.shape)
Gaussian_Kernel=Gaussian_spatial(5, 5)


for i in range(center,img_w-center):
    for j in range(center, img_h-center):
        for k in range(-center,center+1):
            for l in range(-center,center+1):
                
                intensity=img2.item(i,j)-img2.item(i+k,j+l)
                intensity=intensity**2
                val=intensity/(2*(sigma**2))
                val=np.exp(-val)
                kernel[k][l]=val
        sum=0
        kernel=kernel*Gaussian_Kernel
        total=kernel.sum()
        for k in range(-center,center+1):
            for l in range(-center,center+1):
                a=kernel[k-center][l-center]*img2[i+k][j+l]
                sum+=a
        sum=sum/total
        output.itemset((i-center,j-center),sum)
        
cv2.normalize(output,None,0,255,cv2.NORM_MINMAX)
print(output)
cv2.imshow("input",img)
cv2.imshow("output",output)
cv2.waitKey(0)
cv2.destroyAllWindows()
            
    