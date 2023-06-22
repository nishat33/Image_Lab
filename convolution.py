# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 20:43:52 2023

@author: Administrator
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread("noise.png",cv.IMREAD_GRAYSCALE)

def Gaussian_Filter(sigma,size):
    pi=3.1416
    sf=(2*pi*(sigma*sigma))
    sf=1/sf
    print(sf)    
    center=size//2
    
    kernel=np.zeros((size,size),np.float32)
    
    for i in range(-center,center+1):
        for j in range(-center, center+1):
            power=(i*i+j*j)/(sigma*sigma)
            val=np.exp(-power)
            val=sf*val
            kernel[center+i][center+j]=val
    
    print(kernel)
    return kernel
            


sigma=35
size=5

padding=(size-1)//2
center=size//2

kernel=Gaussian_Filter(sigma,size)
kernel=np.rot90(kernel,2)

img2=cv.copyMakeBorder(img,padding,padding,padding,padding, cv.BORDER_WRAP)
cv.imshow("input",img2)
output=np.zeros_like(img,np.float32)

height=img2.shape[0]
widht=img2.shape[1]
kh=kernel.shape[0]
kw=kernel.shape[1]
print(kh,kw)

for i in range(0, height-kh):
    for j in range(0 , widht-kw):
        sum=0
        for k in range(0,kh):
            for l in range(0,kw):
                val=img2[i+k][j+l]*kernel[k][l]
                sum+=val
        output.itemset((i,j),sum)
cv.normalize(output,None, 0, 255, cv.NORM_MINMAX)
cv.imshow("output Method prev",output)
output2=np.zeros_like(img2,np.float32)

#why is this one darker bruh?

img_w=widht
img_h=height      
for i in range(center,img_w-center):
    for j in range(center,img_w-center):
        sum=0
        for k in range(-center,center):
            for l in range(-center,center):
                a=img2[i+k][j+l]*kernel[center-k][center-l]
                sum+=a
        output2.itemset((i-center,j-center),sum)
cv.normalize(output2,None, 0, 255, cv.NORM_MINMAX)
cv.imshow("output method current",output2)





cv.waitKey(0)
cv.destroyAllWindows()


