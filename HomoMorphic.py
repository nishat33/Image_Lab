# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 02:59:29 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 05:21:14 2023

@author: Administrator
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy as dpc

point_list=[]
def min_max_normalize(image):
    h = image.shape[0]
    w = image.shape[1]
    min = np.min(image)
    max = np.max(image)
    output = np.zeros((image.shape), np.uint8)

    for i in range(0, h):
        for j in range(0, w):
            temp = ((image[i][j]-min)/(max-min))*255
            output[i][j] = temp
    return output


def illumination(img):
    height=img.shape[0]
    width=img.shape[1]
    
    x=np.linspace(255, 0, width)
    y=np.linspace(255, 0, height)
    
    xx,yy=np.meshgrid(x,y)
    
    mask=np.clip((xx+yy)/2, 0, 255).astype(np.uint8)
    
    return mask

img1=cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow("original input",img1)
img2=illumination(img1)
cv2.imshow("illumination pattern",img2)
img3=cv2.add(img1,img2)
cv2.imshow("Corrupted input",img3)

img_h=img3.shape[0]
img_w=img3.shape[1]
kernel=np.zeros((img_h,img_w),np.float32)

def homo_morphic_filter():
    sigma = 50.0
    GH = 1.2
    GL = 0.5
    const = 0.3
    D0 = 2 * np.pi * sigma**2
    for i in range(0,img_h):
        for j in range(0,img_w):
            
            u = (i - img_h//2)**2
            v = (j - img_w//2)**2
            r = np.exp(-((const*(u+v))/(2*D0**2)))
            r = (GH-GL) *(1-r) + GL
            kernel[i][j] = r
            
    
    
     
    return kernel


kernel=homo_morphic_filter()

img1 = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

img = dpc(img1)

log_image = np.log1p(img)

spectrum = np.fft.fft2(log_image)

spectrum_shifted = np.fft.fftshift(spectrum)

mag=np.abs(spectrum_shifted)
ang=np.angle(spectrum_shifted)

mag_plot=np.log1p(mag)
mag_plot=min_max_normalize(mag_plot)
cv2.imshow("magnitude of corrpted", mag_plot)

out=mag*kernel
out=np.multiply(out,np.exp(1j*ang))
image=(np.fft.ifft2(np.fft.ifftshift(out)))
corrected_image=np.expm1(np.abs(image))
corrected_image=min_max_normalize(image)

cv2.imshow("output",corrected_image)

        
    

cv2.waitKey(0)
cv2.destroyAllWindows()


