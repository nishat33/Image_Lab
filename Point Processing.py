# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:52:54 2023

@author: Administrator
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread("ace.jpg",cv2.IMREAD_GRAYSCALE)

def Power_Law(img,c,lamda):
    img_w,img_h=img.shape
    
    out=np.zeros_like(img, np.uint8)
    
    for i in range(0,img_w):
        for j in range(0, img_h):
            d=pow(img[i][j],lamda)
            out[i][j]=d*c
    
    
    cv2.imshow("output of Gamma Translation",out)

def Contras_Stretching(img,low,high):
    img_w,img_h=img.shape
    out=np.zeros_like(img)
    for i in range(img_w):
        for j in range(img_h):
            val=(img[i][j]-low)/(high-low)
            out[i][j]=val*255
    cv2.normalize(out,None,low,high,cv2.NORM_MINMAX)
    cv2.imshow("output of contrast stretching",out)

def Inverse_log_Translation(img,c):
    
    img_w,img_h=img.shape
    
    output=np.zeros_like(img,np.uint8)
    
    for i in range(0,img_w):
        for j in range(0, img_h):
            val=img[i][j]//c 
            val=pow(2,(1+val)) - 1
            output[i][j]=val
    
    #cv2.normalize(output,None,0.255,cv2.NORM_MINMAX)
    
    cv2.imshow("Inverse Log Translation",output)

    
g=0.5
c=1/pow(255,g-1)

Power_Law(img, c, g )
Inverse_log_Translation(img, c)
Contras_Stretching(img, 20, 100)

    
cv2.waitKey(0)
cv2.destroyAllWindows() 
            
