# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 23:05:55 2023

@author: Administrator
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def twirl(img, center , alpha, rmax):
    img=img.astype(np.float32)/255.0 #but ken?
    
    height,width=img.shape
    
    y,x=np.mgrid[0:height, 0:width]
    
    x-=center[0]
    y-=center[1]
    
    distance=x**2+y**2
    distance=np.sqrt(distance)
    
    beta=np.arctan2(y,x)+ alpha*((rmax-distance)/rmax)
    
    x2=center[0]+ distance*np.cos(beta)
    y2=center[1]+ distance*np.sin(beta)
    
    mask = distance > rmax
    
    x2[mask]= x[mask] + center[0]
    y2[mask]= y[mask] + center[1]
    
    output = cv2.remap(img, x2.astype(np.float32), y2.astype(np.float32), interpolation= cv2.INTER_LINEAR)
    
    output= (output*255).astype(np.uint8)
    
    return output

img=cv2.imread("tapestry.png", cv2.IMREAD_GRAYSCALE)

out=twirl(img, (150,150), np.radians(90), 300)
    
cv2.imshow("input",img)
cv2.imshow("output",out)
    
    
cv2.waitKey(0)
cv2.destroyAllWindows()