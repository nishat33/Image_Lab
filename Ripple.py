# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 00:14:31 2023

@author: Administrator
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

def Ripple(img, Ax, Ay, Tx, Ty):
    
    img=img.astype(np.float32)/255.0
    
    width=img.shape[0]
    height=img.shape[1]
    
    y,x=np.mgrid[0:height, 0:width]
    
    x2=x+Ax*np.sin((2*np.pi*y)/Tx)
    y2=y+Ay*np.sin((2*np.pi*x)/Ty)
    
    output=cv2.remap(img, x2.astype(np.float32), y2.astype(np.float32), interpolation= cv2.INTER_LINEAR)
    return output

img=cv2.imread("rubiks_cube.png")
out=Ripple(img, 10, 15, 50, 70)

cv2.imshow("output",out)

cv2.waitKey(0)
cv2.destroyAllWindows()