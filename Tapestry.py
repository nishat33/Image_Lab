# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 23:53:39 2023

@author: Administrator
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

def Tapestry(img,amplitude, Tx, Ty,center):
    img=img.astype(np.float32)/255.0
    
    width=img.shape[1]
    height=img.shape[0]
    
    y,x=np.mgrid[0:height,0:width]
    
    x2=x+amplitude*(np.sin(2*np.pi*(x-center[0])/Tx))
    
    y2=y+amplitude*(np.sin(2*np.pi*(y-center[1])/Ty))
    
    output=cv2.remap(img, x2.astype(np.float32), y2.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    output= (output*255).astype(np.uint8)
    return output




img=cv2.imread("tapestry.png")
cv2.imshow("input",img)
out=Tapestry(img,5,30,30, (150,150))
cv2.imshow("Tapestry output", out)


cv2.waitKey(0)
cv2.destroyAllWindows()


    
    