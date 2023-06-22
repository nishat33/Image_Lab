# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 23:42:27 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 23:05:55 2023

@author: Administrator
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def Angular(img, center , amplitude, frequency,rmax):
    img=img.astype(np.float32)/255.0 #but ken?
    
    width,height=img.shape
    
    y,x=np.mgrid[0:height, 0:width]
    
    x-=center[0]
    y-=center[0]
    r=np.sqrt(x**2+y**2)
    
    distance=amplitude*(np.sin((2*np.pi*r)/frequency))
    
    beta=np.arctan2(y,x)+ distance
    
    x2=center[0]+ r*np.cos(beta)
    y2=center[1]+ r*np.sin(beta)
    
    mask = distance > rmax
    
    x2[mask]= x[mask] + center[0]
    y2[mask]= y[mask] + center[1]
    
    output = cv2.remap(img, x2.astype(np.float32), y2.astype(np.float32), interpolation= cv2.INTER_LINEAR)
    
    output= (output*255).astype(np.uint8)
    
    return output

img=cv2.imread("angular.jpg", cv2.IMREAD_GRAYSCALE)
center_w=img.shape[1]//2
center_h=img.shape[0]//2
out=Angular(img, (center_h,center_w), 0.1, 50, 200)
    
cv2.imshow("input",img)
cv2.imshow("output",out)
    
    
cv2.waitKey(0)
cv2.destroyAllWindows()