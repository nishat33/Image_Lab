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
def min_max_normalize(img):
    Minim=np.min(img)
    Maxim=np.max(img)
    
    output=np.zeros_like(img,np.uint8)
    for i in range(0,img.shape[0]):
        for j in range(0, img.shape[1]):
            output[i][j]=((img[i][j]-Minim)/(Maxim-Minim))*255
    return output

def onClick(event):
    global x,y
    
    ax=event.inaxes
    if ax is not None:
        x,y=ax.transData.inverted().transform([event.x,event.y])
        x=int(round(x))
        y=int(round(y))
        point_list.append((x,y))

img_input=cv2.imread('duckNoise.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow("input",img_input)




print(point_list)

img_h=img_input.shape[0]
img_w=img_input.shape[1]
kernel=np.zeros_like((img_input),np.float32)
def butter_worth_notch_filter(D0,n):
    for i in range(img_h):
        for j in range(img_w):
            Du=(i-point_list[0][1])**2
            Dv=(i-point_list[0][0])**2
            
            Dk=np.sqrt(Du+Dv)
            
            _Du=(i+point_list[0][1])**2
            _Dv=(i+point_list[0][0])**2
            
            _Dk=np.sqrt(_Du+_Dv)
            
            Dp=1+(D0/Dk)**(2*n)
            _Dp=1+(D0/_Dk)**(2*n)
            
            Dh=(1/Dp)*(1/_Dp)
            kernel[i][j]=Dh
    return kernel
    



img = dpc(img_input)

image_size = img.shape[0] * img.shape[1]

ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac=np.abs(ft_shift)
magnitude_spectrum = np.log1p(magnitude_spectrum_ac)
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

plt.title("Click here")
im=plt.imshow(magnitude_spectrum_scaled,cmap='gray')
im.figure.canvas.mpl_connect("button_press_event",onClick)
plt.show(block=True)
print(point_list)

kernel=butter_worth_notch_filter(5,2)
kernel=min_max_normalize(kernel)
ang = np.angle(ft_shift)


ans_magnitude=np.multiply(magnitude_spectrum_ac ,kernel)


#add phase
final_result = np.multiply(ans_magnitude, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)



## plot
cv2.imshow("filter",kernel)
cv2.imshow("input", img)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum_scaled)

cv2.imshow("Phase",ang)
cv2.imshow("Inverse transform",img_back_scaled)
        
    

cv2.waitKey(0)
cv2.destroyAllWindows()


