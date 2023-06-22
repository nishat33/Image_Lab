
import cv2

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

point_list=[]

def onClick(event):
    global x,y
    ax=event.inaxes
    
    if ax is not None:
        x,y=ax.transData.inverted().transform([event.x,event.y])
        x=int(round(x))
        y=int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %(event.button, event.x, event.y, x, y))
        point_list.append((x,y))
        

img=cv2.imread('input2.jpg',0)
cv2.imshow("input",img)

plt.title("Click the image to select the point for Hole Filling")

im=plt.imshow(img,cmap='gray')
im.figure.canvas.mpl_connect("button_press_event",onClick)
plt.show(block=True)


kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(50,50))

xk1=np.zeros_like(img)

x,y=point_list[0]
xk1[x][y]=255

while(1):
    xk=cv2.dilate(xk1,kernel,1)
    img_c=cv2.bitwise_not(img)
    xk=cv2.bitwise_and(img_c,xk)
    
    if ((xk==xk1).all()):
        break
    
    xk1=xk
    
out=cv2.bitwise_or(xk1,img)
out=cv2.normalize(out,None,0,255,cv2.NORM_MINMAX)
cv2.imshow("Output",out)
        
        

cv2.waitKey(0)
cv2.destroyAllWindows()