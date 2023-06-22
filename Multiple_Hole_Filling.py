import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

point_list=[]
def onclick(event):
    global x, y
    ax = event.inaxes
    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, x, y))
        point_list.append((x,y))

img = cv2.imread('img2.jpg', 0)
cv2.imshow("Original", img)


plt.title("Please select seed pixel from the input")
im = plt.imshow(img, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

print(point_list)
kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) #cv2.MORPH_RECT for all 1s

print(point_list)
xk1=np.zeros_like(img)

for item in (point_list):
    
    x,y=item
    xk1[y][x]=255
    

while(1):
    xk=cv2.dilate(xk1,kernel1,iterations = 1)
    comp=cv2.bitwise_not(img)
    xk=cv2.bitwise_and(xk,comp)
    
    if((xk==xk1).all()):
        break
    xk1=xk
out=cv2.bitwise_or(xk,img)
img=out

   
    
out=cv2.normalize(out,None,0,255,cv2.NORM_MINMAX)
cv2.imshow('Output',out)
    
    

cv2.waitKey(0)
cv2.destroyAllWindows()