
import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)

def min_max_normal(img):
    mi=np.min(img)
    ma=np.max(img)
    output=np.zeros((img.shape[0],img.shape[1]),np.uint8)
    
    for i in range(0,img.shape[0]):
        for j in range(0, img.shape[1]):
            temp=(img[i][j]-mi)/(ma-mi)
            output[i][j]=temp*255
    return output

def histogram_eq(img):
    
    total=img.shape[0]*img.shape[1]
    height=img.shape[0]
    width=img.shape[1]
    pdf=np.zeros(256)
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            pdf[img[i][j]]+=1

    total=pdf.sum()

    pdf=pdf/total
    
    cdf=pdf
    s=pdf
    sum2=0

    for i in range(256):
        sum2=sum2+pdf[i]
        cdf[i]=sum2
        p=cdf[i]*255
        s[i]=round(p)
    
    plt.title("Cdf:")
    plt.plot(cdf)
    plt.show()
    
    eq=np.zeros((img.shape[0],img.shape[1]))
    
    for i in range(height):
        for j in range(width):
            eq[i][j]=s[img[i][j]]
    
    eq=min_max_normal(eq)
    cv2.imshow("output",eq)
    plt.title("Input Image Histogram")
    plt.hist(img.ravel(),255,[0,255])
    plt.show()
    
    plt.title("Output image histogram")
    plt.hist(eq.ravel(),255,[0,255])
    plt.show()
    
    return eq

out=histogram_eq(img)
eq=histogram_eq(out)

cv2.waitKey(0)
cv2.destroyAllWindows()

    
        
    
    
            
