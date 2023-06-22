import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def min_max_normalization(image):
    h = image.shape[0]
    w = image.shape[1]
    mi = np.min(image)
    ma = np.max(image)
    out = np.zeros((image.shape), np.uint8)
    for i in range(0, h):
        for j in range(0, w):
            out[i][j] = ((image[i][j] - mi) / (ma - mi)) * 255
    return out


image = cv2.imread('eye.png', cv2.IMREAD_GRAYSCALE)

filter = np.ones((5, 5), np.int8)
filter = filter / 25

# border will be changed according to filter, for 5x5 kernel and 1,1 filter, left and top padding will be 3,
# and right and bottom filter will be 1
image2 = cv2.copyMakeBorder(image, 3, 1, 3, 1, cv2.BORDER_CONSTANT)
topleft = 3
bottomright = 1
print(image2.shape)

output = np.zeros((image.shape), np.float32)

# the loop will start so that 0,0 of kernel is set on 0,0 of the padded image
for i in range(topleft, image2.shape[0] - bottomright):
    for j in range(topleft, image2.shape[1] - bottomright):
        temp = 0
        for k in range(-topleft, bottomright + 1):
            for l in range(-topleft, bottomright + 1):
                temp += filter[k + topleft, l + topleft] * image2[i + k, j + l]
        output[i - topleft, j - topleft] = temp

output = min_max_normalization(output)

cv2.imshow('input', image)
cv2.imshow('2nd image', image2)
cv2.imshow('output', output)

cv2.waitKey(0)
cv2.destroyAllWindows()