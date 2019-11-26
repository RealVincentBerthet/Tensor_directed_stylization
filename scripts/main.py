import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt

# Read image
img = cv.imread('./../sources/lena.png',cv.IMREAD_COLOR)
img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
img_y = img_ycrcb[:,:,0]

# Gaussian filter
gaussian = cv.GaussianBlur(img_y,(5,5),0)
cv.imwrite("../output/gaussian.jpg", gaussian)

# Gradient
sobel = cv.Sobel(gaussian, cv.CV_64F, 1, 1, ksize=1)
cv.imwrite("../output/sobel.jpg", sobel)

# Initialize result image
height, width = sobel.shape
result = np.zeros((height,width), np.uint8)
result[:,:] = 255

tmp = np.zeros((height,width), np.uint8)

# Random position
cpt = 0

for i in range(sobel.shape[0]):
    for j in range(sobel.shape[1]):
        if sobel[i,j]>2:
            u = random.randrange(9)
            v = random.randrange(9)
            tmp[:,:] = 0
            cv.line(tmp, (j - u, i - v), (j + u, i + v), 230, 1)
            result+=tmp
            cpt+=1

print(cpt)
cv.imwrite("../output/result.jpg", result)


# Colorisation
print(img.shape)
result3d = img
print(result.shape)
gaussiancolor = cv.GaussianBlur(img,(35,35),0)
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        if result[i,j]<255:
            gaussiancolor[i,j,0] =0
            gaussiancolor[i, j, 1] =0
            gaussiancolor[i, j, 2] =0
cv.imwrite("../output/res_color.jpg", gaussiancolor)



cv.waitKey(0)
cv.destroyAllWindows()