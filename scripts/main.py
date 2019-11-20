import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Read image
img = cv.imread('./../sources/lena.png',cv.IMREAD_COLOR)
img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
img_y = img_ycrcb[:,:,0]

# Gaussian filter TO DO
#blur = cv.blur(img_y,(5,5))

# Gradient
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=1)

cv.imshow('sobelx',sobelx)
cv.imwrite("../output/sobelx.jpg", sobelx)
cv.waitKey(0)
cv.destroyAllWindows()