import cv2 as cv
import numpy as np

# Parameters
L = 4   # Length of a strokes
sigma_gaussian = 10     # standard deviation >=0
epsilon = 2     # level >=0
# Read image
img = cv.imread('./../sources/lena.png',cv.IMREAD_COLOR)
img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
img_y = img_ycrcb[:,:,0]

# Gaussian filter
gaussian = cv.GaussianBlur(img_y,(5,5),sigma_gaussian)
cv.imwrite("../output/gaussian.jpg", gaussian)

# Gradient
sobel = cv.Sobel(gaussian, cv.CV_64F, 1, 1, ksize=1)
sobelx = cv.Sobel(gaussian, cv.CV_64F, 1, 0, ksize=1)
sobely = cv.Sobel(gaussian, cv.CV_64F, 0, 1, ksize=1)
cv.imwrite("../output/sobel.jpg", sobel)

# Initialize result image
height, width = sobel.shape
result = np.zeros((height,width), np.uint8)
result[:,:] = 255

# Random position
cpt = 0
tmp = np.zeros((height,width), np.uint8)
for i in range(sobel.shape[0]):
    for j in range(sobel.shape[1]):
        if sobel[i,j]>epsilon:
            uv = [L/2 *sobely[i,j]/sobel[i,j], L/2 *-sobelx[i,j]/sobel[i,j]]
            tmp[:,:] = 0
            cv.line(tmp, (j - int(uv[0]), i - int(uv[1])), (j + int(uv[0]), i + int(uv[1])), 230, 1)
            result+=tmp
            cpt+=1

print("number of strokes : "+ str(cpt))
cv.imwrite("../output/res_greyscale.jpg", result)


# Colorisation
gaussiancolor = cv.GaussianBlur(img,(35,35),0)
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        if result[i,j]<255:
            gaussiancolor[i,j,0] =0
            gaussiancolor[i, j, 1] =0
            gaussiancolor[i, j, 2] =0
cv.imwrite("../output/res_color.jpg", gaussiancolor)
