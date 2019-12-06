import cv2 as cv
import numpy as np
import blend_modes
import time

# Parameters
start_time = time.time()
sigma = 1       # standard deviation >=0
alpha=10
# Read image
img = cv.imread('./sources/lena.png',cv.IMREAD_COLOR)

# Estimate the smoothed structure tensor field
img_blur = cv.GaussianBlur(img,(5,5),alpha) # blur
cv.imwrite('./output/tensors/img_blur.jpg', img_blur)
img_sobel = cv.Sobel(img, cv.CV_64F, 1, 1, ksize=1) 
cv.imwrite('./output/tensors/img_sobel.jpg', img_sobel)
img_sobel_t=np.zeros((img.shape))
cv.transpose(img_sobel,img_sobel_t)
cv.imwrite('./output/tensors/img_sobel_t.jpg', img_sobel_t)

img_sum=np.zeros((img.shape[0],img.shape[1]))
for i in range(0,img.shape[2]) :
    img_sum+=img_sobel[:,:,i]*img_sobel_t[:,:,i]

img_smoothed=cv.GaussianBlur(img_sum,(5,5),sigma) 
cv.imwrite('./output/tensors/img_smoothed.jpg', img_smoothed)

# @TODO extract eigen values/vectors (lambda/theta)




# Compute the stroke tensor field T of the input color image I


print('time : '+str(round(time.time() - start_time))+' seconds')