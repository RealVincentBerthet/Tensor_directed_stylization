import cv2 as cv
import numpy as np
import time

# Parameters
start_time = time.time()
sigma = 1       # standard deviation >=0
alpha=10
# Read image
img = cv.imread('./sources/lena.png',cv.IMREAD_COLOR) #bgr

# Estimate the smoothed structure tensor field
img_blur = cv.GaussianBlur(img,(5,5),alpha) # blur
cv.imwrite('./output/tensors/img_blur.jpg', img_blur) 
#sobel x
img_sobel_x=cv.Sobel(img,cv.CV_64F,1,0,ksize=1) 
cv.imwrite('./output/tensors/img_sobel_x.jpg', img_sobel_x)
#sobel y
img_sobel_y=cv.Sobel(img,cv.CV_64F,0,1,ksize=1) 
cv.imwrite('./output/tensors/img_sobel_y.jpg', img_sobel_y)

#BGR->LAB conversion
img_sobel_x_lab=np.zeros((img.shape[0],img.shape[1]))
img_sobel_y_lab=np.zeros((img.shape[0],img.shape[1]))
print(type(img_sobel_x))
cv.cvtColor(img_sobel_x,img_sobel_x_lab,cv.COLOR_BGR2Lab)
cv.cvtColor(img_sobel_y,img_sobel_y_lab,cv.COLOR_BGR2Lab)

# @TODO extract eigen values/vectors (lambda/theta)
#img_lab=np.zeros((img.shape[0],img.shape[1]))
#cv.cvtColor(img_smoothed,img_lab,cv.COLOR_BGR2Lab)




# Compute the stroke tensor field T of the input color image I


print('time : '+str(round(time.time() - start_time))+' seconds')