import cv2 as cv
import numpy as np
import time

## Parameters
start_time = time.time()
sigma = 1    
## Read image
img = cv.imread('./sources/lena.png',cv.IMREAD_COLOR) #bgr

## Estimate the smoothed structure tensor field
#sobel x
img_sobel_x=cv.Sobel(img,cv.CV_64F,1,0,ksize=1) 
cv.imwrite('./output/tensors/img_sobel_x.jpg', img_sobel_x)
#sobel y
img_sobel_y=cv.Sobel(img,cv.CV_64F,0,1,ksize=1) 
cv.imwrite('./output/tensors/img_sobel_y.jpg', img_sobel_y)
#BGR->LAB conversion
img_sobel_x_lab = cv.cvtColor(np.uint8(img_sobel_x), cv.COLOR_BGR2LAB)
img_sobel_y_lab = cv.cvtColor(np.uint8(img_sobel_y), cv.COLOR_BGR2LAB)
#compute eigen 
A=img_sobel_x_lab[:,:,0]*img_sobel_x_lab[:,:,0]+img_sobel_x_lab[:,:,1]*img_sobel_x_lab[:,:,1]+img_sobel_x_lab[:,:,2]*img_sobel_x_lab[:,:,2]
B=img_sobel_y_lab[:,:,0]*img_sobel_y_lab[:,:,0]+img_sobel_y_lab[:,:,1]*img_sobel_y_lab[:,:,1]+img_sobel_y_lab[:,:,2]*img_sobel_y_lab[:,:,2]
C=img_sobel_x_lab[:,:,0]*img_sobel_y_lab[:,:,0]+img_sobel_x_lab[:,:,1]*img_sobel_y_lab[:,:,1]+img_sobel_x_lab[:,:,2]*img_sobel_y_lab[:,:,2]
#blur
A=cv.GaussianBlur(A,(5,5),sigma)
B=cv.GaussianBlur(B,(5,5),sigma)
C=cv.GaussianBlur(C,(5,5),sigma)


# @TODO extract lambda +/- for each pixel ?
print(A.shape)





# Compute the stroke tensor field T of the input color image I


print('time : '+str(round(time.time() - start_time))+' seconds')