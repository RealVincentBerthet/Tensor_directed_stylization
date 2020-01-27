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

#compute eigens 
A=img_sobel_x_lab[:,:,0]*img_sobel_x_lab[:,:,0]+img_sobel_x_lab[:,:,1]*img_sobel_x_lab[:,:,1]+img_sobel_x_lab[:,:,2]*img_sobel_x_lab[:,:,2]
B=img_sobel_y_lab[:,:,0]*img_sobel_y_lab[:,:,0]+img_sobel_y_lab[:,:,1]*img_sobel_y_lab[:,:,1]+img_sobel_y_lab[:,:,2]*img_sobel_y_lab[:,:,2]
C=img_sobel_x_lab[:,:,0]*img_sobel_y_lab[:,:,0]+img_sobel_x_lab[:,:,1]*img_sobel_y_lab[:,:,1]+img_sobel_x_lab[:,:,2]*img_sobel_y_lab[:,:,2]
#blur
A=cv.GaussianBlur(A,(5,5),sigma)
B=cv.GaussianBlur(B,(5,5),sigma)
C=cv.GaussianBlur(C,(5,5),sigma)

#Convert A,B,C CV_32FC1
A=np.float64(A)
B=np.float64(B)
C=np.float64(C)

#1. Compute Tensors from eigenValues, eigenVectors
p1=2 #p1>=p2>=0
p2=1
T=np.zeros(A.shape,np.float64)
for i in range(img.shape[1]) :
    for j in range (img.shape[2]) :
        #create symetric matrix 2x2 [[A,C][C,B]]
        tmp=np.zeros((2,2),np.float64)
        tmp[0,0]=A[i,j]
        tmp[0,1]=C[i,j]
        tmp[1,0]=C[i,j]
        tmp[1,1]=B[i,j]

        #extract eigenValues and eigenVectors
        eigen=cv.eigen(tmp)
        lambda_plus=eigen[1][0]
        lambda_moins=eigen[1][1]
        theta_plus=eigen[2][0]
        theta_moins=eigen[2][1]

        #compute Tensor
        c_plus=1/np.power(1+lambda_plus+lambda_moins,p1) 
        c_moins=1/np.power(1+lambda_plus+lambda_moins,2)
        AVERIF=c_plus*theta_plus*theta_plus.transpose()+c_moins*theta_moins*theta_moins.transpose()
        T[i,j]=AVERIF[0]


#2. Decomposition champs de vecteurs
phi=0
angle=np.array([np.cos(phi),np.sin(phi)],np.float64)

#np.multiply(np.sqrt(T),angles) #@TODO 



# normalizedImg= cv.normalize(T, None, 0,255, norm_type=cv.NORM_MINMAX)
# cv.imshow('dst_rt', normalizedImg)
# cv.waitKey(0)

print('time : '+str(round(time.time() - start_time))+' seconds')