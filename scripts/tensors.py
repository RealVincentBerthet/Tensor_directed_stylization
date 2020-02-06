import cv2 as cv
import numpy as np
import time
import tools
from tools import Eigen
from tools import Tensor

def initialization(img,sigma):
    # Estimate the smoothed structure tensor field
    # sobel x
    img_sobel_x=cv.Sobel(img,cv.CV_64F,1,0,ksize=1) 
    cv.imwrite('./output/tensors/img_sobel_x.jpg', img_sobel_x)
    # sobel y
    img_sobel_y=cv.Sobel(img,cv.CV_64F,0,1,ksize=1) 
    cv.imwrite('./output/tensors/img_sobel_y.jpg', img_sobel_y)
    # BGR->LAB conversion
    img_sobel_x_lab = cv.cvtColor(np.uint8(img_sobel_x), cv.COLOR_BGR2LAB)
    img_sobel_y_lab = cv.cvtColor(np.uint8(img_sobel_y), cv.COLOR_BGR2LAB)

    # compute eigens 
    A=img_sobel_x_lab[:,:,0]*img_sobel_x_lab[:,:,0]+img_sobel_x_lab[:,:,1]*img_sobel_x_lab[:,:,1]+img_sobel_x_lab[:,:,2]*img_sobel_x_lab[:,:,2]
    B=img_sobel_y_lab[:,:,0]*img_sobel_y_lab[:,:,0]+img_sobel_y_lab[:,:,1]*img_sobel_y_lab[:,:,1]+img_sobel_y_lab[:,:,2]*img_sobel_y_lab[:,:,2]
    C=img_sobel_x_lab[:,:,0]*img_sobel_y_lab[:,:,0]+img_sobel_x_lab[:,:,1]*img_sobel_y_lab[:,:,1]+img_sobel_x_lab[:,:,2]*img_sobel_y_lab[:,:,2]
    # blur
    A=cv.GaussianBlur(A,(5,5),sigma)
    B=cv.GaussianBlur(B,(5,5),sigma)
    C=cv.GaussianBlur(C,(5,5),sigma)

    # Convert A,B,C CV_64FC1
    A=np.float64(A)
    B=np.float64(B)
    C=np.float64(C)

    return A,B,C

def computeTensors(A,B,C,p1,p2):
    eigen =np.array(np.zeros(A.shape,Eigen))
    T =np.array(np.zeros(A.shape,Tensor))

    for i in range(A.shape[0]) :
        for j in range (A.shape[1]) :
            #create symetric matrix 2x2 [[A,C][C,B]]
            tmp=np.zeros((2,2),np.float64)
            tmp[0,0]=A[i,j]
            tmp[0,1]=C[i,j]
            tmp[1,0]=C[i,j]
            tmp[1,1]=B[i,j]

            #extract eigenValues and eigenVectors
            e=Eigen(cv.eigen(tmp))
            eigen[i,j]=e

            #compute Tensor
            
            c_plus=1/np.power(1+e.getLambdaPlus()+e.getLambdaMoins(),p1) 
            c_moins=1/np.power(1+e.getLambdaPlus()+e.getLambdaMoins(),p2)
            res = c_plus*np.array([e.getThetaPlus()]).T @ np.array([e.getThetaPlus()]) + c_moins*np.array([e.getThetaMoins()]).T @ np.array([e.getThetaMoins()])

            T[i,j]=Tensor(res)

    return eigen,T

def main():
    # Parameters
    start_time = time.time()
    sigma = 1   
    p1=2 #p1>=p2>=0
    p2=1 
    img = cv.imread('./sources/lena.png',cv.IMREAD_COLOR) #bgr

    # Algo
    A,B,C=initialization(img,sigma)
    G,T=computeTensors(A,B,C,p1,p2) # T (tensor de trait), G (tensor de structure)

    print('time : '+str(round(time.time() - start_time))+' seconds')

if __name__ == '__main__':
    main()