import cv2 as cv
import numpy as np
import time
import math
import tools
from tools import Eigen
from tools import Tensor
from tools import VectorField

def initialization(img,sigma1, sigma2):

    img = cv.GaussianBlur(img,(15,15),sigma1)

    # # Estimate the smoothed structure tensor field
    # # sobel x
    # img_sobel_x=cv.Sobel(img,cv.CV_64F,1,0,ksize=1)
    # cv.imwrite('./output/tensors/img_sobel_x.jpg', img_sobel_x)
    # # sobel y
    # img_sobel_y=cv.Sobel(img,cv.CV_64F,0,1,ksize=1)
    # cv.imwrite('./output/tensors/img_sobel_y.jpg', img_sobel_y)
    # # BGR->LAB conversion
    # img_sobel_x_lab = cv.cvtColor(np.uint8(img_sobel_x), cv.COLOR_BGR2LAB)
    # img_sobel_y_lab = cv.cvtColor(np.uint8(img_sobel_y), cv.COLOR_BGR2LAB)

    img_lab = cv.cvtColor(np.uint8(img), cv.COLOR_BGR2LAB)
    # Estimate the smoothed structure tensor field
    # sobel x
    img_sobel_x_lab = cv.Sobel(img_lab, cv.CV_64F, 1, 0, ksize=1)
    cv.imwrite('./output/tensors/img_sobel_x.jpg', img_sobel_x_lab)
    # sobel y
    img_sobel_y_lab = cv.Sobel(img_lab, cv.CV_64F, 0, 1, ksize=1)
    cv.imwrite('./output/tensors/img_sobel_y.jpg', img_sobel_y_lab)
    # BGR->LAB conversion


    # compute eigens 
    A=img_sobel_x_lab[:,:,0]*img_sobel_x_lab[:,:,0]+img_sobel_x_lab[:,:,1]*img_sobel_x_lab[:,:,1]+img_sobel_x_lab[:,:,2]*img_sobel_x_lab[:,:,2]
    B=img_sobel_y_lab[:,:,0]*img_sobel_y_lab[:,:,0]+img_sobel_y_lab[:,:,1]*img_sobel_y_lab[:,:,1]+img_sobel_y_lab[:,:,2]*img_sobel_y_lab[:,:,2]
    C=img_sobel_x_lab[:,:,0]*img_sobel_y_lab[:,:,0]+img_sobel_x_lab[:,:,1]*img_sobel_y_lab[:,:,1]+img_sobel_x_lab[:,:,2]*img_sobel_y_lab[:,:,2]
    # blur
    A=cv.GaussianBlur(A,(15,15),sigma2)
    B=cv.GaussianBlur(B,(15,15),sigma2)
    C=cv.GaussianBlur(C,(15,15),sigma2)

    # Convert A,B,C CV_64FC1
    A=np.float64(A)
    B=np.float64(B)
    C=np.float64(C)

    return A,B,C

def computeTensors(A,B,C,p1,p2):
    print("computeTensors start")
    eigen =np.zeros(A.shape,Eigen)
    T =np.zeros(A.shape,Tensor)

    for i in range(A.shape[0]) :
        for j in range (A.shape[1]) :
            # create symetric matrix 2x2 [[A,C][C,B]]
            tmp=np.zeros((2,2),np.float64)
            tmp[0,0]=A[i,j]
            tmp[0,1]=C[i,j]
            tmp[1,0]=C[i,j]
            tmp[1,1]=B[i,j]

            # extract eigenValues and eigenVectors to compute tensor
            eigen[i,j]=Eigen(cv.eigen(tmp),p1,p2)
            T[i,j]=eigen[i,j].computeTensor()
    print("computeTensors done")
    return eigen,T

def computeVectorField(T):
    print("computeVectorField start")
    w=np.array([np.zeros(T.shape,VectorField),np.zeros(T.shape,VectorField),np.zeros(T.shape,VectorField),np.zeros(T.shape,VectorField)])
    phi=np.array([0,math.pi/4,math.pi/2,3*math.pi/4])

    for i in range(T.shape[0]) :
        for j in range (T.shape[1]) :
            for p in range(len(phi)) :
                w[p,i,j]=VectorField(T[i,j],phi[p])

    print("computeVectorField done")           
    return w

def main():
    # Parameters
    start_time = time.time()
    sigma1 = 0.5
    sigma2 = 1.2
    p1=1.2 #p1>=p2>=0
    p2=0.5
    n=1000000
    epsilon=0.2
    L=4
    # img = cv.imread('./sources/img_test.png',cv.IMREAD_COLOR) #bgr
    # img = cv.imread('./sources/joconde.png', cv.IMREAD_COLOR)  # bgr
    img = cv.imread('./sources/lena.png', cv.IMREAD_COLOR)  # bgr

    # Algo
    A,B,C=initialization(img,sigma1,sigma2)
    G,T=computeTensors(A,B,C,p1,p2) # T (tensor de trait), G (tensor de structure)
    tools.draw_ellipses_G(img, G)
    tools.draw_ellipses_T(img, T)
    w=computeVectorField(T)
    tools.draw_strokes(w,G,n,epsilon,L)
    print('time : '+str(round(time.time() - start_time))+' seconds')

if __name__ == '__main__':
    main()