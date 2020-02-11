import cv2 as cv
import numpy as np
import argparse
import time
import math
import tensorsTools
from tensorsTools import Tensor
from tensorsTools import VectorField

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--image", help="intput image")
    args = parser.parse_args()

    if args.image != None:
        print("Image Loaded : "+str(args.image))
    else:
        print("No image loaded used -i argurment")
        quit()

    return args.image
    
def initialization(img,sigma):
    img = cv.GaussianBlur(img,(15,15),sigma)
    img_lab = cv.cvtColor(np.uint8(img), cv.COLOR_BGR2LAB)
    # Estimate the smoothed structure tensor field
    # sobel x
    img_sobel_x_lab = cv.Sobel(img_lab, cv.CV_64F, 1, 0, ksize=1)
    cv.imwrite('./output/tensors/img_sobel_x.jpg', img_sobel_x_lab)
    # sobel y
    img_sobel_y_lab = cv.Sobel(img_lab, cv.CV_64F, 0, 1, ksize=1)
    cv.imwrite('./output/tensors/img_sobel_y.jpg', img_sobel_y_lab)
 
    return img_sobel_x_lab,img_sobel_y_lab

def computeEigen(img_sobel_x_lab,img_sobel_y_lab,sigma):
    # compute eigens 
    A=img_sobel_x_lab[:,:,0]*img_sobel_x_lab[:,:,0]+img_sobel_x_lab[:,:,1]*img_sobel_x_lab[:,:,1]+img_sobel_x_lab[:,:,2]*img_sobel_x_lab[:,:,2]
    B=img_sobel_y_lab[:,:,0]*img_sobel_y_lab[:,:,0]+img_sobel_y_lab[:,:,1]*img_sobel_y_lab[:,:,1]+img_sobel_y_lab[:,:,2]*img_sobel_y_lab[:,:,2]
    C=img_sobel_x_lab[:,:,0]*img_sobel_y_lab[:,:,0]+img_sobel_x_lab[:,:,1]*img_sobel_y_lab[:,:,1]+img_sobel_x_lab[:,:,2]*img_sobel_y_lab[:,:,2]
    # blur
    A=cv.GaussianBlur(A,(15,15),sigma)
    B=cv.GaussianBlur(B,(15,15),sigma)
    C=cv.GaussianBlur(C,(15,15),sigma)
    # Convert A,B,C CV_64FC1
    A=np.float64(A)
    B=np.float64(B)
    C=np.float64(C)

    return A,B,C

def computeTensors(A,B,C,p1,p2):
    print("computeTensors start")
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
            T[i,j]=Tensor(cv.eigen(tmp),p1,p2)

    print("computeTensors done")
    return T

def computeVectorField(T):
    print("computeVectorField start")

    # phi=np.array([0,math.pi/4,math.pi/2,3*math.pi/4])
    phi = np.array([0, math.pi / 2])
    w = []
    for i in range(len(phi)):
        w.append(np.zeros(T.shape, VectorField))
    w = np.array(w)

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
    n=100000
    epsilon=1
    L=80
    img_path=get_args()
    img = cv.imread(str(img_path),cv.IMREAD_COLOR) #bgr
    coeff = 2
    size = (int(img.shape[1]/coeff), int(img.shape[0]/coeff))
    img = cv.resize(img, size)

    # Algo
    img_sobel_x_lab,img_sobel_y_lab=initialization(img,sigma1)
    A,B,C=computeEigen(img_sobel_x_lab,img_sobel_y_lab,sigma2)
    T=computeTensors(A,B,C,p1,p2) 
    tensorsTools.draw_ellipses_G(img, T) #structure
    tensorsTools.draw_ellipses_T(img, T) #trait
    w=computeVectorField(T)
    tensorsTools.draw_strokes(img, w,T,n,epsilon,L)
    print('time : '+str(round(time.time() - start_time))+' seconds')

if __name__ == '__main__':
    main()