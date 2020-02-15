import cv2 as cv
import numpy as np
import argparse
import math
import tensorsTools
from tensorsTools import Tensor
from tensorsTools import VectorField
from tensorsTools import Bar

def get_args(sigma1,sigma2,p1,p2,n,epsilon,L,coeff,output):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--image", help="input image")
    parser.add_argument("-s1","--sigma1", help="sigma 1")
    parser.add_argument("-s2","--sigma2", help="sigma 2")
    parser.add_argument("-p1","--power1", help="power 1")
    parser.add_argument("-p2","--power2", help="power 2")
    parser.add_argument("-n","--number", help="number of strokes")
    parser.add_argument("-e","--epsilon", help="epsilon to draw strokes")
    parser.add_argument("-l","--length", help="Length of strokes")
    parser.add_argument("-c","--coefficient", help="coefficient to reduce the image")
    parser.add_argument("-o","--output", help="custom output under ./output/tensors directory")
    args = parser.parse_args()

    if args.image != None:
        print("Image Loaded : "+str(args.image))
    else:
        print("No image loaded used -i argurment")
        quit()

    if args.sigma1 != None :
        sigma1=float(args.sigma1)
    if args.sigma2 != None :
        sigma2=float(args.sigma2)
    if args.power1 != None :
        p1=float(args.power1)
    if args.power2 != None :
        p2=float(args.power2)
    if args.number != None :
        n=int(args.number)
    if args.epsilon != None :
        epsilon=float(args.epsilon)
    if args.length != None :
        L=float(args.length)
    if args.coefficient != None :
        coeff=float(args.coefficient)
    if args.output != None :
        output=str(args.output)

    if p1<p2 or p1<0 :
        print("You should have p1>=p2>=0")
        quit()

    return args.image,sigma1,sigma2,p1,p2,n,epsilon,L,coeff,output
    
def initialization(img,sigma):
    img = cv.GaussianBlur(img,(15,15),sigma)
    img_lab = cv.cvtColor(np.uint8(img), cv.COLOR_BGR2LAB)
    # Estimate the smoothed structure tensor field
    # sobel x
    img_sobel_x_lab = cv.Sobel(img_lab, cv.CV_64F, 1, 0, ksize=1)
    # sobel y
    img_sobel_y_lab = cv.Sobel(img_lab, cv.CV_64F, 0, 1, ksize=1)
 
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
    bar=tensorsTools.Bar("Compute Tensors",A.shape[0]*A.shape[1])
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
            bar.next()
    return T

def computeVectorField(T):
    gamma=np.array([0,math.pi/4,math.pi/2,3*math.pi/4])
    #gamma = np.array([0, math.pi / 2])
    w = []
    bar=tensorsTools.Bar("Compute VectorField",T.shape[0]*T.shape[1]*len(gamma))
    for i in range(len(gamma)):
        w.append(np.zeros(T.shape, VectorField))
    w = np.array(w)

    for i in range(T.shape[0]) :
        for j in range (T.shape[1]) :
            for p in range(len(gamma)) :
                w[p,i,j]=VectorField(T[i,j],gamma[p])
                bar.next()

    return w

def main():
    # Parameters
    time = tensorsTools.Timer()
    sigma1 = 0.5
    sigma2 = 1.2
    p1=1.2 #p1>=p2>=0
    p2=0.5
    n=100000
    epsilon=1
    L=80
    coeff=1
    output=""

    # Initialize input image
    img_path,sigma1,sigma2,p1,p2,n,epsilon,L,coeff,output=get_args(sigma1,sigma2,p1,p2,n,epsilon,L,coeff,output)
    img = cv.imread(str(img_path),cv.IMREAD_COLOR) #bgr
    size = (int(img.shape[1]/coeff), int(img.shape[0]/coeff))
    img = cv.resize(img, size)

    # Algo
    img_sobel_x_lab,img_sobel_y_lab=initialization(img,sigma1)
    A,B,C=computeEigen(img_sobel_x_lab,img_sobel_y_lab,sigma2)
    T=computeTensors(A,B,C,p1,p2) 
    tensorsTools.draw_ellipses_G(img, T,output=output) #structure
    tensorsTools.draw_ellipses_T(img, T,output=output) #trait
    w=computeVectorField(T)
    tensorsTools.draw_strokes(img, w,T,n,epsilon,L,output=output)
    print('Time : '+str(time))

if __name__ == '__main__':
    main()