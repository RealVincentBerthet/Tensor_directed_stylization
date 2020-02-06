import cv2 as cv
import numpy as np

def draw_ellipses(img, lambda_plus, lambda_moins, theta_plus, theta_moins):

    #Discretiser l'image et tracer des ellipse
    step=20
    img_ellipse=img

    lambda_plus_normalized=cv.normalize(lambda_plus, None, 0,10, norm_type=cv.NORM_MINMAX)
    lambda_moins_normalized=cv.normalize(lambda_moins, None, 0,10, norm_type=cv.NORM_MINMAX)

    for i in range(0,img_ellipse.shape[0],step) :
        for j in range(0,img_ellipse.shape[1],step) :
            axeLength = (3*lambda_moins_normalized[i,j].astype(int),3*lambda_plus_normalized[i,j].astype(int))
            # print(axeLength)
            center=(i,j)

            # axeLength = (3,3)
            vect = theta_plus
            angle = np.rad2deg(np.arccos(np.clip(np.dot(np.array([0, 1]), vect / np.linalg.norm(vect)), -1.0, 1.0)))

            cv.ellipse(img_ellipse,center,axeLength,angle,0,360,(255,0,0),-1)

    cv.imwrite('./output/tensors/img_ellipse.jpg', img_ellipse)