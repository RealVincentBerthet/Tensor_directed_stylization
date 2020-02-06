import numpy as np
import cv2 as cv
import numpy as np

class Tensor:
    def __init__(self, tensor):
        self.tensor = tensor

class Eigen:
    def __init__(self,eigen):
        self.eigen = eigen
    
    def getLambda(self):
        return self.eigen[1]

    def getLambdaPlus(self):
        return self.eigen[1][1]
    
    def getLambdaMoins(self):
        return self.eigen[1][0]

    def getTheta(self):
        return self.eigen[2]

    def getThetaPlus(self):
        return self.eigen[2][0]
    
    def getThetaMoins(self):
        return self.eigen[2][1]

def draw_ellipses(img, eigen):

    #Discretiser l'image et tracer des ellipse
    step=20
    img_ellipse=img

    lambda_plus_matrix = np.zeros(eigen.shape)
    lambda_moins_matrix = np.zeros(eigen.shape)
    for i in range(eigen.shape[0]):
        for j in range(eigen.shape[1]):
            lambda_plus_matrix[i][j] = eigen[i][j].getLambdaPlus()
            lambda_moins_matrix[i][j] = eigen[i][j].getLambdaMoins()

    lambda_plus_normalized=cv.normalize(lambda_plus_matrix, None, 0,10, norm_type=cv.NORM_MINMAX)
    lambda_moins_normalized=cv.normalize(lambda_moins_matrix, None, 0,10, norm_type=cv.NORM_MINMAX)


    for i in range(0,img_ellipse.shape[0],step) :
        for j in range(0,img_ellipse.shape[1],step) :

            lambda_plus = 3 * lambda_plus_normalized[i][j].astype(int)
            lambda_moins = 3 * lambda_moins_normalized[i][j].astype(int)
            theta_plus = eigen[i][j].getThetaPlus()
            theta_moins = eigen[i][j].getThetaMoins()

            axeLength = (lambda_moins,lambda_plus)
            # print(axeLength)
            center=(i,j)

            vect = theta_plus
            angle = np.rad2deg(np.arccos(np.clip(np.dot(np.array([0, 1]), vect / np.linalg.norm(vect)), -1.0, 1.0)))

            cv.ellipse(img_ellipse,center,axeLength,angle,0,360,(255,0,0),-1)

    cv.imwrite('./output/tensors/img_ellipse.jpg', img_ellipse)