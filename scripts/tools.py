import numpy as np
import cv2 as cv
import numpy as np
import math
import scipy.linalg

class VectorField:
    def __init__(self,T,phy):
        self.aphy=np.array([math.cos(phy),math.sin(phy)])
        self.vector=T.sqrt()@self.aphy

    def getX(self):
        return self.vector[0]
    
    def getY(self):
        return self.vector[1]

class Tensor:
    def __init__(self, tensor, c_plus, c_moins, thetaPlus, thetaMoins):
        self.tensor = tensor
        self.c_plus = c_plus
        self.c_moins = c_moins
        self.thetaPlus = thetaPlus
        self.thetaMoins = thetaMoins

    def getThetaPlus(self):
        return self.thetaPlus

    def getThetaMoins(self):
        return self.thetaMoins

    def getCPlus(self):
        return self.c_plus

    def getCMoins(self):
        return self.c_moins
    
    def sqrt(self):
        return scipy.linalg.sqrtm(self.tensor)


class Eigen:
    def __init__(self, eigen, p1, p2):
        self.eigen = eigen
        self.p1 = p1
        self.p2 = p2

    def getLambda(self):
        return self.eigen[1]

    def getLambdaPlus(self):
        return self.eigen[1][0]

    def getLambdaMoins(self):
        return self.eigen[1][1]

    def getTheta(self):
        return self.eigen[2]

    def getThetaPlus(self):
        return self.eigen[2][0]

    def getThetaMoins(self):
        return self.eigen[2][1]

    def getCPlus(self):
        c_plus = 1 / np.power(1 + self.getLambdaPlus() + self.getLambdaMoins(), self.p1)
        return c_plus

    def getCMoins(self):
        c_moins = 1 / np.power(1 + self.getLambdaPlus() + self.getLambdaMoins(), self.p2)
        return c_moins

    def computeTensor(self):
        T = self.getCPlus() * np.array([self.getThetaPlus()]).T @ np.array(
            [self.getThetaPlus()]) + self.getCMoins() * np.array([self.getThetaMoins()]).T @ np.array(
            [self.getThetaMoins()])
        return Tensor(T,self.getCPlus(),self.getCMoins(),self.getThetaPlus(),self.getThetaMoins())

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def draw_ellipses_G(img, eigen):

    #Discretiser l'image et tracer des ellipses
    step=20
    img_ellipse=img.copy()
    #print(img.shape)
    lambda_plus_matrix = np.zeros(eigen.shape)
    lambda_moins_matrix = np.zeros(eigen.shape)
    for i in range(eigen.shape[0]):
        for j in range(eigen.shape[1]):
            lambda_plus_matrix[i][j] = eigen[i][j].getLambdaPlus()
            lambda_moins_matrix[i][j] = (eigen[i][j].getLambdaMoins())

    lambda_plus_normalized=cv.normalize(lambda_plus_matrix, None, 0,40, norm_type=cv.NORM_MINMAX)
    lambda_moins_normalized=cv.normalize(lambda_moins_matrix, None, 0,40, norm_type=cv.NORM_MINMAX)


    for i in range(0,img_ellipse.shape[0],step) :
        for j in range(0,img_ellipse.shape[1],step) :

            lambda_plus = np.around(lambda_plus_normalized[i][j]).astype(int)
            lambda_moins = np.around(lambda_moins_normalized[i][j]).astype(int)
            theta_plus = eigen[i][j].getThetaPlus()
            theta_moins = eigen[i][j].getThetaMoins()

            axeLength = (lambda_plus,lambda_moins)
            # print(axeLength)
            center=(j,i)

            angle = getAngle((0, 1), (0, 0), theta_moins)
            #print(theta_moins)
            #print(angle)
            cv.ellipse(img_ellipse,center,axeLength,angle,0,360,(255,0,0),-1)

    cv.imwrite('./output/tensors/img_ellipse_G.jpg', img_ellipse)


def draw_ellipses_T(img, eigen):

    #Discretiser l'image et tracer des ellipses
    step=20
    img_ellipse_T=img.copy()
    lambda_plus_matrix = np.zeros(eigen.shape)
    lambda_moins_matrix = np.zeros(eigen.shape)
    for i in range(eigen.shape[0]):
        for j in range(eigen.shape[1]):
            lambda_plus_matrix[i][j] = eigen[i][j].getCPlus()
            lambda_moins_matrix[i][j] = (eigen[i][j].getCMoins())

    lambda_plus_normalized=cv.normalize(lambda_plus_matrix, None, 0,20, norm_type=cv.NORM_MINMAX)
    lambda_moins_normalized=cv.normalize(lambda_moins_matrix, None, 0,20, norm_type=cv.NORM_MINMAX)


    for i in range(0,img_ellipse_T.shape[0],step) :
        for j in range(0,img_ellipse_T.shape[1],step) :

            lambda_plus = np.around(lambda_plus_normalized[i][j]).astype(int)
            lambda_moins = np.around(lambda_moins_normalized[i][j]).astype(int)
            theta_plus = eigen[i][j].getThetaPlus()
            theta_moins = eigen[i][j].getThetaMoins()

            axeLength = (lambda_plus,lambda_moins)
            # print(axeLength)
            center=(j,i)

            angle = getAngle((0, 1), (0, 0), theta_moins)
            #print(theta_moins)
            #print(angle)
            cv.ellipse(img_ellipse_T,center,axeLength,angle,0,360,(255,0,0),-1)

    cv.imwrite('./output/tensors/img_ellipse_T.jpg', img_ellipse_T)


def draw_strokes(w,G,n,epsilon,L):
    print("draw_strokes start")
    img_strokes = np.zeros((w.shape[1],w.shape[2]), np.uint8)
    img_strokes[:,:] = 255

    for counter in range(n):
        x=np.random.randint(0,img_strokes.shape[0])
        y=np.random.randint(0,img_strokes.shape[1])

        if math.sqrt(G[x,y].getLambdaPlus()+G[x,y].getLambdaMoins()) > epsilon :
            for p in range(w.shape[0]) : #nb phy
                tmp = np.zeros((img_strokes.shape[0],img_strokes.shape[1]), np.uint8)
                uv=[L*w[p,x,y].getX(),L*w[p,x,y].getY()]
                cv.line(tmp, (y - int(uv[0]), x - int(uv[1])), (y + int(uv[0]), x + int(uv[1])), 10, 1)
                img_strokes[:,:] = cv.subtract(img_strokes[:,:], tmp[:,:])
        else :
            counter=counter-1


        if round(((counter-1)/n)*100) != round((counter/n)*100) :
            print(str(round((counter/n)*100))+' %')
    
    cv.imwrite('./output/tensors/img_results.jpg', img_strokes)
    print("draw_strokes done")    


