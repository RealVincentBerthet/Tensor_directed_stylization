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
    def __init__(self, eigen, p1, p2):
        self.eigen = eigen
        self.p1 = p1
        self.p2 = p2
        self.tensor=self.getCPlus() * np.array([self.getThetaPlus()]).T @ np.array([self.getThetaPlus()]) + self.getCMoins() * np.array([self.getThetaMoins()]).T @ np.array([self.getThetaMoins()])

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

    def getTensor(self):
        return self.tensor
    
    def sqrt(self):
        return scipy.linalg.sqrtm(self.tensor)

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang

def draw_ellipses_G(img, T):
    #Discrete image and draw elipse for structure tensor
    step=20
    img_ellipse=img.copy()
    lambda_plus_matrix = np.zeros(T.shape)
    lambda_moins_matrix = np.zeros(T.shape)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            lambda_plus_matrix[i][j] = T[i][j].getLambdaPlus()
            lambda_moins_matrix[i][j] = T[i][j].getLambdaMoins()

    lambda_plus_normalized=cv.normalize(lambda_plus_matrix, None, 0,40, norm_type=cv.NORM_MINMAX)
    lambda_moins_normalized=cv.normalize(lambda_moins_matrix, None, 0,40, norm_type=cv.NORM_MINMAX)

    for i in range(0,img_ellipse.shape[0],step) :
        for j in range(0,img_ellipse.shape[1],step) :
            lambda_plus = np.around(lambda_plus_normalized[i][j]).astype(int)
            lambda_moins = np.around(lambda_moins_normalized[i][j]).astype(int)
            theta_plus = T[i][j].getThetaPlus()
            theta_moins = T[i][j].getThetaMoins()

            axeLength = (lambda_plus,lambda_moins)
            center=(j,i)

            angle = getAngle((0, 1), (0, 0), theta_moins)
            cv.ellipse(img_ellipse,center,axeLength,angle,0,360,(255,0,0),-1)

    cv.imwrite('./output/tensors/img_ellipse_G.jpg', img_ellipse)

def draw_ellipses_T(img, T):
    #Discrete image and draw ellipse for stroke tensor
    step=20
    img_ellipse_T=img.copy()
    c_plus_matrix = np.zeros(T.shape)
    c_moins_matrix = np.zeros(T.shape)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            c_plus_matrix[i][j] = T[i][j].getCPlus()
            c_moins_matrix[i][j] = T[i][j].getCMoins()

    c_plus_normalized=cv.normalize(c_plus_matrix, None, 0,20, norm_type=cv.NORM_MINMAX)
    c_moins_normalized=cv.normalize(c_moins_matrix, None, 0,20, norm_type=cv.NORM_MINMAX)

    for i in range(0,img_ellipse_T.shape[0],step) :
        for j in range(0,img_ellipse_T.shape[1],step) :
            c_plus = np.around(c_plus_normalized[i][j]).astype(int)
            c_moins = np.around(c_moins_normalized[i][j]).astype(int)
            theta_plus = T[i][j].getThetaPlus()
            theta_moins = T[i][j].getThetaMoins()

            axeLength = (c_plus,c_moins)
            center=(j,i)

            angle = getAngle((0, 1), (0, 0), theta_moins)
            cv.ellipse(img_ellipse_T,center,axeLength,angle,0,360,(255,0,0),-1)

    cv.imwrite('./output/tensors/img_ellipse_T.jpg', img_ellipse_T)

def draw_strokes(img, w,T,n,epsilon,L):
    print("draw_strokes start")
    img_strokes = np.zeros((w.shape[1],w.shape[2],3), np.uint8)
    img_strokes[:,:,:] = 255

    for counter in range(n):
        x=np.random.randint(0,img_strokes.shape[0])
        y=np.random.randint(0,img_strokes.shape[1])

        if math.sqrt(T[x,y].getLambdaPlus()+T[x,y].getLambdaMoins()) > epsilon :
            for p in range(w.shape[0]) : #nb phy
                tmp = np.zeros((img_strokes.shape[0],img_strokes.shape[1],3), np.uint8)
                uv=[L*w[p,x,y].getX(),L*w[p,x,y].getY()]
                coeff = 25
                b = int((255-img[x,y][0])/coeff)
                g = int((255-img[x,y][1])/coeff)
                r = int((255-img[x,y][2])/coeff)
                cv.line(tmp, (y - int(uv[0]), x - int(uv[1])), (y + int(uv[0]), x + int(uv[1])), (b, g, r) , 1)
                img_strokes[:,:,:] = cv.subtract(img_strokes[:,:,:], tmp[:,:,:])
        else :
            counter=counter-1

        if round(((counter-1)/n)*100) != round((counter/n)*100) :
            print(str(round((counter/n)*100))+' %')
    
    cv.imwrite('./output/tensors/img_results.jpg', img_strokes)
    print("draw_strokes done")    


