import numpy as np
import cv2 as cv
import numpy as np
import math
import scipy.linalg
import progressbar
import os
import time

class VectorField:
    def __init__(self,T,gamma):
        self.agamma=np.array([math.cos(gamma),math.sin(gamma)])
        self.vector=T.sqrt()@self.agamma

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

class Bar:
    def __init__(self, title,size):
        self.size=size
        self.increment=0
        self.bar = progressbar.ProgressBar(maxval=self.size, \
        widgets=[str(title),' ', progressbar.Bar(), ' ', progressbar.Percentage(),' [',progressbar.Counter(),'/',str(size),'] - ',progressbar.Timer()])
        self.bar.start()
    
    def next(self):
        self.increment=self.increment+1
        if self.increment>=self.size:
            self.bar.finish()
        else:
            self.bar.update(self.increment)

class Timer:
    def __init__(self):
        self.time=time.time()
    def start(self):
        self.time=time.time()
    def getTime(self):
        return self.time
    def setTime(self,time):
        self.time=time
    def __str__(self):
        temp=round(time.time() - self.time)
        hours = temp//3600
        temp = temp - 3600*hours
        minutes = temp//60
        seconds = temp - 60*minutes

        h=''
        m=''
        s=''
        if hours > 0 and hours <10:
            h='0'+str(hours)+'h'
        elif hours>9:
            h=str(hours)+'h'

        if minutes > 0 and minutes <10 and hours>0:
            m='0'+str(minutes)+'min'
        elif minutes>9:
            m=str(minutes)+'min'

        if seconds > 0 and seconds <10 and minutes>0:
            s='0'+str(seconds)+'s'
        elif seconds>9:
            s=str(seconds)+'s'
        t=str(h)+str(m)+str(s)

        return str(t)

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang

def draw_ellipses_G(img, T,alpha=0.5,output=""):
    #Discrete image and draw elipse for structure tensor
    step=20
    img_ellipse=img.copy()
    overlay = img.copy()
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
            color=(255,0,0)

            # if (angle > 0 and angle <= 45.0) or (angle > 180 and angle <= 225) :
            #     color=(255,0,0)
            # elif (angle >45 and angle <= 90.0) or (angle > 225 and angle <= 270) :
            #     color=(0,255,0)
            # elif (angle >90 and angle <= 135) or (angle>270 and angle <= 315) :
            #     color=(0,0,255)
            # elif (angle > 135 and angle <=180) or angle>315 :
            #     color=(0,255,255)
            
            cv.ellipse(overlay,center,axeLength,angle,0,360,color,-1)

    img_ellipse = cv.addWeighted(overlay, alpha, img_ellipse, 1 - alpha, 0)

    if not os.path.exists('./output/tensors/'+os.path.dirname(output)):
        os.makedirs('./output/tensors/'+os.path.dirname(output))
    cv.imwrite('./output/tensors/'+str(output)+'img_ellipse_G.jpg', img_ellipse)

def draw_ellipses_T(img, T,alpha=0.5,output=""):
    #Discrete image and draw ellipse for stroke tensor
    step=20
    img_ellipse=img.copy()
    overlay = img.copy()
    c_plus_matrix = np.zeros(T.shape)
    c_moins_matrix = np.zeros(T.shape)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            c_plus_matrix[i][j] = T[i][j].getCPlus()
            c_moins_matrix[i][j] = T[i][j].getCMoins()

    c_plus_normalized=cv.normalize(c_plus_matrix, None, 0,20, norm_type=cv.NORM_MINMAX)
    c_moins_normalized=cv.normalize(c_moins_matrix, None, 0,20, norm_type=cv.NORM_MINMAX)

    for i in range(0,img_ellipse.shape[0],step) :
        for j in range(0,img_ellipse.shape[1],step) :
            c_plus = np.around(c_plus_normalized[i][j]).astype(int)
            c_moins = np.around(c_moins_normalized[i][j]).astype(int)
            theta_plus = T[i][j].getThetaPlus()
            theta_moins = T[i][j].getThetaMoins()

            axeLength = (c_plus,c_moins)
            center=(j,i)

            angle = getAngle((0, 1), (0, 0), theta_moins)
            cv.ellipse(overlay,center,axeLength,angle,0,360,(255,0,0),-1)

    img_ellipse = cv.addWeighted(overlay, alpha, img_ellipse, 1 - alpha, 0)
    if not os.path.exists('./output/tensors/'+os.path.dirname(output)):
        os.makedirs('./output/tensors/'+os.path.dirname(output))
    cv.imwrite('./output/tensors/'+str(output)+'img_ellipse_T.jpg', img_ellipse)

def draw_strokes(img, w,T,n,epsilon,L,output="",gray=False):
    bar=Bar("Draw strokes",n)
    src=img.copy()
    if gray==True:
        src=cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        img_strokes=np.zeros((w.shape[1],w.shape[2]), np.uint8)
        img_strokes[:,:] = 255
    else :
        img_strokes = np.zeros((w.shape[1],w.shape[2],3), np.uint8)
        img_strokes[:,:,:] = 255
 
    for counter in range(n):
        x=np.random.randint(0,img_strokes.shape[0])
        y=np.random.randint(0,img_strokes.shape[1])

        if math.sqrt(T[x,y].getLambdaPlus()+T[x,y].getLambdaMoins()) > epsilon :
            for p in range(w.shape[0]) : #nb phy

                uv=[L*w[p,x,y].getX(),L*w[p,x,y].getY()]
                coeff = 25
                if gray == True :
                    tmp = np.zeros((img_strokes.shape[0],img_strokes.shape[1]), np.uint8)
                    c = int((255-src[x,y])/coeff)
                    cv.line(tmp, (y - int(uv[0]), x - int(uv[1])), (y + int(uv[0]), x + int(uv[1])), c, 1)
                    img_strokes[:,:] = cv.subtract(img_strokes[:,:], tmp[:,:])
                else:
                    tmp = np.zeros((img_strokes.shape[0],img_strokes.shape[1],3), np.uint8)
                    b = int((255-src[x,y][0])/coeff)
                    g = int((255-src[x,y][1])/coeff)
                    r = int((255-src[x,y][2])/coeff)
                    cv.line(tmp, (y - int(uv[0]), x - int(uv[1])), (y + int(uv[0]), x + int(uv[1])), (b, g, r) , 1)
                    img_strokes[:,:,:] = cv.subtract(img_strokes[:,:,:], tmp[:,:,:])
        else :
            counter=counter-1
        
        bar.next()
    if not os.path.exists('./output/tensors/'+os.path.dirname(output)):
        os.makedirs('./output/tensors/'+os.path.dirname(output))
    cv.imwrite('./output/tensors/'+str(output)+'img_results.jpg', img_strokes)   


