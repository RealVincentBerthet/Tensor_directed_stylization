import cv2 as cv
import numpy as np
import time

def strokes(x,y,cpt,result) :
    tmp = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    
    # if np.sqrt(lambda_plus[x,y]+lambda_moins[x,y])>epsilon :
    #     px1=x-L*wx
    #     py1=y-L*wy
    #     px2=
    #     py2=
    #     cv.line(tmp, (y - int(uv[0]), x - int(uv[1])), (y + int(uv[0]), x + int(uv[1])), 10, 1)
    #     result[:,:] = cv.subtract(result[:,:], tmp[:,:])
    #     cpt+=1
    
    # return cpt,result

## Parameters
start_time = time.time()
sigma = 1    
nb_strokes=10000
epsilon = 0             # ~0
L=4
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

#Convert A,B,C CV_64FC1
A=np.float64(A)
B=np.float64(B)
C=np.float64(C)

#1. Compute Tensors from eigenValues, eigenVectors
p1=2 #p1>=p2>=0
p2=1
T=np.zeros(A.shape,np.float64)
lambda_plus=np.zeros(A.shape,np.float64)
lambda_moins=np.zeros(A.shape,np.float64)
theta_plus_u=np.zeros(A.shape,np.float64)
theta_plus_v=np.zeros(A.shape,np.float64)
theta_moins_u=np.zeros(A.shape,np.float64)
theta_moins_v=np.zeros(A.shape,np.float64)

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
        lambda_plus[i,j]=eigen[1][0]
        lambda_moins[i,j]=eigen[1][1]

        theta_plus_u=eigen[2][0][0]
        theta_plus_v=eigen[2][0][1]
        theta_moins_u=eigen[2][1][0]
        theta_moins_v=eigen[2][1][0]

        theta_plus=eigen[2][0]
        theta_moins=eigen[2][1]

        #compute Tensor
        c_plus=1/np.power(1+lambda_plus[i,j]+lambda_moins[i,j],p1) 
        c_moins=1/np.power(1+lambda_plus[i,j]+lambda_moins[i,j],p2)
        res=c_plus*theta_plus*theta_plus.transpose()+c_moins*theta_moins*theta_moins.transpose()
        T[i,j]=res[0]

#TEST : Draw ellipse @TODO normaliser tous depuis le debut entre 0 et 1 ????
#Discretiser l'image et tracer des ellipse
step=20
img_ellipse=img

lambda_plus_normalized=cv.normalize(lambda_plus, None, 0,10, norm_type=cv.NORM_MINMAX)
lambda_moins_normalized=cv.normalize(lambda_moins, None, 0,10, norm_type=cv.NORM_MINMAX)
for i in range(0,img_ellipse.shape[0],step) : 
    for j in range(0,img_ellipse.shape[1],step) : 
        axeLength=(lambda_moins_normalized[i,j].astype(int),lambda_plus_normalized[i,j].astype(int))
        print(axeLength)
        center=(i,j)
        cv.ellipse(img_ellipse,center,axeLength,0,0,360,(255,0,0),-1)


#2. Decomposition champs de vecteurs pour differents angles @TODO plus d'angle
phi=0
angle=np.array([np.cos(phi),np.sin(phi)],np.float64)
Tsqrt=np.sqrt(T)
wx=np.zeros(T.shape,np.float64)
wy=np.zeros(T.shape,np.float64)

for i in range(Tsqrt.shape[0]) :
    for j in range(Tsqrt.shape[1]) :
        tmp=Tsqrt[i,j]*angle
        wx[i,j]=tmp[0]
        wy[i,j]=tmp[1]

#3. Initialize a white background image
img_grayscale = np.zeros((img.shape[0],img.shape[1]), np.uint8)
img_grayscale[:,:] = 255

#4. Pick a random location (x,y) in  img_grayscale
for cpt in range(0,nb_strokes+1) :
        x=np.random.randint(img_grayscale.shape[0])
        y=np.random.randint(img_grayscale.shape[1])
        #cpt,img_grayscale=strokes(x,y,cpt,img_grayscale)
        #if round(((cpt-1)/nb_strokes)*100) != round((cpt/nb_strokes)*100) :
            #print('strokes : '+str(nb_strokes)+' - '+str(round((cpt/nb_strokes)*100))+' %')

    


#  normalizedImg=cv.normalize(wu, None, 0,255, norm_type=cv.NORM_MINMAX)
#  cv.imshow('dst_rt', normalizedImg)
#  cv.waitKey(0)

print('time : '+str(round(time.time() - start_time))+' seconds')