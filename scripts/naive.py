import cv2 as cv
import numpy as np

def strokes(x,y,cpt,result) :
    tmp = np.zeros((sobel.shape[0],sobel.shape[1]), np.uint8)
    if sobel[x,y]>epsilon:
        uv = [L/2 *sobely[x,y]/sobel[x,y], L/2 *-sobelx[x,y]/sobel[x,y]]
        cv.line(tmp, (y - int(uv[0]), x - int(uv[1])), (y + int(uv[0]), x + int(uv[1])), 10, 1)
        result-=tmp
        cpt+=1
    return cpt,result

# Parameters
L = 4                     # length of a strokes
sigma_gaussian = 10       # standard deviation >=0
epsilon = 2               # level >=0
random=100000              # set random to -1 to cross all the image

# Read image
img = cv.imread('./../sources/lena.png',cv.IMREAD_COLOR)
img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
img_y = img_ycrcb[:,:,0]

# Gaussian filter
gaussian = cv.GaussianBlur(img_y,(5,5),sigma_gaussian)
cv.imwrite("../output/gaussian_s_"+str(sigma_gaussian)+".jpg", gaussian)

# Gradient
sobel = cv.Sobel(gaussian, cv.CV_64F, 1, 1, ksize=1)
sobelx = cv.Sobel(gaussian, cv.CV_64F, 1, 0, ksize=1)
sobely = cv.Sobel(gaussian, cv.CV_64F, 0, 1, ksize=1)
cv.imwrite("../output/sobel_s_"+str(sigma_gaussian)+".jpg", sobel)

# Initialize result image
height, width = sobel.shape
result = np.zeros((height,width), np.uint8)
result[:,:] = 255

# Random position
cpt = 0

if random>0 :
    # pick random location
    for cpt in range(0,random+1) :
        x=np.random.randint(0,sobel.shape[0])
        y=np.random.randint(0,sobel.shape[1])
        cpt,result=strokes(x,y,cpt,result)
else :
    # cross all the image
    for x in range(sobel.shape[0]):
        for y in range(sobel.shape[1]):
            cpt,result=strokes(x,y,cpt,result)

print("number of strokes : "+ str(cpt))
cv.imwrite("../output/res_greyscale_s_"+str(sigma_gaussian)+"_l_"+str(L)+"_e_"+str(epsilon)+".jpg", result)


# Colorisation
gaussiancolor = cv.GaussianBlur(img,(35,35),0)
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        if result[i,j]<255:
            gaussiancolor[i,j,0] =0
            gaussiancolor[i, j, 1] =0
            gaussiancolor[i, j, 2] =0
cv.imwrite("../output/res_color_s_"+str(sigma_gaussian)+"_l_"+str(L)+"_e_"+str(epsilon)+".jpg", gaussiancolor)

