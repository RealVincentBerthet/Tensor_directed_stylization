import cv2 as cv
import time
import tensors
import tensorsTools
import os

def main():
    time = tensorsTools.Timer()
    images=['./sources/ratatouille.jpg']
    number=[50000]
    epsilon=[4]
    length=[100,150]
    coefficient=[3]
    sigma1=[0.5]
    sigma2=[1.2]
    power1=[1.2]
    power2=[0.5]

    for img in images :
        for c in coefficient :
            source = cv.imread(str(img),cv.IMREAD_COLOR) #bgr
            size = (int(source.shape[1]/c), int(source.shape[0]/c))
            source = cv.resize(source, size)
            for s1 in sigma1 :
                for s2 in sigma2:
                    img_sobel_x_lab,img_sobel_y_lab=tensors.initialization(source,s1)
                    A,B,C=tensors.computeEigen(img_sobel_x_lab,img_sobel_y_lab,s2)
                    for p1 in power1 :
                        for p2 in power2 :
                            if p1>=p2 and p1>=0 :
                                T=tensors.computeTensors(A,B,C,p1,p2)
                                o=os.path.splitext(os.path.basename(img))[0]+"/"
                                w=tensors.computeVectorField(T)
                                for n in number :
                                    for e in epsilon :
                                        for l in length :
                                            path=str(o)+'n_'+str(n)+'_e_'+str(e)+'_l_'+str(l)+'_p1_'+str(p1)+'_p2_'+str(p2)
                                            print('\nDraw : ./output/tensors/'+path+'.jpg')
                                            tensorsTools.draw_strokes(source, w,T,n,e,l,output=path)
                                            cv.imwrite('./output/tensors/'+str(o)+'img.jpg', source)

                            
   
    print('Time : '+str(time))
    
if __name__ == '__main__':
    main()

