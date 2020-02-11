import os
import time

def main():
    start_time = time.time()
    images=['./sources/olive.png']
    number=[50000,100000,150000,200000]
    epsilon=[1]
    length=[20,40,50,80]
    coefficient=[1]
    power1=[1.2]
    power2=[0.5]
    output=["lemeur/"]
    for img in images :
        for n in number :
            for e in epsilon :
                for l in length :
                    for c in coefficient :
                        for p1 in power1 :
                            for p2 in power2 :
                                for o in output :
                                    detail='n_'+str(n)+'_e_'+str(e)+'_l_'+str(l)+'_p1_'+str(p1)+'_p2_'+str(p2)
                                    cmd = 'python ./Scripts/tensors.py -i '+str(img)+' -n '+str(n)+' -e '+str(e)+' -l '+str(l)+' -p1 '+str(p1)+' -p2 '+str(p2)+' -c '+str(c)+' -o '+str(o)+str(detail)
                                    print("Execute : "+str(cmd))
                                    os.system(cmd)

    print('time : '+str(round(time.time() - start_time))+' seconds')
    
if __name__ == '__main__':
    main()

