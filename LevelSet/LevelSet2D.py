import numpy as np
from enum import Enum
import cv2
import time

gridsize = 500
wband = 10
wreset = 3
initialoffset = 1
maxnarrowband = 100 * gridsize * wband

maxfront = 100 * gridsize

FARAWAY = 1
BAND = 2
RESETBAND = 3
FRONT = 4

dt = 1
gain = 0.1

phi = np.empty((gridsize,gridsize))
dphi = np.empty((gridsize,gridsize))
F = np.empty((gridsize,gridsize))
gray = np.empty((gridsize,gridsize),dtype=np.uint8)
status = np.empty((gridsize,gridsize),dtype=np.uint8)
Front = np.empty((maxfront,2))
NFront = None
NarrowBand = np.empty((maxnarrowband,2),dtype=np.int32)
NNarrowBand = None
CircleMap = np.empty((wband+1,7*(wband + 1),2),dtype=np.int32)
NCircleMap = np.empty((wband+1,),dtype=np.int32)

def InitializeCircleMap():
    NCircleMap.fill(0)
    
    for x in range(-wband, wband+1):
        for y in range(-wband, wband+1):
            d = int(np.linalg.norm([x,y]))
            if(d <= wband):
                CircleMap[d,NCircleMap[d],0] = x
                CircleMap[d,NCircleMap[d],1] = y
                NCircleMap[d] += 1
              
def InitializeFrontPosition():
    n = 0
    status.fill(FARAWAY)

    for x in range(initialoffset, gridsize-initialoffset):
        status[x,initialoffset] = FRONT
        Front[n,0] = x
        Front[n,1] = initialoffset
        phi[x,initialoffset] = 0.0
        n += 1
    
    for y in range(initialoffset, gridsize-initialoffset):
        status[gridsize-1-initialoffset,y] = FRONT
        Front[n,0] = gridsize-1-initialoffset
        Front[n,1] = y
        phi[gridsize-1-initialoffset,y] = 0.0
        n += 1
    
    for x in range(gridsize-1-initialoffset, initialoffset-1, -1):
        status[x,gridsize-1-initialoffset] = FRONT
        Front[n,0] = x
        Front[n,1] = gridsize-1-initialoffset
        phi[x,gridsize-1-initialoffset] = 0.0
        n += 1 

    for y in range(gridsize-1-initialoffset, initialoffset-1, -1):
        status[initialoffset,y] = FRONT
        Front[n,0] = initialoffset
        Front[n,1] = y
        phi[initialoffset,y] = 0.0
        n += 1

    global NFront
    NFront = n

    for x in range(gridsize):
        for y in range(gridsize):
            if(status[x,y] != FRONT):
                if(x > initialoffset and x < gridsize-initialoffset-1
                and y > initialoffset and y < gridsize-initialoffset-1):
                    phi[x,y] = -wband
            else:
                phi[x,y] = wband

    global NNarrowBand
    NNarrowBand = 0
    SetSpeedFunction(1)

def SetSpeedFunction(reset):
    Fs = 0
    global NNarrowBand

    xy = NarrowBand[:NNarrowBand]
    F[xy] = 0.0
    dphi[xy] = 0.0

    if reset:
        status[xy] = np.where(status[xy] != FRONT, FARAWAY, FRONT)
    
    # print(NFront)
    for f in Front[:NFront]:
        x = int(f[0])
        y = int(f[1])

        if x < 1 or x > gridsize - 2 or y < 1 or y > gridsize - 2:
            continue

        dx = float(gray[x+1, y]) - gray[x,y]
        dy = float(gray[x,y+1]) - gray[x,y]

        F[x,y] = 1.0 /(1.0 + np.sqrt(dx*dx+dy*dy))

        dfx = ((phi[x+1,y+1] - phi[x-1,y+1]) 
                + 2.0*(phi[x+1,y] - phi[x-1,y])
                + (phi[x+1,y-1] - phi[x-1,y-1]))/4.0/2.0
        dfy = ((phi[x+1,y+1] - phi[x+1,y-1]) 
                + 2.0*(phi[x,y+1] - phi[x,y-1])
                + (phi[x-1,y+1] - phi[x-1,y-1]))/4.0/2.0
        dfxy = ((phi[x+1,y+1] - phi[x-1,y+1]) 
                - (phi[x+1,y-1] - phi[x-1,y-1]))/2.0/2.0
        dfx2 = ((phi[x+1,y+1] + phi[x-1,y+1] - 2.0 * phi[x,y+1]) 
                + 2.0*(phi[x+1,y] + phi[x-1,y] - 2.0 * phi[x,y])
                + (phi[x+1,y-1] + phi[x-1,y-1]) - 2.0 * phi[x,y-1])/4.0       
        dfy2 = ((phi[x+1,y+1] + phi[x+1,y-1] - 2.0 * phi[x+1,y]) 
                + 2.0*(phi[x,y+1] + phi[x,y-1] - 2.0 * phi[x,y])
                + (phi[x-1,y+1] + phi[x-1,y-1]) - 2.0 * phi[x-1,y])/4.0

        df = np.sqrt(dfx*dfx+dfy*dfy)
        if df != 0.0:
            kappa = (dfx2 * dfy * dfy - 2.0 * dfx * dfy * dfxy
                    + dfy2 * dfx * dfx) / (df * df * df)
        else:
            kappa = 0.0

        F[x,y] = F[x,y] * (-1.0 - gain * kappa)

        Fs += F[x,y] 

    for d in range(wband, 0, -1):
        for f in Front[:NFront]:
            xf = int(f[0])
            yf = int(f[1])

            if reset: 
                phi[xf,yf] = 0.0

            update_x = CircleMap[d,:NCircleMap[d],0] + xf
            update_y = CircleMap[d,:NCircleMap[d],1] + yf

            
            skip = (update_x < 0) | (update_x > gridsize - 1) | (update_y < 0) | (update_y > gridsize - 1)
            skip[~skip] = (status[update_x[~skip], update_y[~skip]] == FRONT)
            # print(skip)

            F[update_x[~skip], update_y[~skip]] = F[xf,yf]

            if reset:
                if d > wband - wreset:
                    status[update_x[~skip], update_y[~skip]] = RESETBAND
                else:
                    status[update_x[~skip], update_y[~skip]] = BAND
                phi[update_x[~skip], update_y[~skip]] = np.where(phi[update_x[~skip], update_y[~skip]] < 0, -d, d)

            # for cm in CircleMap[d,:NCircleMap[d]]:
            #     x = xf + cm[0]
            #     y = yf + cm[1]

            #     if x < 0 or x > gridsize - 1 or y < 0 or y > gridsize - 1:
            #         continue

            #     if status[x,y] == FRONT:
            #         continue

            #     if reset:
            #         if d > wband - wreset:
            #             status[x,y] = RESETBAND
            #         else:
            #             status[x,y] = BAND
            #         phi[x,y] = -d if phi[x,y] < 0 else d
            
                # F[x,y] = F[xf,yf]

    if reset:
        n = 0
        for x in range(gridsize):
            for y in range(gridsize):
                if status[x,y] != FARAWAY:
                    NarrowBand[n,0] = x
                    NarrowBand[n,1] = y
                    n += 1
                    if(n >= maxnarrowband):
                        print('maxnarrowband')
                        return 0

        NNarrowBand = n

    return Fs          

def FrontPropagation():
    for x, y in NarrowBand[:NNarrowBand]:

        if x < 1 or x > gridsize - 2 or y < 1 or y > gridsize - 2:
            continue

        if F[x,y] > 0.0 :
            fdxm = np.max(phi[x,y] - phi[x-1,y],0)
            fdxp = np.min(phi[x+1,y] - phi[x,y],0)
            fdym = np.max(phi[x,y] - phi[x,y-1],0)
            fdyp = np.min(phi[x,y+1] - phi[x,y],0)

            dphi[x,y] = F[x,y] * np.sqrt(fdxm * fdxm + fdxp * fdxp + fdym * fdym + fdyp * fdyp) * dt
        else:
            fdxp = np.max(phi[x+1,y] - phi[x,y],0)
            fdxm = np.min(phi[x,y] - phi[x-1,y],0)
            fdyp = np.max(phi[x,y+1] - phi[x,y],0)
            fdym = np.min(phi[x,y] - phi[x,y-1],0)

            dphi[x,y] = F[x,y] * np.sqrt(fdxm * fdxm + fdxp * fdxp + fdym * fdym + fdyp * fdyp) * dt


    for i in range(NNarrowBand):
        x = NarrowBand[i,0]
        y = NarrowBand[i,1]
        phi[x,y] = phi[x,y] - dphi[x,y]


def ReLabeling():
    n = 0
    flg = 0

    for i in range(NNarrowBand):
        x = NarrowBand[i,0]
        y = NarrowBand[i,1]

        if x < 1 or x > gridsize - 2 or y < 1 or y > gridsize - 2:
            continue

        if ((phi[x,y] >= 0.0 and
            ((phi[x+1,y] + phi[x,y] <= 0.0) or (phi[x,y+1] + phi[x,y] <= 0.0) or
             (phi[x-1,y] + phi[x,y] <= 0.0) or (phi[x,y-1] + phi[x,y] <= 0.0))) or
            (phi[x,y] <= 0.0 and
            ((phi[x+1,y] + phi[x,y] >= 0.0) or (phi[x,y+1] + phi[x,y] >= 0.0) or
             (phi[x-1,y]+ phi[x,y] >= 0.0) or (phi[x,y-1] + phi[x,y] >= 0.0)))):

            if status[x,y] == RESETBAND:
                flg = 1
            
            status[x,y] = FRONT
            Front[n,0] = x
            Front[n,1] = y

            n += 1
            if n >= maxfront:
                print('maxfront')
                return -1
        else:
            if status[x,y] == FRONT:
                status[x,y] = BAND

    global NFront
    NFront = n
    return flg

def DrawContour():
    global dst
    dst = np.copy(img)
    for i in range(NFront-1):
        dst = cv2.rectangle(dst, (int(Front[i,1]), int(Front[i,0])),
                            (int(Front[i,1])+1, int(Front[i,0])+1), (100,100,100))

img = cv2.imread('LevelSet/sample.bmp')
dst = np.copy(img)
gray = img = cv2.imread('LevelSet/sample.bmp', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('Image', 1)
cv2.resizeWindow('Image',gridsize, gridsize)

InitializeCircleMap()

InitializeFrontPosition()
# print(status)
# cv2.imshow('Image', 50 * status)
# cv2.waitKey(0)

reset = 1
Fs = 0

while True:
    cv2.imshow('Image', dst)

    Fs = SetSpeedFunction(reset)
    # print(Fs)
    print(NFront)
    if Fs == 0.0:
        break

    FrontPropagation()

    reset = ReLabeling()
    if reset < 0:
        break

    DrawContour()

    key = cv2.waitKey(10)
    if(key == 'q' or key == 27):
        break


