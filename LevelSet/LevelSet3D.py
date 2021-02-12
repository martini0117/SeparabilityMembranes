import numpy as np
from enum import Enum
import cv2
import time
import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import Toolbox as tb

gridsize = 200
wband = 10
wreset = 3
initialoffset = 1
maxnarrowband = 100 * gridsize * gridsize * wband

maxfront = 100 * gridsize * gridsize

FARAWAY = 1
BAND = 2
RESETBAND = 3
FRONT = 4

dt = 1
gain = 0.1

phi = np.empty((gridsize,gridsize,gridsize))
dphi = np.empty((gridsize,gridsize,gridsize))
F = np.empty((gridsize,gridsize,gridsize))
gray = np.empty((gridsize,gridsize,gridsize),dtype=np.uint8)
status = np.empty((gridsize,gridsize,gridsize),dtype=np.uint8)
Front = np.empty((maxfront,3))
NFront = None
NarrowBand = np.empty((maxnarrowband,2),dtype=np.int32)
NNarrowBand = None
CircleMap = np.empty((wband+1,4*wband**3,3),dtype=np.int32)
NCircleMap = np.empty((wband+1,),dtype=np.int32)

def InitializeCircleMap():
    NCircleMap.fill(0)
    
    for x in range(-wband, wband+1):
        for y in range(-wband, wband+1):
            for z in range(-wband, wband+1):
                d = int(np.linalg.norm([x,y,z]))
                if(d <= wband):
                    CircleMap[d,NCircleMap[d],0] = x
                    CircleMap[d,NCircleMap[d],1] = y
                    CircleMap[d,NCircleMap[d],2] = z
                    NCircleMap[d] += 1
                
def InitializeFrontPosition():
    n = 0
    status.fill(FARAWAY)

    for x in range(initialoffset, gridsize-initialoffset):
        for y in range(initialoffset, gridsize-initialoffset):
            status[x,y,initialoffset] = FRONT
            Front[n,0] = x
            Front[n,1] = y
            Front[n,2] = initialoffset
            phi[x,y,initialoffset] = 0.0
            n += 1

    for x in range(initialoffset, gridsize-initialoffset):
        for y in range(initialoffset, gridsize-initialoffset):
            status[x,y,gridsize-1-initialoffset] = FRONT
            Front[n,0] = x
            Front[n,1] = y
            Front[n,2] = gridsize-1-initialoffset
            phi[x,y,gridsize-1-initialoffset] = 0.0
            n += 1

    for x in range(initialoffset, gridsize-initialoffset):
        for z in range(initialoffset, gridsize-initialoffset):
            status[x,initialoffset,z] = FRONT
            Front[n,0] = x
            Front[n,1] = initialoffset
            Front[n,2] = z
            phi[x,initialoffset,y] = 0.0
            n += 1

    for x in range(initialoffset, gridsize-initialoffset):
        for z in range(initialoffset, gridsize-initialoffset):
            status[x,gridsize-1-initialoffset,z] = FRONT
            Front[n,0] = x
            Front[n,1] = gridsize-1-initialoffset
            Front[n,2] = z
            phi[x,gridsize-1-initialoffset,z] = 0.0
            n += 1
    
    for y in range(initialoffset, gridsize-initialoffset):
        for z in range(initialoffset, gridsize-initialoffset):
            status[initialoffset,y,z] = FRONT
            Front[n,0] = initialoffset
            Front[n,1] = y
            Front[n,2] = z
            phi[initialoffset,y,z] = 0.0
            n += 1
    
    for y in range(initialoffset, gridsize-initialoffset):
        for z in range(initialoffset, gridsize-initialoffset):
            status[gridsize-1-initialoffset,y,z] = FRONT
            Front[n,0] = gridsize-1-initialoffset
            Front[n,1] = y
            Front[n,2] = z
            phi[gridsize-1-initialoffset,y,z] = 0.0
            n += 1

    global NFront
    NFront = n

    for x in range(gridsize):
        for y in range(gridsize):
            for z in range(gridsize):
                if(status[x,y,z] != FRONT):
                    if(x > initialoffset and x < gridsize-initialoffset-1
                    and y > initialoffset and y < gridsize-initialoffset-1
                    and z > initialoffset and z < gridsize-initialoffset-1):
                        phi[x,y,z] = -wband
                else:
                    phi[x,y,z] = wband

    global NNarrowBand
    NNarrowBand = 0
    SetSpeedFunction(1)

def SetSpeedFunction(reset):
    Fs = 0
    global NNarrowBand

    xyz = NarrowBand[:NNarrowBand]
    F[xyz] = 0.0
    dphi[xyz] = 0.0

    if reset:
        status[xyz] = np.where(status[xyz] != FRONT, FARAWAY, FRONT)
    
    # print(NFront)
    for f in Front[:NFront]:
        x = int(f[0])
        y = int(f[1])
        z = int(f[2])

        if x < 1 or x > gridsize - 2 or y < 1 or y > gridsize - 2 or z < 1 or z > gridsize - 2:
            continue

        dx = float(gray[x+1, y, z]) - gray[x,y,z]
        dy = float(gray[x,y+1, z]) - gray[x,y,z]
        dz = float(gray[x, y, z+1]) - gray[x,y,z]

        F[x,y,z] = 1.0 /(1.0 + np.sqrt(dx*dx+dy*dy+dz*dz))

        dx  = (phi[x-1,y,z]-phi[x+1,y,z])/2 # (l-r)/2
        dxx = phi[x-1,y,z]-2*phi[x,y,z]+phi[x+1,y,z] # l-2c+r
        dx2 = dx*dx

        dy  = (phi[x,y-1,z]-phi[x,y+1,z])/2 # (u-d)/2
        dyy = phi[x,y-1,z]-2*phi[x,y,z]+phi[x,y+1,z] # u-2c+d
        dy2 = dy*dy

        dz  = (phi[x,y,z-1]-phi[x,y,z+1])/2 # (b-f)/2
        dzz = phi[x,y,z-1]-2*phi[x,y,z]+phi[x,y,z+1] # b-2c+f
        dz2 = dz*dz

        # (ul+dr-ur-dl)/4
        dxy = (phi[x-1,y-1,z]+phi[x+1,y+1,z]-phi[x+1,y-1,z]-phi[x-1,y+1,z])/4

        # (lf+rb-rf-lb)/4
        dxz = (phi[x-1,y,z+1]+phi[x+1,y,z-1]-phi[x+1,y,z+1]-phi[x-1,y,z-1])/4

        # (uf+db-df-ub)/4
        dyz = (phi[x,y-1,z+1]+phi[x,y+1,z-1]-phi[x,y+1,z+1]-phi[x,y-1,z-1])/4

        #-- compute curvature (Kappa)

        df = dx2 + dy2 + dz2
        if df != 0.0:
            kappa = ( (dxx*(dy2+dz2)+dyy*(dx2+dz2)+dzz*(dx2+dy2)-
                    2*dx*dy*dxy-2*dx*dz*dxz-2*dy*dz*dyz)/
                    (dx2+dy2+dz2) )
        else:
            kappa = 0.0

        F[x,y,z] = F[x,y,z] * (-1.0 - gain * kappa)

        Fs += F[x,y,z] 

    for d in range(wband, 0, -1):
        for f in Front[:NFront]:
            xf = int(f[0])
            yf = int(f[1])
            zf = int(f[2])

            if reset: 
                phi[xf,yf,zf] = 0.0

            update_x = CircleMap[d,:NCircleMap[d],0] + xf
            update_y = CircleMap[d,:NCircleMap[d],1] + yf
            update_z = CircleMap[d,:NCircleMap[d],2] + zf

            
            skip = (update_x < 0) | (update_x > gridsize - 1) \
                 | (update_y < 0) | (update_y > gridsize - 1) \
                 | (update_z < 0) | (update_z > gridsize - 1)
            skip[~skip] = (status[update_x[~skip], update_y[~skip], update_z[~skip]] == FRONT)

            F[update_x[~skip], update_y[~skip], update_z[~skip]] = F[xf,yf,zf]

            if reset:
                if d > wband - wreset:
                    status[update_x[~skip], update_y[~skip], update_z[~skip]] = RESETBAND
                else:
                    status[update_x[~skip], update_y[~skip], update_z[~skip]] = BAND
                phi[update_x[~skip], update_y[~skip], update_z[~skip]] = \
                    np.where(phi[update_x[~skip], update_y[~skip], update_z[~skip]] < 0, -d, d)

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
                if status[x,y,z] != FARAWAY:
                    NarrowBand[n,0] = x
                    NarrowBand[n,1] = y
                    NarrowBand[n,2] = z
                    n += 1
                    if(n >= maxnarrowband):
                        print('maxnarrowband')
                        return 0

        NNarrowBand = n

    return Fs          

def FrontPropagation():
    for x, y, z in NarrowBand[:NNarrowBand]:

        if x < 1 or x > gridsize - 2 or y < 1 or y > gridsize - 2 or z < 1 or z > gridsize - 2:
            continue

        if F[x,y,z] > 0.0 :
            fdxm = np.max(phi[x,y,z] - phi[x-1,y,z],0)
            fdxp = np.min(phi[x+1,y,z] - phi[x,y,z],0)
            fdym = np.max(phi[x,y,z] - phi[x,y-1,z],0)
            fdyp = np.min(phi[x,y+1,z] - phi[x,y,z],0)
            fdzm = np.max(phi[x,y,z] - phi[x,y,z-1],0) 
            fdzp = np.min(phi[x,y,z+1] - phi[x,y,z],0)

            dphi[x,y,z] = F[x,y,z] * np.sqrt(fdxm**2 + fdxp**2 + fdym**2 + fdyp**2 + fdzm**2 + fdzp**2) * dt
        else:
            fdxp = np.max(phi[x+1,y,z] - phi[x,y,z],0)
            fdxm = np.min(phi[x,y,z] - phi[x-1,y,z],0)
            fdyp = np.max(phi[x,y+1,z] - phi[x,y,z],0)
            fdym = np.min(phi[x,y,z] - phi[x,y-1,z],0)
            fdzp = np.max(phi[x,y,z+1] - phi[x,y,z],0)
            fdzm = np.min(phi[x,y,z] - phi[x,y,z-1],0) 

            dphi[x,y,z] = F[x,y,z] * np.sqrt(fdxm**2 + fdxp**2 + fdym**2 + fdyp**2 + fdzm**2 + fdzp**2) * dt


    for i in range(NNarrowBand):
        x = NarrowBand[i,0]
        y = NarrowBand[i,1]
        z = NarrowBand[i,2]
        phi[x,y,z] = phi[x,y,z] - dphi[x,y,z]


def ReLabeling():
    n = 0
    flg = 0

    for i in range(NNarrowBand):
        x = NarrowBand[i,0]
        y = NarrowBand[i,1]
        z = NarrowBand[i,2]

        if x < 1 or x > gridsize - 2 or y < 1 or y > gridsize - 2 or z < 1 or z > gridsize - 2:
            continue

        if ((phi[x,y,z] >= 0.0 and
            ((phi[x+1,y,z] + phi[x,y,z] <= 0.0) or (phi[x,y+1,z] + phi[x,y,z] <= 0.0) or (phi[x,y,z+1] + phi[x,y,z] <= 0.0) or
             (phi[x-1,y,z] + phi[x,y,z] <= 0.0) or (phi[x,y-1,z] + phi[x,y,z] <= 0.0) or (phi[x,y,z-1] + phi[x,y,z] <= 0.0))) or
            (phi[x,y] <= 0.0 and
            ((phi[x+1,y,z] + phi[x,y,z] >= 0.0) or (phi[x,y+1,z] + phi[x,y,z] >= 0.0) or (phi[x,y,z+1] + phi[x,y,z] >= 0.0) or
             (phi[x-1,y,z]+ phi[x,y,z] >= 0.0) or (phi[x,y-1,z] + phi[x,y,z] >= 0.0) or (phi[x,y,z-1] + phi[x,y,z] >= 0.0)))):

            if status[x,y,z] == RESETBAND:
                flg = 1
            
            status[x,y,z] = FRONT
            Front[n,0] = x
            Front[n,1] = y
            Front[n,2] = z

            n += 1
            if n >= maxfront:
                print('maxfront')
                return -1
        else:
            if status[x,y,z] == FRONT:
                status[x,y,z] = BAND

    global NFront
    NFront = n
    return flg

def DrawContour():
    global dst
    dst = np.copy(img)
    for i in range(NFront-1):
        dst = cv2.rectangle(dst, (int(Front[i,1]), int(Front[i,0])),
                            (int(Front[i,1])+1, int(Front[i,0])+1), (100,100,100))

# img = cv2.imread('LevelSet/sample.bmp')
# dst = np.copy(img)
gray = tb.make_sphere_voxel(gridsize,gridsize/2,gridsize/4).astype(np.uint8) * 255
print(gray.dtype)

# cv2.namedWindow('Image', 1)
# cv2.resizeWindow('Image',gridsize, gridsize)

InitializeCircleMap()
print(NCircleMap)

InitializeFrontPosition()
# print(status)
# cv2.imshow('Image', 50 * status)
# cv2.waitKey(0)

reset = 1
Fs = 0

while True:
    # cv2.imshow('Image', dst)

    Fs = SetSpeedFunction(reset)
    print(Fs)
    print(NFront)
    if Fs == 0.0:
        break

    FrontPropagation()

    reset = ReLabeling()
    if reset < 0:
        break

    # DrawContour()

    # key = cv2.waitKey(10)
    # if(key == 'q' or key == 27):
    #     break


