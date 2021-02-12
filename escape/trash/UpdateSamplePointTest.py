import sys; sys.path.append('/home/kitamiya/Documents/python/snake3D/src/Toolbox')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import Toolbox as tb

w,h,d = (20,20,20)
rect_size = (8,2,2)

image = np.zeros((w,h,d))
image[int(w/2):,:,:] = 1

center = np.array([int(w/2)+2,int(h/2),int(d/2)],dtype='float32')
neighborhood = np.array([center, center,center,center])
neighborhood[0] += np.array([0.5,-1,0])
neighborhood[1] += np.array([0, 0,-1])
neighborhood[2] += np.array([-0.5, 1,0])
neighborhood[3] += np.array([0, 0,1])

print(center)
print(tb.update_sample_point(center,neighborhood,image,rect_size))
