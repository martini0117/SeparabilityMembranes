import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')

import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from skimage import morphology
from scipy import ndimage as ndi
from skimage.viewer import ImageViewer
from skimage.viewer import CollectionViewer
from geomdl import BSpline
from geomdl.fitting import approximate_surface
from geomdl.knotvector import generate
import geomdl.visualization.VisMPL as VisMPL

import Toolbox as tb

def cut_out(img, center, filter_size):
    return img[int(center[0]-filter_size):int(center[0]+filter_size+1),int(center[1]-filter_size):int(center[1]+filter_size+1),int(center[2]-filter_size):int(center[2]+filter_size+1)]

# 球の輪郭を抽出するデモ

# 球のパラメータ
# res = 100
# center = res/2
# radius_true = 25

# # ノイズを乗せる
# img = 0.2*tb.make_sphere_voxel(res,center,radius_true)
# img += 0.8*np.random.rand(res,res,res)

# tb.show_image_collection(img)

img = tb.load_test_medical_image()
tb.show_image_collection(img)

filter_size = 11

img = tb.add_margin(img, filter_size)
filtered_img = np.zeros(img.shape)

center = int(filter_size/2)
radius = 5

sfilter = tb.make_sphere_voxel(filter_size,center,radius)

tb.show_image_collection(sfilter)

w,h,d = sfilter.shape
N = w*h*d

n_2 = np.sum(sfilter)
n_1 = N - n_2 

for x in range(filter_size,img.shape[0]-filter_size):
    for y in range(filter_size,img.shape[1]-filter_size):
        for z in range(filter_size,img.shape[2]-filter_size):
            center = np.array([x,y,z])
            
            cut_img = cut_out(img,center,int(filter_size/2))

            
            variance_all = N * np.var(cut_img)
            average_all = np.mean(cut_img)

            P_1 = np.mean(np.where(sfilter == 0,cut_img,0))
            P_2 = np.mean(np.where(sfilter == 1,cut_img,0))

            variance_boundary = n_1 * (P_1 - average_all) ** 2 + n_2 * (P_2 - average_all) ** 2
    
            separability = variance_boundary/variance_all if variance_all != 0 else 0
            filtered_img[x,y,z] = separability

            print('(' + str(x) + ', ' + str(y) + ', ' + str(z) + ')')


tb.show_image_collection(filtered_img)





