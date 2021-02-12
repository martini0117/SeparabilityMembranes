import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/ellipsoid_fit')
import math

import matplotlib

import geomdl.visualization.VisMPL as VisMPL
import matplotlib.pyplot as plt
import numpy as np
from geomdl import BSpline
from geomdl.fitting import approximate_surface
from geomdl.knotvector import generate
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from skimage.segmentation import flood_fill
import nibabel as nib
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import Toolbox as tb
import ellipsoid_fit as ef

# cases = [5,30,31,55,62,73,75,81,151]
# cases = range(210,299)
# data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/TestData/'

removed = True

# cases = [1,18,22,23,29,50,52,59,71,73,75,76,78,80,82,97,98,106,109,114,127,128,135,144,166,167,170,171,173,175,179,185,196,203,209]
cases = [52]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/ValidationData/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/RemovedW7DataLong/'

for case in cases:
    
    res = 50
    wdh = np.array([10,15,20])
    pos = np.array([20,20,20])
    img = tb.make_rectangular_voxel(res,wdh,pos)
    # tb.show_ct_image(img)

    # # 球のパラメータ
    # res = 300
    # center = [res/2, res/2 + 20, res/2 + 40]
    # radius_true = 40

    # # ノイズを乗せる
    # img = tb.make_sphere_voxel_v(res,center,radius_true)

    # tb.show_image_collection(img)

    rect_size = np.array([10,3,3])

    center = np.array([20.5,25.5,25.5])
    normal = np.array([0,0,1])
    normal = normal / np.linalg.norm(normal)
    clipped_image = tb.clip_rect(img,center,rect_size,normal)
    separability, boundary = tb.cul_separability(clipped_image)

    updated = center + (boundary + 1 - rect_size[0]/2) * normal
    print(center)
    print(updated)

    print(boundary)
    print(separability)
    print(clipped_image.transpose(2,0,1))
    tb.show_image_collection(clipped_image.transpose(2,0,1))

    mask = np.zeros(img.shape)
    mask[int(center[0]),int(center[1]),int(center[2])] = 1
    mask[int(updated[0]),int(updated[1]),int(updated[2])] = 1              
    
    color_img = tb.draw_segmentation(img, mask,mark_val=1)

    tb.show_ct_image(color_img)



