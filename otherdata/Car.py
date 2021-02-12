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

from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
from matplotlib import cm

import os

data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/volume/Car1/img/'
files = os.listdir(data_path)

img = []
for i in range(len(files)):
    tmp = cv2.imread(data_path + '{:0=4}.jpg'.format(i+1), cv2.IMREAD_GRAYSCALE)
    img.append(tmp)
img = np.array(img)


print(img.shape)

tb.show_image_collection(img.transpose(0,1,2))

# radius_init = 10
# center = np.array([30,116,90])
rect_size = np.array([40,8,8])

# points = np.array([[radius_init*math.cos(u)*math.cos(v), radius_init*math.cos(v)*math.sin(u),radius_init*math.sin(v)] 
#                     for u in np.linspace(0,2*math.pi,num=20) 
#                     for v in np.linspace(-math.pi/2+0.01,math.pi/2-0.01,num=20)])

# points += center

# # 制御点補完
# surf = approximate_surface(points.tolist(),20,20,3,3,ctrlpts_size_u=8,ctrlpts_size_v=8)

points = [[133,100,251],[210,102,203],[337,100,200],[460,100,200],[660,100,190],[830,100,250]]
radius = [40,35,30,25,20,20]

surf = tb.manual_init(points, radius,pn=30,qn=8)
print('pass')
# tb.show_image_collection(tb.draw_contour_u(img, surf,color=[255,255,0]))
# surf.vis = VisMPL.VisSurface()
# surf.render(colormap=cm.cool)

# final = tb.draw_contour_u(img, surf,color=[255,255,0],delta=0.001)
# tb.show_image_collection(final)


surf = tb.cul_contour(img,surf,rect_size,div=40,N_limit=10,c=0.95,ctrlpts_size=8,w=1,dif_limit=-0.005,dif_abs=False,ctrlpts_increasing=True)

final = tb.draw_contour_u(img, surf,color=[255,255,0],delta=0.001)
tb.show_image_collection(final)