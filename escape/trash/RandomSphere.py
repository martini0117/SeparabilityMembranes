import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
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
import random

import Toolbox as tb

# 球のパラメータ
res = 100
center = res/2
radius_true = 25

# 初期輪郭
radius_init = 30

psize = 20
qsize = 8

points = np.array([[radius_init*math.cos(u)*math.cos(v), radius_init*math.cos(v)*math.sin(u),radius_init*math.sin(v)] 
                    for u in np.linspace(0,2*math.pi,num=20) 
                    for v in np.linspace(-math.pi/2+0.01,math.pi/2-0.01,num=psize)])

points += center

random.shuffle(points)

surf = approximate_surface(points.tolist(),psize,psize,3,3,ctrlpts_size_u=qsize,ctrlpts_size_v=qsize)

surf.delta = 0.05
surf.vis = VisMPL.VisSurfWireframe()
surf.render()