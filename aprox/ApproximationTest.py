import sys
sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/ellipsoid_fit')
import math

import matplotlib

import geomdl.visualization.VisMPL as VisMPL
import matplotlib.pyplot as plt
import numpy as np
from geomdl import BSpline
from geomdl.fitting import approximate_surface
from geomdl.knotvector import generate

from geomdl.helpers import basis_function
from geomdl.helpers import basis_function_one
from geomdl.knotvector import generate

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from skimage.segmentation import flood_fill
import nibabel as nib
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import Toolbox as tb
import ellipsoid_fit as ef

n_cp = 10
radius_x = 5
radius_y = 10
n_sp = 100
degree = 3

sp = np.array([[radius_x * math.cos(t), radius_y * math.sin(t)] for t in np.linspace(0, 2*math.pi, n_sp)])
t = np.linspace(0,1,n_sp)

# plt.scatter(sp[:,0], sp[:,1])
# plt.show()

n_knot = n_cp + degree + 1
# knot = np.linspace(0,1, n_knot).tolist()
knot = generate(degree, n_cp)

A = np.array([[basis_function_one(degree, knot,j,t[i]) for j in range(n_cp)]for i in range(n_sp)])
print(A.shape)

invATA = np.linalg.inv(np.dot(A.T, A))
cp = np.dot(np.dot(invATA, A.T), sp).tolist()


crv = BSpline.Curve()
crv.degree = degree
crv.ctrlpts = cp


crv.knotvector = knot
crv.vis = VisMPL.VisCurve2D()
crv.render()