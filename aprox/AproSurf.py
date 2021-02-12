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
radius = [10, 5, 20]
n_sp = 40
degree = 3

sp = np.array([[radius[0]*math.cos(u)*math.cos(v), radius[1]*math.cos(v)*math.sin(u),radius[2]*math.sin(v)] 
                    for u in np.linspace(0,2*math.pi,num=n_sp) 
                    for v in np.linspace(-math.pi/2+0.01,math.pi/2-0.01,num=n_sp)])
u = np.linspace(0,1,n_sp)
v = np.linspace(0,1,n_sp)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot(sp[:,0],sp[:,1],sp[:,2],'o',markersize=2)
# plt.show()

knot = generate(degree, n_cp)

A = np.array([[basis_function_one(degree, knot,k,u[i]) * basis_function_one(degree, knot,l,v[j]) 
                for k in range(n_cp) for l in range(n_cp)]
                for i in range(n_sp) for j in range(n_sp)])
print(A.shape)

invATA = np.linalg.inv(np.dot(A.T, A))
cp = np.dot(np.dot(invATA, A.T), sp).tolist()

print(cp)

surf = BSpline.Surface()
surf.degree_u = degree
surf.degree_v = degree

surf.set_ctrlpts(cp,n_cp,n_cp)

surf.knotvector_u = knot
surf.knotvector_v = knot

tb.surf_render(surf)