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
from sklearn.decomposition import PCA

import Toolbox as tb

def reshaping(array,size_u,size_v):
    array = np.reshape(array,(size_u*size_v,3))

    p1_list = []
    for p in array:
        p1_list.append(list(p))
    
    return p1_list

def projection_surf(P,surf):
    pts = np.array(surf.ctrlpts)
    
    before_pro = np.reshape(pts,(pts.shape[0]*pts.shape[1]))
    after_pro = np.dot(P,before_pro)

    after_pro_f = reshaping(after_pro,surf.ctrlpts_size_u,surf.ctrlpts_size_v)

    surf.set_ctrlpts(after_pro_f,surf.ctrlpts_size_u,surf.ctrlpts_size_v)

    return surf



points = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/membrane_init.npy')
ellipsoid = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/ellipsoid_init.npy')

surf = BSpline.Surface()

surf.degree_u = 3
surf.degree_v = 3

ctrpts = reshaping(points[0,:],8,8)
surf.set_ctrlpts(ctrpts,8,8)

surf.knotvector_u = [0.0, 0.0, 0.0, 0.0, 0.19743454036826685, 0.3274630611338445, 0.6108806095649904, 0.7665449559338162, 1.0, 1.0, 1.0, 1.0]
surf.knotvector_v = [0.0, 0.0, 0.0, 0.0, 0.12251864276679962, 0.3407999431987575, 0.5965610356590559, 0.8289629540770189, 1.0, 1.0, 1.0, 1.0]

pca = PCA(n_components=20)
pca.fit(points)

P = np.dot(pca.components_.transpose(),pca.components_)

tb.surf_render(surf)
surf = projection_surf(P,surf)

tb.surf_render(surf)

np.save('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/Projection_8.npy',P)

