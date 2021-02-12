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

cs = int(sys.argv[1])
pca_dim = int(sys.argv[2])

method = sys.argv[3]

init = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/' + method + str(cs) + '_init.npy')

pca = PCA(n_components=pca_dim)
pca.fit(init)

P = np.dot(pca.components_.transpose(),pca.components_)

np.save('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/' + method + str(cs) + '_dim' + str(pca_dim) + '_projection.npy',P)

