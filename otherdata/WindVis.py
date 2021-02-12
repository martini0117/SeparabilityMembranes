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


# cases = range(150,190)
cases = [155]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/volume/windmat/'

u = []
for i in range(15):
    u.append(np.loadtxt(data_path + 'matu/' + str(i) + '.csv', delimiter=','))
u = np.array(u)

v = []
for i in range(15):
    v.append(np.loadtxt(data_path + 'matv/' + str(i) + '.csv', delimiter=','))
v = np.array(v)

w = []
for i in range(15):
    w.append(np.loadtxt(data_path + 'matw/' + str(i) + '.csv', delimiter=','))
w = np.array(w)

magnitude = np.zeros(u.shape)

for i in range(u.shape[0]):
    for j in range(u.shape[1]):
        for k in range(u.shape[2]):
            m = np.array([u[i,j,k],v[i,j,k],w[i,j,k]])
            magnitude[i,j,k] = np.linalg.norm(m)

# tb.show_ct_image(magnitude)

distance = np.zeros(u.shape)

for i in range(u.shape[0]):
    for j in range(u.shape[1]):
        for k in range(u.shape[2]):
            m = np.array([u[i,j,k],v[i,j,k],w[i,j,k]])
            m = m / np.linalg.norm(m)
            diff = np.array([1,0,0]) - m
            distance[i,j,k] = np.linalg.norm(diff)

# tb.show_ct_image(distance)

vs = []
for i in range(u.shape[0]):
    for j in range(u.shape[1]):
        for k in range(u.shape[2]):
            m = np.array([u[i,j,k],v[i,j,k],w[i,j,k]])
            vs.append(m / np.linalg.norm(m))


pcaimg = np.zeros(u.shape)
vs = np.array(vs)

pca = PCA(n_components=1)
pca.fit(vs)

for i in range(u.shape[0]):
    for j in range(u.shape[1]):
        for k in range(u.shape[2]):
            m = np.array([u[i,j,k],v[i,j,k],w[i,j,k]])
            m = m / np.linalg.norm(m)
            m = m.reshape([1,3])
            pcaimg[i,j,k] = pca.transform(m)[0,0]


tb.show_ct_image(pcaimg)