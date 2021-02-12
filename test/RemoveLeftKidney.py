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
from sklearn.neighbors import BallTree


import Toolbox as tb
import ellipsoid_fit as ef

# cases = [5,30,31,55,62,73,75,81,151]
# cases = range(210,299)
# data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/TestData/'

np.random.seed(seed=173)

# cases = range(150)
cases = [40]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'

for case in cases:
    case_str = tb.get_full_case_id(case)
    print(case_str)

    nii0=nib.load(data_path + case_str + '/segmentation.nii.gz')
    seg = nii0.get_data().astype(np.uint8)

    seg = np.where(seg==2,0,seg).astype(np.uint8)

    contour_pts = tb.get_full_pts(seg)

    kmeans = KMeans(n_clusters=2,random_state=173).fit(contour_pts)
    pred = np.array(kmeans.labels_)
    selected_clu = int(kmeans.cluster_centers_[0,2] < kmeans.cluster_centers_[1,2])

    not_selected = contour_pts[pred != selected_clu,:]
    removed_seg = seg.copy()

    for i in range(not_selected.shape[0]):
        removed_seg[not_selected[i,0],not_selected[i,1],not_selected[i,2]] = 0
    
    tb.show_image_collection(removed_seg * 255)

    seg[:,:,:int(seg.shape[2]/2)] = 0
    tb.show_image_collection(seg * 255)

    print(tb.cul_dice(seg,removed_seg))
