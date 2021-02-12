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

import Toolbox as tb
import ellipsoid_fit as ef

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

from sklearn.cluster import AgglomerativeClustering

case = 'case_00005'



nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Data/' + case + '/segmentation.nii.gz')
gt = nii0.get_data().astype(np.uint8)

nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Data/' + case + '/tumor.nii.gz')
seg = nii0.get_data().astype(np.uint8)

tb.show_ct_image(seg)

nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Data/' + case + '/imaging.nii.gz')
img = nii0.get_data()

# img = tb.threshold_image_seg(img,seg)
# t_min = -80
# t_max = 200
# img = tb.threshold_image_minmax(img,t_min,t_max)

# tb.show_ct_image(img)

rect_size = np.array([10,3,3])
margin = rect_size[0]
img = tb.add_margin(img, margin)
seg = tb.add_margin(seg, margin)

seg = np.where(seg==2,1,0).astype(np.uint8)

contour_pts = tb.get_full_pts(seg)
print(contour_pts.shape)

# num_clu = 1
# pred = KMeans(n_clusters=num_clu,random_state=173).fit_predict(contour_pts)

# pred = AffinityPropagation().fit_predict(contour_pts)
# pred = MeanShift().fit_predict(contour_pts)
# pred = AgglomerativeClustering(n_clusters=None,distance_threshold=1).fit_predict(contour_pts)
pred = np.array(DBSCAN(eps=1,min_samples=5).fit_predict(contour_pts))
# pred = np.array(OPTICS(min_samples=5,max_eps=1).fit_predict(contour_pts))

num_clu = np.max(pred) + 1

selected_clu = np.argmax(np.array([np.sum(pred==i) for i in range(num_clu)]))

not_selected = contour_pts[pred != selected_clu,:]

removed_seg = seg.copy()

print(not_selected.shape[0])

for i in range(not_selected.shape[0]):
    removed_seg[not_selected[i,0],not_selected[i,1],not_selected[i,2]] = 0

tb.show_ct_image(removed_seg)
contour_pts = tb.get_contour_pts(removed_seg)

