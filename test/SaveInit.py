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

cases = [1,18,22,23,29,50,52,59,71,73,75,76,78,80,82,97,98,106,109,114,127,128,135,144,166,167,170,171,173,175,179,185,196,203,209]
# cases = [18]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/ValidationData/'

save_pts = []
index = []
knotvector_u = []
knotvector_v = []

for case in cases:
    case_str = tb.get_full_case_id(case)
    print(case_str)

    nii0=nib.load(data_path + case_str + '/segmentation.nii.gz')
    seg = nii0.get_data().astype(np.uint8)
    img = nii0.get_data().astype(np.uint8)

    rect_size = np.array([30,3,3])
    margin = rect_size[0]
    img = tb.add_margin(img, margin)
    seg = tb.add_margin(seg, margin)

    seg = np.where(seg==2,1,0).astype(np.uint8)

    contour_pts = tb.get_full_pts(seg)
    if np.sum(seg) < 50:
        continue

    pred = np.array(DBSCAN(eps=1).fit_predict(contour_pts))
    num_clu = np.max(pred) + 1

    selected_clu = np.argmax(np.array([np.sum(pred==i) for i in range(num_clu)]))
    not_selected = contour_pts[pred != selected_clu,:]
    removed_seg = seg.copy()

    # print(not_selected.shape[0])

    for i in range(not_selected.shape[0]):
        removed_seg[not_selected[i,0],not_selected[i,1],not_selected[i,2]] = 0

    contour_pts, contour_img = tb.get_contour_pts_img(removed_seg)

    center, evecs, radii = ef.ellipsoid_fit(contour_pts)

    if 1.5 * np.sum(removed_seg) < 4/3*math.pi*radii[0]*radii[1]*radii[2]:
        continue

    surf = tb.make_ellipsoid_axis_surf(center,radii,evecs)

    # print(surf.knotvector_u)
    # print(surf.knotvector_v)

    cs = 8
    rect_size[0] = int(10 + 5*math.pow(np.sum(removed_seg)/500000,1/3))

    surf = tb.cul_contour_seg(img,seg,surf,rect_size,div=20,N_limit=10,c=0.95,ctrlpts_size=cs,dif_limit=-0.01,dif_abs=False,w=0.9,ctrlpts_increasing=False,increasing_limit=0.005,weight=1.0)

    pts = np.array(surf.ctrlpts)
    pts = np.reshape(pts,(pts.shape[0]*pts.shape[1]))
    print(pts.shape)
    save_pts.append(pts)
    index.append(case)
    knotvector_u.append(surf.knotvector_u)
    knotvector_v.append(surf.knotvector_v)


save_pts = np.array(save_pts)
index = np.array(index)
knotvector_u = np.array(knotvector_u)
knotvector_v = np.array(knotvector_v)

print(save_pts.shape)
print(index.shape)

np.save('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/membrane' + str(cs) + '_init.npy',save_pts)
np.save('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/membrane' + str(cs) + '_case.npy',index)
np.save('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/membrane' + str(cs) + '_knotvector_u.npy',knotvector_u)
np.save('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/membrane' + str(cs) + '_knotvector_v.npy',knotvector_v)
