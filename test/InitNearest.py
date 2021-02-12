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

vis = False
np.random.seed(seed=173)

cases = [1,18,22,23,29,50,52,59,71,73,75,76,78,80,82,97,98,106,109,114,127,128,135,144,166,167,170,171,173,175,179,185,196,203,209]
# cases = [1]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/ValidationData/'

save_pts = []
index = []
knotvector_u = []
knotvector_v = []

cs = int(sys.argv[1])

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

    seg = np.where(seg==2,0,seg).astype(np.uint8)

    contour_pts = tb.get_full_pts(seg)

    pred = np.array(DBSCAN(eps=1).fit_predict(contour_pts))
    num_clu = np.max(pred) + 1

    selected_clu = np.argmax(np.array([np.sum(pred==i) for i in range(num_clu)]))
    not_selected = contour_pts[pred != selected_clu,:]
    removed_seg = seg.copy()

    # print(not_selected.shape[0])

    for i in range(not_selected.shape[0]):
        removed_seg[not_selected[i,0],not_selected[i,1],not_selected[i,2]] = 0

    contour_pts = tb.get_contour_pts(removed_seg)

    center, evecs, radius = ef.ellipsoid_fit(contour_pts)

    if 1.5 * np.sum(removed_seg) < 4/3*math.pi*radius[0]*radius[1]*radius[2]:
        continue

    if vis:
        img_mask = tb.get_image_mask_points(img,contour_pts)

        color_img = tb.draw_segmentation(seg, img_mask,mark_val=255)
        tb.show_ct_image(color_img)

    # print(contour_pts.shape)
    # for i in range(8):
    #     contour_pts_add = contour_pts + np.random.rand(contour_pts.shape[0],contour_pts.shape[1])
    #     contour_pts = np.concatenate([contour_pts,contour_pts_add])

    surf = tb.make_nearest_surf(center,radius,evecs,contour_pts,vis=vis,seg=seg,qsize=cs,psize=40)

    if vis:
        tb.surf_render(surf)
        
        div = 20
        evalpts = tb.evalpts_uv(surf,div,0.00001)
        
        evalpts = evalpts[1:div+1,1:div+1,:]
        evalpts = np.array(evalpts)

        evalpts = np.reshape(evalpts,(div*div,3))
        img_mask = tb.get_image_mask_points(img,evalpts)

        color_img = tb.draw_segmentation(seg, img_mask,mark_val=255)
        tb.show_ct_image(color_img)

    rect_size[0] = 6

    surf = tb.cul_contour_seg(img,seg,surf,rect_size,div=40,N_limit=10,c=0.95,ctrlpts_size=cs,dif_limit=-0.01,dif_abs=False,w=0.95,ctrlpts_increasing=False,increasing_limit=0.005,weight=1.0)

    pts = np.array(surf.ctrlpts)
    pts = np.reshape(pts,(pts.shape[0]*pts.shape[1]))
    print(pts.shape)
    save_pts.append(pts)
    index.append(case)
    knotvector_u.append(surf.knotvector_u)
    knotvector_v.append(surf.knotvector_v)


save_pts = np.array(save_pts)
print(save_pts.shape)

method = 'nearest_axis_surf'

np.save('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/' + method + str(cs) + '_init.npy',save_pts)
np.save('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/' + method + str(cs) + '_case.npy',index)
np.save('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/' + method + str(cs) + '_knotvector_u.npy',knotvector_u)
np.save('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/' + method + str(cs) + '_knotvector_v.npy',knotvector_v)
  