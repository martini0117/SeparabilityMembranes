import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/ellipsoid_fit')
import math
import os

import matplotlib
import pickle

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

def saveImages(img,surf,seg_original,removed_seg,case_str,output_path,removed_output_path):    
    img_mask =tb.get_image_mask(img,surf)
    con_seg = tb.fill_contour(img_mask)

    save_con_seg = con_seg[margin:con_seg.shape[0]-margin,margin:con_seg.shape[1]-margin,margin:con_seg.shape[2]-margin]
    save_con_seg_c = np.where(save_con_seg==255, 2, seg_original)

    nii1 = nib.Nifti1Image(save_con_seg_c,affine=None)
    nib.save(nii1,output_path + case_str + '/prediction_contour.nii.gz')

    removed_seg = removed_seg[margin:con_seg.shape[0]-margin,margin:con_seg.shape[1]-margin,margin:con_seg.shape[2]-margin]
    seg_original = np.where(removed_seg==1,1,seg_original)

    save_con_seg_c = np.where(save_con_seg==255, 2, seg_original)

    nii1 = nib.Nifti1Image(save_con_seg_c,affine=None)
    nib.save(nii1,removed_output_path + case_str + '/prediction_contour.nii.gz')


cases = [1,18,22,23,29,50,52,59,71,73,75,76,78,80,82,97,98,106,109,114,127,128,135,144,166,167,170,171,173,175,179,185,196,203,209]
# cases = [18]

weight = 5
method = 'pca_loop'
pca_dim = int(sys.argv[2])
min_dim = 6
max_dim = 20

data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/ValidationData/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/' + method + '_dim' + str(pca_dim) + '_W' + str(weight) + 'Data/'

case = int(sys.argv[1])

case_str = tb.get_full_case_id(case)

os.makedirs(output_path+case_str,exist_ok=True)

# fo = open(output_path+case_str+'/output.txt', 'w')
# sys.stdout = fo

print(case_str)

nii0=nib.load(data_path + case_str + '/prediction.nii.gz')
seg_original = nii0.get_data().astype(np.uint8)

nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
img = nii0.get_data()

rect_size = np.array([30,3,3])
margin = rect_size[0]
img = tb.add_margin(img, margin)
seg = tb.add_margin(seg_original, margin)

seg = np.where(seg==2,1,0).astype(np.uint8)

contour_pts = tb.get_full_pts(seg)
if np.sum(seg) < 50:
    sys.exit(0)

pred = np.array(DBSCAN(eps=1).fit_predict(contour_pts))
num_clu = np.max(pred) + 1

selected_clu = np.argmax(np.array([np.sum(pred==i) for i in range(num_clu)]))
not_selected = contour_pts[pred != selected_clu,:]
removed_seg = seg.copy()

# print(not_selected.shape[0])

for i in range(not_selected.shape[0]):
    removed_seg[not_selected[i,0],not_selected[i,1],not_selected[i,2]] = 0

contour_pts, contour_img = tb.get_contour_pts_img(removed_seg)

# contour_img = contour_img[margin:contour_img.shape[0]-margin,margin:contour_img.shape[1]-margin,margin:contour_img.shape[2]-margin]
# nii1 = nib.Nifti1Image(contour_img,affine=None)
# nib.save(nii1,data_path + case_str + '/contour_img.nii.gz')

center, evecs, radii = ef.ellipsoid_fit(contour_pts)
print(center)
print(radii)

print(4/3*math.pi*radii[0]*radii[1]*radii[2])
print(np.sum(removed_seg))

rect_size[0] = int(10 + 5*math.pow(np.sum(removed_seg)/500000,1/3))
# rect_size = np.array([10,3,3])

if 1.5 * np.sum(removed_seg) < 4/3*math.pi*radii[0]*radii[1]*radii[2]:
    sys.exit(0)

surf = tb.make_ellipsoid_axis_surf(center,radii,evecs)

contour_img = tb.get_image_mask(img,surf)
contour_img = contour_img[margin:contour_img.shape[0]-margin,margin:contour_img.shape[1]-margin,margin:contour_img.shape[2]-margin]
nii1 = nib.Nifti1Image(contour_img,affine=None)
nib.save(nii1,data_path + case_str + '/contour_img.nii.gz')

cs = int(10 + 4*math.pow(np.sum(removed_seg)/500000,1/3))
# cs = 6

P = tb.load_projection_matrix('nearest_axis',pca_dim)

loop = 5
for i in range(loop):
    surf = tb.cul_contour_seg(img,seg,surf,rect_size,restraint=False,div=40,N_limit=10,c=0.95,ctrlpts_size=cs,dif_limit=-0.01,dif_abs=False,w=0.95,ctrlpts_increasing=True,increasing_limit=0.005,weight=0.1*weight)
    f = open(output_path+case_str+'/loop'+str(i),'wb')
    pickle.dump(surf,f)
    f.close
    cs = surf.ctrlpts_size_u
    surf = tb.projection_surf(P[cs-min_dim],surf)
