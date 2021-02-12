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

import os

import Toolbox as tb
import ellipsoid_fit as ef

np.random.seed(seed=173)

#160,181,189
cases = range(150,190)
# cases = range(150,160)
# cases = [185]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/KidneyContour/NonPCA/'

window_center = 0
window_width = 300

human = '_h'

#射影行列の読み込み
mindim = 6
maxdim = 20

pca_dim = 60

P = []
for i in range(mindim,maxdim+1):
    Pi = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/KidneyContour/init/Nearest/' + str(i) + '_dim' + str(pca_dim) + '_projection.npy')
    P.append(Pi)

for case in cases:
    case_str = tb.get_full_case_id(case)
    print(case_str)

    # データ読み込み
    nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
    img = nii0.get_data()
    img = tb.window_function(img,window_center,window_width)

    rect_size = np.array([30,4,4])
    margin = rect_size[0]
    img = tb.add_margin(img, margin)

    # 初期輪郭取得
    center = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_center' + human + '.npy')
    radii = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_radius' + human + '.npy')
    evecs = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_rotation' + human + '.npy')
    points = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_points' + human + '.npy')
    center += margin
    points += margin


    surf = tb.make_ellipsoid_axis_surf(center,radii,evecs)
    init = tb.draw_contour_u(img,surf,color=[255,255,0])
    # tb.save_image_slice(init[:,235:535,315:615],76, 'kidney_example_initial_1')
    # tb.show_image_collection(init)

    # 初期輪郭の保存
    contour_img = tb.get_image_mask(img,surf)
    contour_img = contour_img[margin:contour_img.shape[0]-margin,margin:contour_img.shape[1]-margin,margin:contour_img.shape[2]-margin]
    nii1 = nib.Nifti1Image(contour_img,affine=None)
    nib.save(nii1,data_path + case_str + '/contour_img' + human + '.nii.gz')

    surf = tb.make_nearest_surf(center,radii,evecs,points)

    contour_img = tb.get_image_mask(img,surf)
    contour_img = contour_img[margin:contour_img.shape[0]-margin,margin:contour_img.shape[1]-margin,margin:contour_img.shape[2]-margin]
    nii1 = nib.Nifti1Image(contour_img,affine=None)
    nib.save(nii1,data_path + case_str + '/contour_img_nearest' + human + '.nii.gz')
    # init = tb.draw_contour_u(img,surf,color=[255,255,255])
    # tb.surf_render(surf)
    # surf = tb.projection_surf(P[8-mindim],surf)
    # tb.surf_render(surf)

    continue

    print(center)
    print(radii)

    # 最終輪郭の取得
    cs = 8
    rect_size[0] = 20
    surf = tb.cul_contour(img,surf,rect_size,div=40,N_limit=20,c=0.95,ctrlpts_size=cs,dif_limit=-0.01,dif_abs=False,w=0.95,ctrlpts_increasing=True,increasing_limit=0.01)
    
    # tb.surf_render(surf)

    final = tb.draw_contour_u(img,surf,color=[255,255,0])
    # final = tb.draw_contour_thick(img,surf,color=[255,255,255])
    tb.show_image_collection(final)
    # tb.save_image_slice(final[:,235:535,315:615],76, 'kidney_example_final_1')
    
    # surf = tb.projection_surf(P[surf.ctrlpts_size_u-mindim],surf)
    
    # surf = tb.cul_contour_pca(img,surf,rect_size,P,div=40,N_limit=23,c=0.95,ctrlpts_size=cs,dif_limit=-0.01,w=0.95,increasing_limit=0.02,step=3)

    # 輪郭の保存
    con_seg = tb.get_image_mask(img,surf)
    save_con_seg = con_seg[margin:con_seg.shape[0]-margin,margin:con_seg.shape[1]-margin,margin:con_seg.shape[2]-margin]
    os.makedirs(output_path + case_str,exist_ok=True)
    nii1 = nib.Nifti1Image(save_con_seg,affine=None)
    nib.save(nii1,output_path + case_str + '/prediction_contour' + human + '.nii.gz')

