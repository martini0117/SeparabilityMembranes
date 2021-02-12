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

#CNNの推定結果を初期輪郭として分離度膜を適用するコード

removed = True

cases = [1,18,22,23,29,50,52,59,71,73,75,76,78,80,82,97,98,106,109,114,127,128,135,144,166,167,170,171,173,175,179,185,196,203,209]
# cases = [18]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/ValidationData/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/W5Data/'
removed_output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/RemovedW5Data/'

for case in cases:
    case_str = tb.get_full_case_id(case)
    print(case_str)

    nii0=nib.load(data_path + case_str + '/prediction.nii.gz')
    seg_original = nii0.get_data().astype(np.uint8)

    nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
    img = nii0.get_data()

    # img = tb.threshold_image_seg(img,seg)
    # t_min = -80
    # t_max = 200
    # img = tb.threshold_image_minmax(img,t_min,t_max)

    # tb.show_ct_image(img)

    rect_size = np.array([30,3,3])
    margin = rect_size[0]
    img = tb.add_margin(img, margin)
    seg = tb.add_margin(seg_original, margin)

    seg = np.where(seg==2,1,0).astype(np.uint8)

    # contour_pts = tb.get_contour_pts(seg)

    # pred = np.array(DBSCAN(eps=5).fit_predict(contour_pts))
    # num_clu = np.max(pred) + 1

    # c_no = np.argmax(np.array([np.sum(pred==i) for i in range(num_clu)]))

    contour_pts = tb.get_full_pts(seg)
    if np.sum(seg) < 50:
        print(case_str + ' is skipped because seg is all zero.')
        nii1 = nib.Nifti1Image(seg_original,affine=None)
        nib.save(nii1,removed_output_path + case_str + '/prediction_contour.nii.gz')
        nib.save(nii1,output_path + case_str + '/prediction_contour.nii.gz')
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
        print(case_str + ' is skipped.')
        nii1 = nib.Nifti1Image(seg_original,affine=None)
        nib.save(nii1,removed_output_path + case_str + '/prediction_contour.nii.gz')
        nib.save(nii1,output_path + case_str + '/prediction_contour.nii.gz')
        continue

    surf = tb.make_ellipsoid_axis_surf(center,radii,evecs)

    contour_img = tb.get_image_mask(img,surf)
    contour_img = contour_img[margin:contour_img.shape[0]-margin,margin:contour_img.shape[1]-margin,margin:contour_img.shape[2]-margin]
    nii1 = nib.Nifti1Image(contour_img,affine=None)
    nib.save(nii1,data_path + case_str + '/contour_img.nii.gz')

    cs = int(10 + 8*math.pow(np.sum(removed_seg)/500000,1/3))

    surf = tb.cul_contour_seg(img,seg,surf,rect_size,div=40,N_limit=30,c=0.95,ctrlpts_size=cs,dif_limit=-0.01,dif_abs=False,w=0.95,ctrlpts_increasing=True,increasing_limit=0.01,weight=0.5)

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

    print(np.sum(img_mask))

    # if np.sum(save_con_seg==2)/np.sum(seg_original==2) < 1.1:
    #     nii1 = nib.Nifti1Image(save_con_seg,affine=None)
    #     nib.save(nii1,data_path + case_str + '/prediction_contour.nii.gz')
    # else:
    #     print(case_str + ' is skipped because diff is too big.')
    #     nii1 = nib.Nifti1Image(seg_original,affine=None)
    #     nib.save(nii1,data_path + case_str + '/prediction_contour.nii.gz')


    # print(np.sum(save_con_seg==2)/np.sum(seg_original==2))

    # tb.show_ct_image(tb.draw_segmentation(img,con_seg,mark_val=255))
