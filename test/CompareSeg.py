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

case = 18
case_str = tb.get_full_case_id(case)
# data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/TestData/'
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/ValidationData/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/PCARemovedW5Data/'


nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
img = nii0.get_data()

nii0=nib.load(data_path + case_str + '/segmentation.nii.gz')
gt = nii0.get_data().astype(np.uint8)

nii0=nib.load(data_path + case_str + '/prediction.nii.gz')
seg = nii0.get_data().astype(np.uint8)

nii1 = nib.load(output_path + case_str + '/prediction_contour.nii.gz')
con_seg = nii1.get_data()

nii1 = nib.load(data_path + case_str + '/contour_img.nii.gz')
contour_img = nii1.get_data()
# margin = 30
# contour_img = contour_img[margin:contour_img.shape[0]-margin,margin:contour_img.shape[1]-margin,margin:contour_img.shape[2]-margin]

# con_seg = con_seg[15:con_seg.shape[0]-15,15:con_seg.shape[1]-15,15:con_seg.shape[2]-15]

# nii1 = nib.Nifti1Image(con_seg,affine=None)
# nib.save(nii1,'/Users/shomakitamiya/Documents/python/snake3D/data/Data/' + case + '/contour_seg.nii.gz')

print(con_seg.shape)
print(con_seg.dtype)
print(np.max(con_seg))

# tb.show_ct_image(gt)
# tb.show_ct_image(seg)
# tb.show_ct_image(con_seg)

tb.show_ct_image(tb.draw_segmentation(img,gt,mark_val=2))
tb.show_ct_image(tb.draw_segmentation(img,seg,mark_val=2))
# tb.show_ct_image(tb.draw_segmentation(img,contour_img,mark_val=255))
tb.show_ct_image(tb.draw_segmentation(img,con_seg,mark_val=2))

