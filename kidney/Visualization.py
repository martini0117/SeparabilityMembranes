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

case = 162
case_str = tb.get_full_case_id(case)

human = '_h'

method = 'NonPCA'

data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/KidneyContour/' + method + '/'

window_center = 0
window_width = 300

nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
img = nii0.get_data()
img = tb.window_function(img,window_center,window_width)

nii0=nib.load(data_path + case_str + '/contour_img' + human + '.nii.gz')
init = nii0.get_data().astype(np.uint8)

nii0=nib.load(output_path + case_str + '/prediction_contour' + human + '.nii.gz')
contour = nii0.get_data().astype(np.uint8)


# tb.show_image_collection(init)
# tb.show_image_collection(contour)

# tb.show_image_collection(img)

tb.show_ct_image(tb.draw_segmentation(img,init,mark_val=255))
tb.show_ct_image(tb.draw_segmentation(img,contour,mark_val=255))

