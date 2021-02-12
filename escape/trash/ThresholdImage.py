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

case = 'case_00030'

nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Data/' + case + '/imaging.nii.gz')
img = nii0.get_data()

# tb.show_ct_image(img)


nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Data/' + case + '/kidney.nii.gz')
seg = nii0.get_data().astype(np.uint8)

clipped_img = img[np.where(seg == 1)]
threshold_max = np.max(clipped_img)
threshold_min = np.min(clipped_img)

print(threshold_max)
print(threshold_min)

threshold_max = 200
threshold_min = -80



img = np.where(np.logical_and(threshold_min <= img,img <= threshold_max),img,0)

tb.show_ct_image(img)






