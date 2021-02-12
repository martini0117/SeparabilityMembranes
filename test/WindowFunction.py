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

# cases = [1,18,22,23,29,50,52,59,71,73,75,76,78,80,82,97,98,106,109,114,127,128,135,144,166,167,170,171,173,175,179,185,196,203,209]
case = 170
case_str = tb.get_full_case_id(case)
# data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/TestData/'
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/ValidationData/'

nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
img = nii0.get_data()

nii0=nib.load(data_path + case_str + '/segmentation.nii.gz')
seg = nii0.get_data()

img_tmp = np.where(1 <= seg,img,0)
print(np.mean(img_tmp))
print(np.max(img_tmp))
print(np.min(img_tmp))


center = 0
width = 300
wmax = center + width
wmin = center - width

img = np.where(wmax < img,wmax,img)
img = np.where(img < wmin,wmin,img)

img = (img - wmin) / (wmax - wmin) * 255
img = img.astype(np.uint8)

tb.show_image_collection(img)

