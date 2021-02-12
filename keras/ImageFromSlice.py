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
import random
from PIL import Image


import Toolbox as tb
import ellipsoid_fit as ef

import os

center = 0
width = 300

num_slices = 30

# cases = [1,18,22,23,29,50,52,59,71,73,75,76,78,80,82,97,98,106,109,114,127,128,135,144,166,167,170,171,173,175,179,185,196,203,209]
cases = range(150)
# case = 170
# data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/TestData/'
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/'

os.makedirs(output_path + 'kidney',exist_ok=True)
os.makedirs(output_path + 'notkidney',exist_ok=True)

random.seed(173)

for case in cases:
    case_str = tb.get_full_case_id(case)
    print(case_str)
    nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
    img = nii0.get_data()

    nii0=nib.load(data_path + case_str + '/segmentation.nii.gz')
    seg = nii0.get_data()
    seg = np.where(seg==2,0,seg)
    # seg[:,:,:int(seg.shape[2]/2)] = 0

    img = tb.window_function(img,center,width)

    # print([np.sum(seg[:,:,i]) for i in range(seg.shape[2])])
    # exit(0)

    # x
    limit_x = 1000
    is_kidney_x = [limit_x < np.sum(seg[i,:,:]) for i in range(seg.shape[0])]
    k_slice_x = [i for i in range(seg.shape[0]) if is_kidney_x[i]]
    n_slice_x = [i for i in range(seg.shape[0]) if not is_kidney_x[i]]

    sampling = random.sample(k_slice_x, min(num_slices,len(k_slice_x)))

    for s in sampling:
        pil_img = Image.fromarray(img[s,:,:])
        pil_img.save(output_path + '/kidney/' + str(case) + '_' + str(s) + '_x.png')

    sampling = random.sample(n_slice_x, min(num_slices,len(n_slice_x)))

    for s in sampling:
        pil_img = Image.fromarray(img[s,:,:])
        pil_img.save(output_path + '/notkidney/' + str(case) + '_' + str(s) + '_x.png')

    # y
    limit_y = 1000
    is_kidney_y = [limit_y < np.sum(seg[:,i,:]) for i in range(seg.shape[1])]
    k_slice_y = [i for i in range(seg.shape[1]) if is_kidney_y[i]]
    n_slice_y = [i for i in range(seg.shape[1]) if not is_kidney_y[i]]

    sampling = random.sample(k_slice_y, min(num_slices,len(k_slice_y)))

    for s in sampling:
        pil_img = Image.fromarray(img[:,s,:])
        pil_img.save(output_path + '/kidney/' + str(case) + '_' + str(s) + '_y.png')

    sampling = random.sample(n_slice_y, min(num_slices,len(n_slice_y)))

    for s in sampling:
        pil_img = Image.fromarray(img[:,s,:])
        pil_img.save(output_path + '/notkidney/' + str(case) + '_' + str(s) + '_y.png')

    # z
    limit_z = 1000
    is_kidney_z = [1000 < np.sum(seg[:,:,i]) for i in range(seg.shape[2])]
    k_slice_z = [i for i in range(seg.shape[2]) if is_kidney_z[i]]
    n_slice_z = [i for i in range(seg.shape[2]) if not is_kidney_z[i]]

    sampling = random.sample(k_slice_z, min(num_slices,len(k_slice_z)))

    for s in sampling:
        pil_img = Image.fromarray(img[:,:,s])
        pil_img.save(output_path + '/kidney/' + str(case) + '_' + str(s) + '_z.png')

    sampling = random.sample(n_slice_z, min(num_slices,len(n_slice_z)))

    for s in sampling:
        pil_img = Image.fromarray(img[:,:,s])
        pil_img.save(output_path + '/notkidney/' + str(case) + '_' + str(s) + '_z.png')









