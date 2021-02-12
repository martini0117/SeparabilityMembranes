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
import glob


import Toolbox as tb
import ellipsoid_fit as ef

import os

def datasize_load(path):
    pathes = glob.glob(path + '/*.png')

    images = np.array([Image.open(path).size for path in pathes])

    return images

center = 0
width = 300

num_slices = 30

# cases = [1,18,22,23,29,50,52,59,71,73,75,76,78,80,82,97,98,106,109,114,127,128,135,144,166,167,170,171,173,175,179,185,196,203,209]
# cases = range(150)
# cases = [152]
cases = range(150,198)
# case = 170
# data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/TestData/'
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/'

resolution = datasize_load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/Resolution/notkidney')
kmeans = KMeans(n_clusters=10, random_state=173).fit(resolution)

random.seed(173)

for case in cases:
    case_str = tb.get_full_case_id(case)
    print(case_str)
    nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
    img = nii0.get_data()

    img = tb.window_function(img,center,width)

    # x
    os.makedirs(output_path + '/all/' + str(case) + '/x',exist_ok=True)
    for s in range(img.shape[0]):
        pil_img = Image.fromarray(img[s,:,:])
        size_label = kmeans.predict(np.array([pil_img.size]))[0]
        pil_img = pil_img.resize(tuple(kmeans.cluster_centers_[size_label,:].astype(np.int32)))
        pil_img.save(output_path + '/all/' + str(case) + '/x/' + str(s).zfill(3) + '.png')

    # y
    os.makedirs(output_path + '/all/' + str(case) + '/y',exist_ok=True)
    for s in range(img.shape[1]):
        pil_img = Image.fromarray(img[:,s,:])
        size_label = kmeans.predict(np.array([pil_img.size]))[0]
        pil_img = pil_img.resize(tuple(kmeans.cluster_centers_[size_label,:].astype(np.int32)))
        pil_img.save(output_path + '/all/' + str(case) + '/y/' + str(s).zfill(3) + '.png')

    # z
    os.makedirs(output_path + '/all/' + str(case) + '/z',exist_ok=True)
    for s in range(img.shape[2]):
        pil_img = Image.fromarray(img[:,:,s])
        size_label = kmeans.predict(np.array([pil_img.size]))[0]
        pil_img = pil_img.resize(tuple(kmeans.cluster_centers_[size_label,:].astype(np.int32)))
        pil_img.save(output_path + '/all/' + str(case) + '/z/' + str(s).zfill(3) + '.png')









