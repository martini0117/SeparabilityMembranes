import glob
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys

import os

def datasize_load(path):
    pathes = glob.glob(path + '/*.png')

    images = np.array([Image.open(path).size for path in pathes])

    return images

def data_load(path,xyz):
    pathes = glob.glob(path + '/*_' + xyz +'.png')

    images = [Image.open(path) for path in pathes]

    return images

output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/resized'

# データのロード

resolution = datasize_load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/Resolution/notkidney')
kmeans = KMeans(n_clusters=10, random_state=173).fit(resolution)

# plt.scatter(resolution[:,0], resolution[:,1])
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1])
# plt.show()

kn = sys.argv[1]
xyz = sys.argv[2]

if kn == 'k':
    kidney = data_load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/kidney',xyz)

    k_res = np.array([k.size for k in kidney])
    k_label = kmeans.predict(k_res)

    for i,(l,k) in enumerate(zip(k_label,kidney)):
        k = k.resize(tuple(kmeans.cluster_centers_[l,:].astype(np.int32)))
        os.makedirs(output_path + '/kidney/' + str(l),exist_ok=True)
        k.save(output_path + '/kidney/' + str(l) + '/' + str(i) + '_' + xyz + '.png')
else:
    notkidney = data_load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/notkidney',xyz)

    n_res = np.array([n.size for n in notkidney])
    n_label = kmeans.predict(n_res)

    for i,(l,n) in enumerate(zip(n_label,notkidney)):
        n = n.resize(tuple(kmeans.cluster_centers_[l,:].astype(np.int32)))
        os.makedirs(output_path + '/notkidney/' + str(l),exist_ok=True)
        n.save(output_path + '/notkidney/' + str(l) + '/' + str(i) + '_' + xyz + '.png')
