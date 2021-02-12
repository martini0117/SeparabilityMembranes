from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from spp.SpatialPyramidPooling import SpatialPyramidPooling
import glob
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pickle
import random
import matplotlib.pyplot as plt
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def data_load(path):
    pathes = glob.glob(path + '/*.png')
    pathes.sort()
    # print(pathes)

    images = [cv2.imread(path) for path in pathes]

    return images

vis = False
one_model = True

# cases = [189]
cases = range(150,190)
# cases = range(160,190)
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all'
if one_model:
    model = load_model('/Users/shomakitamiya/Documents/python/snake3D/src/keras/more_small_model.h5',custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})
else:
    model = []
    model.append(load_model('/Users/shomakitamiya/Documents/python/snake3D/src/keras/small_model_x.h5',custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling}))
    model.append(load_model('/Users/shomakitamiya/Documents/python/snake3D/src/keras/small_model_y.h5',custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling}))
    model.append(load_model('/Users/shomakitamiya/Documents/python/snake3D/src/keras/small_model_z.h5',custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling}))

for case in cases:

    slices = []
    slices.append(np.array(data_load(output_path + '/' + str(case) + '/x')))
    slices.append(np.array(data_load(output_path + '/' + str(case) + '/y')))
    slices.append(np.array(data_load(output_path + '/' + str(case) + '/z')))

    if one_model:
        preds = [np.squeeze(model.predict(slices[i])) for i in range(3)]
    else:
        preds = [np.squeeze(model[i].predict(slices[i])) for i in range(3)]
    preds[2][:int(preds[2].shape[0]/2)] = 0

    indice = [np.where(0.5 <= preds[i])[0] for i in range(3)]

    if len(indice[0]) == 0 or len(indice[1]) == 0 or len(indice[2]) == 0:
        continue

    bounding_box = np.zeros((3,2),dtype=np.int32)

    # kmeans = KMeans(n_clusters=2, random_state=173).fit(indice[2].reshape(-1,1))
    # right_label = int(kmeans.cluster_centers_[0] < kmeans.cluster_centers_[1])
    # indice[2] = indice[2][np.where(kmeans.labels_ == right_label)[0]]
    
    for i, index in enumerate(indice):
        start = 0
        for j in range(1,index.shape[0]):

            if index[j-1] + 1 != index[j]:
                if bounding_box[i,1] - bounding_box[i,0] + 1 < j - start:
                    bounding_box[i,0] = index[start]
                    bounding_box[i,1] = index[j - 1]
        
                start = j
        j = index.shape[0]
        if bounding_box[i,1] - bounding_box[i,0] + 1 < j - start:
            bounding_box[i,0] = index[start]
            bounding_box[i,1] = index[j - 1]
            start = j

    if vis:
        for i in range(3):
            plt.plot(preds[i])
            plt.show()

            pred = np.zeros(preds[i].shape)
            bb = slice(bounding_box[i,0],bounding_box[i,1]+1)
            pred[bb] = preds[i][bb]
            plt.plot(pred)
            plt.show()

    np.save(output_path + '/' + str(case) + '/bounding_box.npy',bounding_box)

    print(bounding_box)





