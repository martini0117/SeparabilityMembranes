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

def data_load(path):
    pathes = glob.glob(path + '/*.png')

    images = [cv2.imread(path) for path in pathes]

    return images

# データのロード

kidney = data_load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/kidney')
not_kidney = data_load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/notkidney')

print(len(kidney))
print(len(not_kidney))

k_label = [1 for i in range(len(kidney))]
n_label = [0 for i in range(len(not_kidney))]

X = kidney + not_kidney
y = k_label + n_label

l = list(zip(X,y))
np.random.shuffle(l)

X, y = zip(*l)

(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)

x_train = [x_train[i].astype(np.float32)/255 for i in range(len(x_train))]
x_test = [x_test[i].astype(np.float32)/255 for i in range(len(x_test))]

    