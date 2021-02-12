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

def data_load(path):
    pathes = glob.glob(path + '/*.png')

    images = [cv2.imread(path) for path in pathes]

    return images


batch_size = 32
epochs = 1

load_target = range(10)

# データのロード
Xs = []
ys = []

for i in load_target:
    kidney = data_load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/resized/kidney/' + str(i))
    not_kidney = data_load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/resized/notkidney/' + str(i))


    # kidney = data_load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/sample/kidney/' + str(i))
    # not_kidney = data_load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/sample/notkidney/' + str(i))

    k_label = [1 for i in range(len(kidney))]
    n_label = [0 for i in range(len(not_kidney))]

    X = kidney + not_kidney
    y = k_label + n_label

    l = list(zip(X,y))
    np.random.shuffle(l)

    X, y = zip(*l)

    X = np.array(X)
    y = np.array(y)

    # X = X.astype(np.float32) / 255

    Xs.append(X)
    ys.append(y)

# x_train = [x_train[i].astype(np.float32)/255 for i in range(len(x_train))]
# x_test = [x_test[i].astype(np.float32)/255 for i in range(len(x_test))]

print('Data is ready.')

model = load_model('/Users/shomakitamiya/Documents/python/snake3D/src/keras/more_small_model.h5',custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})

for _ in range(epochs):
    l = list(range(len(load_target)))
    random.shuffle(l)
    for i in l:
        model.fit(Xs[i], ys[i],
                batch_size=batch_size,
                epochs=1,
                verbose=1,
                validation_split=0.1,
                )

model.save('/Users/shomakitamiya/Documents/python/snake3D/src/keras/more_small_model.h5')
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])