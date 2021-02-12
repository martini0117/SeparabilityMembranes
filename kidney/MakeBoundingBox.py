import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/ellipsoid_fit')
import math

import glob
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import nibabel as nib

import Toolbox as tb

def data_load(path):
    pathes = glob.glob(path + '/*.png')
    pathes.sort()
    # print(pathes)

    images = [cv2.imread(path) for path in pathes]

    return images

# noise 180
cases = [178]
# cases = range(158,160)
# cases = range(166,170)
# cases = range(188,190)
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all'

window_center = 0
window_width = 300

for case in cases:
    case_str = tb.get_full_case_id(case)
    print(case_str)

    nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
    img = nii0.get_data()
    img = tb.window_function(img,window_center,window_width)

    bounding_box = np.zeros((3,2),dtype=np.int32)

    print('x?')
    tb.show_image_collection(img)
    bounding_box[0,:] = np.array([int(x) for x in input().split()])
    print('y?')
    tb.show_image_collection(np.transpose(img,(1,0,2)))
    bounding_box[1,:] = np.array([int(x) for x in input().split()])
    print('z?')
    tb.show_image_collection(np.transpose(img,(2,0,1)))
    bounding_box[2,:] = np.array([int(x) for x in input().split()])

    print(bounding_box)


    bb = np.zeros(img.shape,dtype=np.int32)
    for i in range(bb.shape[0]):
        if i < bounding_box[0,0] or bounding_box[0,1] < i:
            continue 
        X_indices, Y_indices = np.indices(bb.shape[1:])
        bb[i,:,:] = np.where(
                                np.logical_and(
                                    np.logical_and(
                                        X_indices >= bounding_box[1,0],
                                        X_indices <= bounding_box[1,1]
                                    ),
                                    np.logical_and(
                                        Y_indices >= bounding_box[2,0],
                                        Y_indices <= bounding_box[2,1]
                                    )
                                ),
                            1,0)

    tb.show_ct_image(tb.draw_segmentation(img,bb,mark_val=1))
    np.save(output_path + '/' + str(case) + '/bounding_box_h.npy',bounding_box)

    





