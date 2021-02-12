import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
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

import Toolbox as tb

img_mask = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/contour.npy')

tb.show_image_collection(img_mask)

print(np.max(img_mask))

img_contour = np.zeros(img_mask.shape,dtype=np.uint8)


# for i in range(img_mask.shape[0]):
#     contour,_ = cv2.findContours(img_mask[i,:,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     for c in contour:
#         for coord in c:
#             img_contour[i,coord[0][1],coord[0][0]] = 255

for i in range(img_mask.shape[0]):
    contour,_ = cv2.findContours(img_mask[i,:,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_contour[i,:,:] = cv2.fillPoly(img_contour[i,:,:],contour,255)

for j in range(img_mask.shape[1]):
    contour,_ = cv2.findContours(img_mask[:,j,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_contour[:,j,:] = cv2.fillPoly(img_contour[:,j,:],contour,255)

for k in range(img_mask.shape[2]):
    contour,_ = cv2.findContours(img_mask[:,:,k],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_contour[:,:,k] = cv2.fillPoly(img_contour[:,:,k],contour,255).get()

tb.show_image_collection(img_contour)

for i in range(img_mask.shape[0]):
    contour,_ = cv2.findContours(img_contour[i,:,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_contour[i,:,:] = cv2.fillPoly(img_contour[i,:,:],contour,255)



tb.show_image_collection(img_contour)
