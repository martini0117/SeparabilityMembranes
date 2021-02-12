import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from skimage import morphology
from scipy import ndimage as ndi
from skimage.viewer import ImageViewer
from skimage.viewer import CollectionViewer
from geomdl import BSpline
from geomdl.fitting import approximate_surface
from geomdl.knotvector import generate
import geomdl.visualization.VisMPL as VisMPL
import nibabel as nib
from pathlib import Path
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import csv

import Toolbox as tb

# sn = 2
# sigma = 50/sn

clipped_image = np.zeros((60,30,30))
clipped_image[:20,:,:] += 255
# clipped_image[29:31,:,:] -= 25 

# clipped_image += np.random.normal(0,sigma,clipped_image.shape)
clipped_image = clipped_image.astype(np.uint8)
tb.show_image_collection(clipped_image.transpose(2,0,1))

# resized_clip = np.zeros((clipped_image.shape[0]*10,clipped_image.shape[1]*10,1),dtype=np.uint8)

# resized_clip[:,:,0] = cv2.resize(clipped_image[:,:,0],None,fx=10,fy=10,interpolation=cv2.INTER_NEAREST)
# tb.save_image_slice(resized_clip.transpose(2,0,1),0,'Separability_Image')


w,h,d = clipped_image.shape
N = w*h*d

variance_all = N * np.var(clipped_image)
average_all = np.mean(clipped_image)
variance_boundary = [(i+1)*h*d*(np.mean(clipped_image[:i+1,:,:])-average_all) ** 2
                   + (w-i-1)*h*d*(np.mean(clipped_image[i+1:,:,:])-average_all) ** 2
                    for i in range(w-1)]

separability = variance_boundary/variance_all

print(separability)

plt.plot(separability)
plt.show()

with open('/Users/shomakitamiya/Documents/python/snake3D/data/exp/separability_graph.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(separability)