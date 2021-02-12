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

import Toolbox as tb

# 心臓
# nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Task02_Heart/imagesTr/la_003.nii.gz')
# inside = nii0.get_data().astype(np.float32)

# rect_size = np.array([20,6,6])
# center = np.array([145,175,75])
# radius = 25

# 肺
# nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/finding-lungs-in-ct-data/3d_images/IMG_0002.nii.gz')
# inside = nii0.get_data().astype(np.float32)

# rect_size = np.array([35,6,6])
# center = np.array([150,270,160])
# radius = np.array([140,120,100])


# 脾臓
nii0 = nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Task09_Spleen/imagesTr/spleen_2.nii.gz')
inside = nii0.get_data().astype(np.float32).transpose(2,0,1)

rect_size = np.array([30,6,6])
center = np.array([77,145,200])
radius = np.array([10,60,70])

# 前立腺
# nii0 = nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Task05_Prostate/labelsTr/prostate_00.nii.gz')
# inside = nii0.get_data().astype(np.float32).transpose(2,0,1)

# nii0 = nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Task05_Prostate/imagesTr/prostate_00.nii.gz')

# inside = nii0.get_data().astype(np.float32).transpose(2,0,1,3)
# inside = inside[:,:,:,1]

# rect_size = np.array([20,6,6])
# center = np.array([7,155,175])
# radius = np.array([5,35,25])

# img = tb.load_test_medical_image()
# tb.show_ct_image(img)

margin = rect_size[0]
center += margin

# tb.show_ct_image(inside)

img_shape = tuple([s + 2*margin for s in inside.shape])

img = np.zeros(img_shape)
img += np.min(inside)
img[margin:img_shape[0]-margin,margin:img_shape[1]-margin,margin:img_shape[2]-margin] += inside - np.min(inside)

tb.show_ct_image(img)


surf = tb.make_ellipsoid_surf(center,radius)

surf = tb.cul_contour(img,surf,rect_size,div=20,N_limit=15,c=0.95,ctrlpts_size=8,dif_limit=0.001,w=0.95)

surf.delta = 0.05
surf.vis = VisMPL.VisSurfWireframe()
surf.render()

surf.delta = 0.0025
surface_points = np.array(surf.evalpts)

img_mask = np.zeros(img.shape,dtype=np.uint8)

indices = surface_points.astype(np.int32)
img_mask[indices[:,0],indices[:,1],indices[:,2]] = 255

# img_mask = flood_fill(img_mask, tuple(center),255)

# tb.show_image_collection(img_mask)

color_img = np.zeros((img.shape[0],img.shape[1],img.shape[2],3))
color_img[:,:,:,0] = np.where(img_mask==255,1,img)
color_img[:,:,:,1] = np.where(img_mask==255,0,img)
color_img[:,:,:,2] = np.where(img_mask==255,0,img)

# color_img[center[0],center[1],center[2],2] = 1
color_img[center[0]-5:center[0]+5,center[1]-5:center[1]+5,center[2]-5:center[2]+5,2] = 1

tb.show_ct_image(color_img)


