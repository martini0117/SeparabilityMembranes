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

import Toolbox as tb
import ellipsoid_fit as ef

from skimage.filters import threshold_otsu

from mpl_toolkits.mplot3d import Axes3D

import ellipsoid_fit as ef

vis = True
# human = ''
human = '_h'

# cases = range(150,160)
# cases = range(150,190)
cases = [152]
data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'

window_center = 0
window_width = 300

for case in cases:
    case_str = tb.get_full_case_id(case)
    print(case_str)

    nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
    img = nii0.get_data()
    img = tb.window_function(img,window_center,window_width)

    bounding_box = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bounding_box' + human + '.npy')

    bb = np.zeros(img.shape,dtype=np.uint8)
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
                            
    x_range = slice(bounding_box[0,0],bounding_box[0,1])
    y_range = slice(bounding_box[1,0],bounding_box[1,1])
    z_range = slice(bounding_box[2,0],bounding_box[2,1])

    clip = img[x_range,y_range,z_range]

    clip_binary = np.zeros(clip.shape,dtype=np.bool)

    # thresh = threshold_otsu(clip[i,:,:])
    thresh = 180
    for i in range(clip.shape[0]):
        clip_binary[i,:,:] = thresh < clip[i,:,:]

    points = np.zeros((np.sum(clip_binary),3))

    count = 0

    for i in range(clip.shape[0]):
        for j in range(clip.shape[1]):
            for k in range(clip.shape[2]):
                if clip_binary[i,j,k]:
                    points[count,:] = np.array([i,j,k]) + bounding_box[:,0] + 0.0001 * np.random.rand(3)
                    count += 1

    center, rotation, radius = ef.ellipsoid_fit(points)
    # center = np.array(center) + bounding_box[:,0]

    print(center)
    print(rotation)
    print(radius)

    np.save('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_center' + human +'.npy',center)
    np.save('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_rotation' + human +'.npy',rotation)
    np.save('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_radius' + human +'.npy',radius)
    np.save('/Users/shomakitamiya/Documents/python/snake3D/data/SliceLabel/all/' + str(case) + '/bb_points' + human +'.npy',points)

    if vis:
        con, bb = tb.get_contour_pts_img(bb)
        tb.show_image_collection(tb.draw_segmentation_u(img,bb,mark_val=255,color=[255,255,255]))
        tb.show_image_collection(clip)
        tb.show_image_collection(clip_binary.astype(np.uint8) * 255)

        psize = 20

        app_points = np.array([[radius[0]*math.cos(u)*math.cos(v), radius[1]*math.cos(v)*math.sin(u),radius[2]*math.sin(v)] 
                        for u in np.linspace(0,2*math.pi,num=psize) 
                        for v in np.linspace(-math.pi/2+0.01,math.pi/2-0.01,num=psize)])
        for i in range(len(app_points)):
            app_points[i] = np.dot(app_points[i],rotation)
        app_points += center

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.plot(points[:,0],points[:,1],points[:,2],marker="o",linestyle='None')
        ax.plot(app_points[:,0],app_points[:,1],app_points[:,2],marker="o",linestyle='None')
        plt.show()

        surf = tb.make_nearest_surf(center,radius,rotation,points)
        tb.surf_render(surf)