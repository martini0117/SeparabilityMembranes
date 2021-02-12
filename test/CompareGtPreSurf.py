import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import math

import matplotlib
import pickle
import os

import geomdl.visualization.VisMPL as VisMPL
import matplotlib.pyplot as plt
import numpy as np
from geomdl import BSpline
from geomdl.fitting import approximate_surface
from geomdl.knotvector import generate
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from skimage.segmentation import flood_fill
import nibabel as nib
import random
from sklearn.decomposition import PCA

import Toolbox as tb

def reshaping(array,size_u,size_v):
    array = np.reshape(array,(size_u*size_v,3))

    p1_list = []
    for p in array:
        p1_list.append(list(p))
    
    return p1_list

def projection_surf(P,surf):
    pts = np.array(surf.ctrlpts)
    
    before_pro = np.reshape(pts,(pts.shape[0]*pts.shape[1]))
    after_pro = np.dot(P,before_pro)

    after_pro_f = reshaping(after_pro,surf.ctrlpts_size_u,surf.ctrlpts_size_v)

    surf.set_ctrlpts(after_pro_f,surf.ctrlpts_size_u,surf.ctrlpts_size_v)

    return surf

data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/ValidationData/'

# fo = open('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/' + method + str(cs)+'.txt', 'w')
# sys.stdout = fo

cases = [1,18,22,23,29,50,52,59,71,73,75,76,78,80,82,97,98,106,109,114,127,128,135,144,166,167,170,171,173,175,179,185,196,203,209]

dice_sum = 0

weight = 7
method = 'pca_loop'
pca_dim = 20
min_dim = 6
max_dim = 14


for i,case in enumerate(cases):
    case_str = tb.get_full_case_id(case)
    print(case_str)

    nii0=nib.load(data_path + case_str + '/segmentation.nii.gz')
    gt_escape = nii0.get_data().astype(np.uint8)
    gt_escape = tb.add_margin(gt_escape,30)

    loop = 5
    for i in range(loop):
        print('loop' + str(i) + ':')

        gt = gt_escape

        surf_path = '/Users/shomakitamiya/Documents/python/snake3D/data/' + method + '_dim' + str(pca_dim) + '_W' + str(weight) + 'Data/' + case_str + '/loop' + str(i)

        if os.path.exists(surf_path):
            f = open(surf_path,'rb')
            surf = pickle.load(f)
            f.close
        else:
            continue

        # tb.surf_render(surf)

        img_mask = tb.get_image_mask(gt,surf)
        prediction = tb.fill_contour(img_mask)

        gt = np.where(gt==2,1,0)
        prediction = np.where(prediction==255,1,0)

        # tb.show_ct_image(gt)
        # tb.show_ct_image(prediction)
        
        ans = tb.cul_dice(gt,prediction)
        print(ans)
    # dice_sum += ans

# print(dice_sum/cases.shape[0])