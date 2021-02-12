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

method = 'Nearest'

data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/KidneyContour/Init/' + method + '/'

cs = int(sys.argv[1])

fo = open(output_path + str(cs) +'_evaluation.txt', 'w')
sys.stdout = fo

init = np.load(output_path + str(cs) + '_init.npy')
cases = np.load(output_path + str(cs) + '_case.npy')
knotvector_u = np.load(output_path + str(cs) + '_knotvector_u.npy')
knotvector_v = np.load(output_path + str(cs) + '_knotvector_v.npy')

print(init.shape)

error_sum = 0

for i,case in enumerate(cases):
    case_str = tb.get_full_case_id(case)
    print(case_str)

    nii0=nib.load(data_path + case_str + '/segmentation.nii.gz')
    gt = nii0.get_data().astype(np.uint8)
    gt = np.where(gt==2,0,gt)
    gt = tb.add_margin(gt,30)

    surf = BSpline.Surface()

    surf.degree_u = 3
    surf.degree_v = 3

    ctrpts = reshaping(init[i,:],cs,cs)
    surf.set_ctrlpts(ctrpts,cs,cs)

    surf.knotvector_u = knotvector_u[i,:]
    surf.knotvector_v = knotvector_v[i,:]

    prediction = tb.get_image_mask(gt,surf)
    
    prediction = np.where(prediction==255,1,0)

    # tb.show_ct_image(gt)
    # tb.show_ct_image(prediction)

    ans = tb.eval_contour_error(prediction,gt)
    print(ans)
    error_sum += ans

print(error_sum/cases.shape[0])