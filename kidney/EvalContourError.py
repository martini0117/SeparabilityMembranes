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

#計算済みのデータの精度を評価

# cases = np.array(range(150,160))
cases = np.array(range(150,190))
# cases = np.array([185])

data_path = '/Users/shomakitamiya/Documents/python/snake3D/data/kits19/data/'
output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/KidneyContour/NonPCA/'

error_sum = 0
init_error_sum = 0

window_center = 0
window_width = 300

human = '_h'

count = 0

for i,case in enumerate(cases):
    if case == 165:
        continue
    
    case_str = tb.get_full_case_id(case)
    print(case_str)

    nii0=nib.load(data_path + case_str + '/imaging.nii.gz')
    img = nii0.get_data()
    img = tb.window_function(img,window_center,window_width)
    
    nii1 = nib.Nifti1Image(img,affine=None)
    nib.save(nii1,data_path + case_str + '/imaging_uint8.nii.gz')

    nii0=nib.load(data_path + case_str + '/segmentation.nii.gz')
    gt = nii0.get_data().astype(np.uint8)
    gt = np.where(gt==2,0,gt)
    gt = tb.remove_left_kidney(gt)

    nii1 = nib.Nifti1Image(gt,affine=None)
    nib.save(nii1,data_path + case_str + '/segmentation_left.nii.gz')
    # tb.show_image_collection(255*gt)
    # gt[:,:,:int(gt.shape[2]/2)] = 0

    nii0=nib.load(data_path + case_str + '/contour_img' + human + '.nii.gz')
    init = nii0.get_data().astype(np.uint8)
    
    init_fill = tb.fill_contour(init)
    # tb.show_image_collection(init_fill)
    nii1 = nib.Nifti1Image(init_fill,affine=None)
    nib.save(nii1,data_path + case_str + '/contour_img' + human + '_fill.nii.gz')

    init = init / 255

    nii0=nib.load(data_path + case_str + '/contour_img_nearest' + human + '.nii.gz')
    init_n = nii0.get_data().astype(np.uint8)
    
    init_fill = tb.fill_contour(init_n)
    # tb.show_image_collection(init_fill)
    nii1 = nib.Nifti1Image(init_fill,affine=None)
    nib.save(nii1,data_path + case_str + '/contour_img_nearest' + human + '_fill.nii.gz')

    nii0=nib.load(output_path + case_str + '/prediction_contour' + human + '.nii.gz')
    prediction = nii0.get_data().astype(np.uint8)

    prediction_fill = tb.fill_contour(prediction)
    nii1 = nib.Nifti1Image(prediction_fill,affine=None)
    nib.save(nii1,data_path + case_str + '/prediction_contour_fill' + human + '_fill.nii.gz')

    prediction = prediction / 255

    # tb.show_ct_image(gt)
    # tb.show_ct_image(prediction)

    ans_init = tb.eval_contour_error(init,gt)
    print('init: ' + str(ans_init))
    
    ans = tb.eval_contour_error(prediction,gt)
    print('prediction: ' + str(ans))
    
    if ans < ans_init:
        init_error_sum += ans_init
        error_sum += ans
        count += 1 

print(init_error_sum/cases.shape[0])
print(error_sum/cases.shape[0])
print(count)