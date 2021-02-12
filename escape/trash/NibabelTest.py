import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import numpy as np
import matplotlib.pyplot as plt # 描画用
import nibabel as nib # NiBabelの導入の確認　エラーが出なければ成功
import Toolbox as tb


# nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Task02_Heart/labelsTr/la_003.nii.gz')
# nii0 = nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Task09_Spleen/labelsTr/spleen_2.nii.gz')
# img= nii0.get_data().astype(np.float32)
# img= nii0.get_data().astype(np.float32).transpose(2,0,1)
# img = tb.load_test_medical_image()

# nii0 = nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Task05_Prostate/imagesTr/prostate_00.nii.gz')
# img= nii0.get_data().astype(np.float32).transpose(2,0,1,3)
# img = img[:,:,:,0]

# nii0 = nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Task05_Prostate/labelsTr/prostate_00.nii.gz')
# img= nii0.get_data().astype(np.float32).transpose(2,0,1)

# nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Task09_Spleen/imagesTr/spleen_2.nii.gz')
# img = nii0.get_data().astype(np.float32).transpose(2,0,1)

nii0=nib.load('/Users/shomakitamiya/Documents/python/snake3D/data/Data/case_00031/tumor.nii.gz')
img = nii0.get_data()

print(img.dtype)

tb.show_ct_image(img)

# img = tb.load_test_medical_image()

# center = np.array([140,310,330])
# radius = np.array([30,30,30])

# surf = tb.make_ellipsoid_surf(center,radius)

# surf.delta = 0.0025
# surface_points = np.array(surf.evalpts)

# img_mask = np.zeros(img.shape,dtype=np.uint8)

# indices = surface_points.astype(np.int32)
# img_mask[indices[:,0],indices[:,1],indices[:,2]] = 255

# img -= np.min(img)
# img /= np.max(img)

# color_img = np.zeros((img.shape[0],img.shape[1],img.shape[2],3))
# color_img[:,:,:,0] = np.where(img_mask==255,1,img)
# color_img[:,:,:,1] = np.where(img_mask==255,0,img)
# color_img[:,:,:,2] = np.where(img_mask==255,0,img)

# tb.show_image_collection(color_img)



