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
from geomdl.fitting import interpolate_curve
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score
import copy

def make_sphere_voxel(resolution,center,radius):
    x, y, z = np.indices((resolution, resolution, resolution))
    sphere = ((np.square(x - center) + np.square(y - center) + np.square(z - center))\
            < radius ** 2).astype(np.float32)
    return sphere

def make_sphere_voxel_v(resolution,center,radius):
    x, y, z = np.indices((resolution, resolution, resolution))
    sphere = ((np.square(x - center[0]) + np.square(y - center[1]) + np.square(z - center[2]))\
            < radius ** 2).astype(np.float32)
    return sphere

def make_rectangular_voxel(resolution,wdh,position):
    x, y, z = np.indices((resolution, resolution, resolution))
    rectangular = ((position[0] <= x) & (x < position[0] + wdh[0]) \
                & (position[1] <= y) & (y < position[1] + wdh[1]) \
                & (position[2] <= z) & (z < position[2] + wdh[2])).astype(np.float32)
    return rectangular

# def clip_rect(img, center, rect_size, direction):
#     rotated_img = img.copy()
#     unit_x = np.array([1,0,0])
    
#     #z軸周りの回転でdirectionをxz平面上に移動
#     direction_xy = (direction - np.array([0,0,direction[2]]))
#     direction_xy_norm = np.linalg.norm(direction_xy)
#     if direction_xy_norm != 0:
#         direction_xy = direction_xy / direction_xy_norm
#     else:
#         direction_xy = np.zeros(3)
#     rotation_angle_xy = np.sign(np.cross(direction_xy, unit_x)[2]) * math.acos(direction_xy[0])
    
#     M = cv2.getRotationMatrix2D((center[1],center[0]),np.degrees(rotation_angle_xy),1)
#     for z in range(img.shape[2]):
#         rotated_img[:,:,z] = cv2.warpAffine(rotated_img[:,:,z],M,(img.shape[1],img.shape[0]))
#     rotation_matrix_z = np.array([[math.cos(rotation_angle_xy), -math.sin(rotation_angle_xy), 0],
#                               [math.sin(rotation_angle_xy), math.cos(rotation_angle_xy), 0],
#                               [0,0,1]
#                              ])

#     rotated_z = np.dot(rotation_matrix_z,direction)
#     rotated_z = rotated_z / np.linalg.norm(rotated_z)

#     # print(rotated_z)

#     #y軸周りの回転でdirectionをx軸上に移動
#     rotation_angle_xz = np.sign(np.cross(rotated_z, unit_x)[1]) * math.acos(rotated_z[0])

#     # # 正しく回転できているかの確認用
#     # # rotatedがx軸の単位ベクトルになっていればよい
#     # rotation_matrix_y = np.array([[math.cos(rotation_angle_xz), 0,math.sin(rotation_angle_xz)],
#     #                           [0,1,0],
#     #                           [-math.sin(rotation_angle_xz),0, math.cos(rotation_angle_xz)]
#     #                          ])

#     # rotated = np.dot(rotation_matrix_y,np.dot(rotation_matrix_z,direction))
#     # rotated = rotated / np.linalg.norm(rotated)
#     # print(rotated)

#     M = cv2.getRotationMatrix2D((center[2],center[0]),-math.degrees(rotation_angle_xz),1)
#     for y in range(img.shape[1]):
#         rotated_img[:,y,:] = cv2.warpAffine(rotated_img[:,y,:],M,(img.shape[2],img.shape[0]))
#     # fig, ax = plt.subplots()
#     # ax.imshow(rotated_img[59,:,:], cmap='gray')
#     # print(rotated_img[59,46,78])

#     # plt.show()

#     # cv2.imshow('test',rotated_img[50,:,:])
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     center_index = [math.floor(center[i]) for i in range(3)]

#     clipped_img = rotated_img[center_index[0]-math.floor(rect_size[0]/2):center_index[0]+math.floor(rect_size[0]/2),
#                               center_index[1]-math.floor(rect_size[1]/2):center_index[1]+math.floor(rect_size[1]/2),
#                               center_index[2]-math.floor(rect_size[2]/2):center_index[2]+math.floor(rect_size[2]/2)]

#     # fig = plt.figure()
#     # ax = fig.gca(projection='3d')
#     # ax.voxels(rotated_img)

#     # plt.show()

#     return clipped_img

def clip_rect(img, center, rect_size, direction):
    simg_size = rect_size[0] + 10
    start = np.array([int(center[i])-int(simg_size/2) for i in range(3)])
    end = np.array([int(center[i])+int(simg_size/2) + (1 if simg_size % 2 == 1 else 0) for i in range(3)])
    
    rotated_img = img[start[0]:end[0], start[1]:end[1], start[2]:end[2]].copy()
    unit_x = np.array([1,0,0])
    
    #z軸周りの回転でdirectionをxz平面上に移動
    direction_xy = (direction - np.array([0,0,direction[2]]))
    direction_xy_norm = np.linalg.norm(direction_xy)
    if direction_xy_norm != 0:
        direction_xy = direction_xy / direction_xy_norm
    else:
        direction_xy = np.zeros(3)
    rotation_angle_xy = np.sign(np.cross(direction_xy, unit_x)[2]) * math.acos(direction_xy[0])
    
    # M = cv2.getRotationMatrix2D((simg_size/2 + center[1] - int(center[1]),simg_size/2 + center[0] - int(center[0])),np.degrees(rotation_angle_xy),1)
    M = cv2.getRotationMatrix2D((simg_size/2,simg_size/2),np.degrees(rotation_angle_xy),1)
    
    for z in range(simg_size):
        rotated_img[:,:,z] = cv2.warpAffine(rotated_img[:,:,z],M,(simg_size,simg_size),flags=cv2.INTER_NEAREST)
    rotation_matrix_z = np.array([[math.cos(rotation_angle_xy), -math.sin(rotation_angle_xy), 0],
                              [math.sin(rotation_angle_xy), math.cos(rotation_angle_xy), 0],
                              [0,0,1]
                             ])

    rotated_z = np.dot(rotation_matrix_z,direction)
    rotated_z = rotated_z / np.linalg.norm(rotated_z)

    # print(rotated_z)

    #y軸周りの回転でdirectionをx軸上に移動
    rotation_angle_xz = np.sign(np.cross(rotated_z, unit_x)[1]) * math.acos(rotated_z[0])

    # # 正しく回転できているかの確認用
    # # rotatedがx軸の単位ベクトルになっていればよい
    # rotation_matrix_y = np.array([[math.cos(rotation_angle_xz), 0,math.sin(rotation_angle_xz)],
    #                           [0,1,0],
    #                           [-math.sin(rotation_angle_xz),0, math.cos(rotation_angle_xz)]
    #                          ])

    # rotated = np.dot(rotation_matrix_y,np.dot(rotation_matrix_z,direction))
    # rotated = rotated / np.linalg.norm(rotated)
    # print(rotated)

    # M = cv2.getRotationMatrix2D((simg_size/2 + center[2] - int(center[2]),simg_size/2 + center[0] - int(center[0])),-math.degrees(rotation_angle_xz),1)
    M = cv2.getRotationMatrix2D((simg_size/2,simg_size/2),-math.degrees(rotation_angle_xz),1)    
    for y in range(simg_size):
        rotated_img[:,y,:] = cv2.warpAffine(rotated_img[:,y,:],M,(simg_size,simg_size),flags=cv2.INTER_NEAREST)
    # fig, ax = plt.subplots()
    # ax.imshow(rotated_img[59,:,:], cmap='gray')
    # print(rotated_img[59,46,78])


    # plt.show()

    # cv2.imshow('test',rotated_img[50,:,:])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    
    start  = np.array([int(simg_size/2)-int(rect_size[i]/2) for i in range(3)])
    end = np.array([int(simg_size/2)+int(rect_size[i]/2) + (1 if rect_size[i] % 2 == 1 else 0) for i in range(3)])

    clipped_img = rotated_img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.voxels(rotated_img)

    # plt.show()

    return clipped_img

def cul_normal_vector(center, neighborhood):
    nbv = [neighborhood[i]-center for i in range(4)]
    nbv_norm = [nbv[i] / np.linalg.norm(nbv[i]) for i in range(4)]
    normals = np.array([np.cross(nbv_norm[i],nbv[(i + 1) % 4]) for i in range(4)])
    normal = np.mean(normals,axis=0) 
    return normal / np.linalg.norm(normal)

def cul_separability(clipped_image):
    w,h,d = clipped_image.shape
    N = w*h*d

    variance_all = N * np.var(clipped_image)
    average_all = np.mean(clipped_image)
    variance_boundary = [(i+1)*h*d*(np.mean(clipped_image[:i+1,:,:])-average_all) ** 2
                   + (w-i-1)*h*d*(np.mean(clipped_image[i+1:,:,:])-average_all) ** 2
                    for i in range(w-1)]
    
    separability = variance_boundary/variance_all

    # print(separability)

    return (np.max(separability), np.argmax(separability)) if variance_all != 0 else (0, int(w/2))

def update_sample_point(center, neighborhood, image, rect_size, w):
    normal = cul_normal_vector(center,neighborhood)
    clipped_image = clip_rect(image,center,rect_size,normal)
    separability, boundary = cul_separability(clipped_image)



    # dif = np.abs(np.sum(np.array([-0.98295677,0.16477033,-0.08152745])) - np.sum(normal))
    # dif = 0
    # if dif < 10 ** -6:
    #     print(boundary)
    #     print(normal)
    #     print(center)
    #     print(center + (boundary + 1 - rect_size[0]/2) * normal * w)
    #     print('')
    #     show_ct_image(clipped_image.transpose(1,0,2))

    add = np.mean(clipped_image[:boundary+1,:,:]) < np.mean(clipped_image[boundary+1:,:,:])
    # add = 1

    return separability, center + (boundary + add - int(rect_size[0]/2)) * normal * w

def evalpts_uv(surf,div,boundary=0.01):
    range_uv = np.concatenate([np.array([0]),np.linspace(0+boundary,1-boundary,div),np.array([1])])
    return np.array([surf.evaluate_list([(u,v) for v in range_uv]) for u in range_uv])

def update_evalpts(evalpts,img,rect_size,div,w=1.0):
    sep_pts = np.array([update_sample_point(evalpts[u,v],
            [evalpts[u-1,v],evalpts[u,v-1],evalpts[u+1,v],evalpts[u,v+1]],
            img,rect_size,w) for u in range(1,div+1) for v in range(1,div+1)])
    return np.mean(sep_pts[:,0]), np.array([sep_pts[i,1] for i in range(div*div)])

def show_image_collection(im3d):
    viewer3d = CollectionViewer(im3d)
    viewer3d.show()

def load_test_medical_image():
    return np.load("/Users/shomakitamiya/Documents/python/snake3D/data/numpy/case" + str(123) + ".npy")

def show_ct_image(img):
    img = img - np.min(img)
    img = img / np.max(img)

    show_image_collection(img)

def make_sphere_surf(center,radius,psize=20,qsize=8):
    points = np.array([[radius*math.cos(u)*math.cos(v), radius*math.cos(v)*math.sin(u),radius*math.sin(v)] 
                    for u in np.linspace(0,2*math.pi,num=psize) 
                    for v in np.linspace(-math.pi/2+0.01,math.pi/2-0.01,num=psize)])
    points += center
    return approximate_surface(points.tolist(),psize,psize,3,3,ctrlpts_size_u=qsize,ctrlpts_size_v=qsize)

def make_ellipsoid_surf(center,radius,psize=20,qsize=8):
    points = np.array([[radius[0]*math.cos(u)*math.cos(v), radius[1]*math.cos(v)*math.sin(u),radius[2]*math.sin(v)] 
                    for u in np.linspace(0,2*math.pi,num=psize) 
                    for v in np.linspace(-math.pi/2+0.01,math.pi/2-0.01,num=psize)])
    points += center
    return approximate_surface(points.tolist(),psize,psize,3,3,ctrlpts_size_u=qsize,ctrlpts_size_v=qsize)

def make_ellipsoid_axis_surf(center,radius,rotation,psize=20,qsize=8):
    points = np.array([[radius[0]*math.cos(u)*math.cos(v), radius[1]*math.cos(v)*math.sin(u),radius[2]*math.sin(v)] 
                    for u in np.linspace(0,2*math.pi,num=psize) 
                    for v in np.linspace(-math.pi/2+0.01,math.pi/2-0.01,num=psize)])
    for i in range(len(points)):
        points[i] = np.dot(points[i],rotation)
    points += center
    return approximate_surface(points.tolist(),psize,psize,3,3,ctrlpts_size_u=qsize,ctrlpts_size_v=qsize)

def make_nearest_surf(center,radius,rotation,contour_pts,psize=20,qsize=8,vis=False,seg=None):

    points = np.array([[radius[0]*math.cos(u)*math.cos(v), radius[1]*math.cos(v)*math.sin(u),radius[2]*math.sin(v)] 
                    for u in np.linspace(0,2*math.pi,num=psize) 
                    for v in np.linspace(-math.pi/2+0.01,math.pi/2-0.01,num=psize)])
    for i in range(len(points)):
        points[i] = np.dot(points[i],rotation)
    points += center

    tree = BallTree(contour_pts)              
    _, ind = tree.query(points, k=1)
    ind = np.reshape(ind,(ind.shape[0]))

    points = contour_pts[ind,:].astype(np.float64)
    noise = 0.001
    points += np.random.rand(points.shape[0],points.shape[1]) * noise

    if vis:        
        img_mask = get_image_mask_points(seg,points)

        color_img = draw_segmentation(seg, img_mask,mark_val=255)
        show_ct_image(color_img)

    return approximate_surface(points.tolist(),psize,psize,3,3,ctrlpts_size_u=qsize,ctrlpts_size_v=qsize)


def cul_contour(img,init,rect_size,div=20,N_limit=30,dif_limit=0.001,dif_abs=True,increasing_limit=0.01,ctrlpts_increasing=False,c=0.95,ctrlpts_size=8,w=0.95):
    prev_separability = 0
    for _ in range(N_limit):
        evalpts = evalpts_uv(init,div,0.00001)
        mean_separability, updated_evalpts = update_evalpts(evalpts,img,rect_size.astype(np.int64),div,w)
  
        print(mean_separability)
        if dif_abs :
            if math.fabs(mean_separability - prev_separability) < dif_limit:
                break
        else:
            if mean_separability - prev_separability < dif_limit:
                break

        init = approximate_surface(updated_evalpts.tolist(),div,div,3,3,ctrlpts_size_u=ctrlpts_size,ctrlpts_size_v=ctrlpts_size)

        if ctrlpts_increasing:
            if math.fabs(mean_separability - prev_separability) < increasing_limit:
                ctrlpts_size = ctrlpts_size + 1
        
        print(ctrlpts_size)

        prev_separability = mean_separability

        print(rect_size.astype(np.int64))
        rect_size = np.array([rs * c if 1 <= rs * c else 1 for rs in rect_size])
        
    return init

def cul_contour_pca(img,init,rect_size,P,div=40,N_limit=24,dif_limit=0.001,increasing_limit=0.01,c=0.95,ctrlpts_size=8,w=0.95,mindim=6,maxdim=20,step=5):
    prev_separability = 0
    for i in range(N_limit):
        evalpts = evalpts_uv(init,div,0.00001)
        mean_separability, updated_evalpts = update_evalpts(evalpts,img,rect_size.astype(np.int64),div,w)

        print(mean_separability)
        if mean_separability - prev_separability < dif_limit:
            break
        

        init = approximate_surface(updated_evalpts.tolist(),div,div,3,3,ctrlpts_size_u=ctrlpts_size,ctrlpts_size_v=ctrlpts_size)
        

        if math.fabs(mean_separability - prev_separability) < increasing_limit and ctrlpts_size < maxdim:
            ctrlpts_size = ctrlpts_size + 1
        
        print(ctrlpts_size)

        prev_separability = mean_separability

        if i % step == step - 1:
            print('pca')
            init = projection_surf(P[init.ctrlpts_size_u - mindim],init)
            prev_separability = 0
        
        print(rect_size.astype(np.int64))
        rect_size = np.array([rs * c if 1 <= rs * c else 1 for rs in rect_size])

    return init

def sphere_fit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return np.array([C[0,0],C[2,0],C[1,0]]), radius

def add_margin(inside, margin):
    img_shape = tuple([s + 2*margin for s in inside.shape])

    img = np.zeros(img_shape,dtype=inside.dtype)
    img += np.min(inside)
    img[margin:img_shape[0]-margin,margin:img_shape[1]-margin,margin:img_shape[2]-margin] += inside - np.min(inside)

    return img

def surf_render(surf, delta=0.05):
    surf.delta = delta
    surf.vis = VisMPL.VisSurfWireframe()
    surf.render()

def get_image_mask_points(img, evalpts):
    indices = np.array(evalpts,dtype=np.int32)

    img_mask = np.zeros(img.shape,dtype=np.uint8)

    img_mask[indices[:,0],indices[:,1],indices[:,2]] = 255

    
    return img_mask

def get_image_mask(img, surf, delta=0.0025):
    surf.delta = delta
    indices = np.array(surf.evalpts,dtype=np.int32)

    inside = [np.logical_and(i <= indices[:,i],indices[:,i] < img.shape[i]) for i in range(3)]
    inside = np.all(inside, axis=0)

    indices = indices[inside]

    img_mask = np.zeros(img.shape,dtype=np.uint8)
    img_mask[indices[:,0],indices[:,1],indices[:,2]] = 255

    # for i in range(indices.shape[0]):
    #     if 0 <= indices[i,0] and indices[i,0] < img_mask.shape[0] and 0 <= indices[i,1] and indices[i,1] < img_mask.shape[1] and 0 <= indices[i,2] and indices[i,2] < img_mask.shape[2]:
    #         img_mask[indices[i,0],indices[i,1],indices[i,2]] = 255
    
    return img_mask

def get_image_mask_thick(img, surf, delta=0.0025):
    surf.delta = 0.0025
    indices = np.array(surf.evalpts,dtype=np.int32)

    xyz = [-1, 0, 1]

    img_mask = np.zeros(img.shape,dtype=np.uint8)

    for x in xyz:
        for y in xyz:
            for z in xyz:
                img_mask[indices[:,0]+x,indices[:,1]+y,indices[:,2]+z] = 255

    return img_mask

def draw_contour(img, surf, delta=0.0025):
    img_mask = get_image_mask(img,surf,delta)

    color_img = np.zeros((img.shape[0],img.shape[1],img.shape[2],3))
    color_img[:,:,:,0] = np.where(img_mask==255,1,img)
    color_img[:,:,:,1] += img 
    color_img[:,:,:,2] += img

    return color_img

def draw_contour_u(img, surf, delta=0.0025,color=[255,0,0]):
    img_mask = get_image_mask(img,surf,delta)

    color_img = np.zeros((img.shape[0],img.shape[1],img.shape[2],3),dtype=np.uint8)
    color_img[:,:,:,0] = np.where(img_mask==255,color[0],img)
    color_img[:,:,:,1] += np.where(img_mask==255,color[1],img)
    color_img[:,:,:,2] += np.where(img_mask==255,color[2],img)

    return color_img

def draw_contour_thick(img, surf, delta=0.0025, color=[255,0,0]):
    img_mask = get_image_mask_thick(img,surf,delta)

    color_img = np.zeros((img.shape[0],img.shape[1],img.shape[2],3),dtype=np.uint8)
    color_img[:,:,:,0] = np.where(img_mask==255,color[0],img)
    color_img[:,:,:,1] += np.where(img_mask==255,color[1],img)
    color_img[:,:,:,2] += np.where(img_mask==255,color[2],img)

    return color_img

def draw_segmentation(img, seg, mark_val=1):
    maxval = np.max(img)

    color_img = np.zeros((img.shape[0],img.shape[1],img.shape[2],3))
    color_img[:,:,:,0] += np.where(seg==mark_val,maxval/5+img,img)
    color_img[:,:,:,1] += img
    color_img[:,:,:,2] += img

    return color_img

def draw_segmentation_u(img, seg, mark_val=1,color=[255,0,0]):

    color_img = np.zeros((img.shape[0],img.shape[1],img.shape[2],3),dtype=np.uint8)
    color_img[:,:,:,0] += np.where(seg==mark_val,color[0],img)
    color_img[:,:,:,1] += np.where(seg==mark_val,color[1],img)
    color_img[:,:,:,2] += np.where(seg==mark_val,color[2],img)

    return color_img


def get_contour_pts(seg):
    contour_pts_x = []

    for i in range(seg.shape[0]):
        contour,_ = cv2.findContours(seg[i,:,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        for c in contour:
            for pts in c:
                contour_pts_x.append((i,pts[0,1],pts[0,0]))

    contour_pts_y = []

    for i in range(seg.shape[1]):
        contour,_ = cv2.findContours(seg[:,i,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        for c in contour:
            for pts in c:
                contour_pts_y.append((pts[0,1],i,pts[0,0]))
    
    contour_pts_z = []

    for i in range(seg.shape[2]):
        contour,_ = cv2.findContours(seg[:,:,i],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        for c in contour:
            for pts in c:
                contour_pts_z.append((pts[0,1],pts[0,0],i))

    cpx_set = set(contour_pts_x)
    cpy_set = set(contour_pts_y)
    cpz_set = set(contour_pts_z)

    union = cpx_set | cpy_set | cpz_set

    contour_pts = list(union)

    return np.array(contour_pts)

def get_contour_pts_img(seg):
    contour_pts = []
    img_contour = np.zeros(seg.shape,dtype=np.uint8)

    for i in range(seg.shape[0]):
        contour,_ = cv2.findContours(seg[i,:,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        for c in contour:
            for pts in c:
                contour_pts.append([i,pts[0,1],pts[0,0]])

        img_contour[i,:,:] = cv2.polylines(img_contour[i,:,:],contour,True,255)
    # show_ct_image(img_contour)

    return np.array(contour_pts), img_contour

def get_full_pts(seg):
    contour_pts = np.array(np.where(seg==1)).transpose()

    return contour_pts

def fill_contour(img_mask):
    img_contour = np.zeros(img_mask.shape,dtype=np.uint8)

    for i in range(img_mask.shape[0]):
        contour,_ = cv2.findContours(img_mask[i,:,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        img_contour[i,:,:] = cv2.fillPoly(img_contour[i,:,:],contour,255)

    for j in range(img_mask.shape[1]):
        contour,_ = cv2.findContours(img_mask[:,j,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        img_contour[:,j,:] = cv2.fillPoly(img_contour[:,j,:],contour,255)

    for k in range(img_mask.shape[2]):
        contour,_ = cv2.findContours(img_mask[:,:,k],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        img_contour[:,:,k] = cv2.fillPoly(img_contour[:,:,k],contour,255).get()

    # tb.show_image_collection(img_contour)

    for i in range(img_mask.shape[0]):
        contour,_ = cv2.findContours(img_contour[i,:,:],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        img_contour[i,:,:] = cv2.fillPoly(img_contour[i,:,:],contour,255)

    # tb.show_image_collection(img_contour)
    return img_contour

def threshold_image_seg(img, seg):
    clipped_img = img[np.where(seg == 1)]
    threshold_max = np.max(clipped_img)
    threshold_min = np.min(clipped_img)

    threshold_max -= 30
    threshold_min += 30

    threshold_img = np.where(np.logical_and(threshold_min <= img,img <= threshold_max),img,0)

    return threshold_img
    
def threshold_image_minmax(img, t_min, t_max):
    threshold_max = t_max
    threshold_min = t_min


    threshold_img = np.where(np.logical_and(threshold_min <= img,img <= threshold_max),img,0)

    return threshold_img


def cul_separability_seg(clipped_image,clipped_seg,weight):
    w,h,d = clipped_image.shape
    N = w*h*d

    variance_all = N * np.var(clipped_image)
    average_all = np.mean(clipped_image)
    variance_boundary = np.array([(i+1)*h*d*(np.mean(clipped_image[:i+1,:,:])-average_all) ** 2
                   + (w-i-1)*h*d*(np.mean(clipped_image[i+1:,:,:])-average_all) ** 2
                    for i in range(w-1)])

    variance_all_seg = N * np.var(clipped_seg)
    average_all_seg = np.mean(clipped_seg)
    variance_boundary_seg = np.array([(i+1)*h*d*(np.mean(clipped_seg[:i+1,:,:])-average_all_seg) ** 2
                   + (w-i-1)*h*d*(np.mean(clipped_seg[i+1:,:,:])-average_all_seg) ** 2
                    for i in range(w-1)])

    # print(type(weight))
    # print(type(variance_all_seg))
    # print(type(variance_all))
    # print(type(variance_boundary))
    # print(type(variance_boundary_seg))
    
    separability = weight * variance_boundary_seg / variance_all_seg if variance_all_seg != 0 else np.zeros(variance_boundary.shape)
    separability += (1 - weight) * variance_boundary / variance_all if variance_all != 0 else np.zeros(variance_boundary.shape)
    # print(separability)

    return (np.max(separability), np.argmax(separability)) if variance_all != 0 or variance_all_seg != 0 else (0, int(w/2))

def update_sample_point_seg(center, neighborhood, image, seg, rect_size, w, weight,debug=False):
    normal = cul_normal_vector(center,neighborhood)
    clipped_image = clip_rect(image,center,rect_size,normal)
    clipped_seg = clip_rect(seg,center,rect_size,normal)


    separability, boundary = cul_separability_seg(clipped_image,clipped_seg,weight)

    if debug:
        print(separability)
        print(boundary)
        print(normal)
        print(center)
        updated = center + (boundary + 1 - rect_size[0]/2) * normal * w
        print(updated)
        print('')
        print(clipped_seg.transpose(2,0,1))

        mask = np.zeros(seg.shape)
        mask[int(center[0]),int(center[1]),int(center[2])] = 1
        mask[int(updated[0]),int(updated[1]),int(updated[2])] = 1              
        
        color_img = draw_segmentation(seg, mask,mark_val=1)
        if np.linalg.norm(center - np.array([166.84496258,284.33027991,395.81938328])) < 0.000001:
            show_ct_image(clipped_seg.transpose(2,0,1))
            show_ct_image(color_img)
    
    add = np.mean(clipped_image[:boundary+1,:,:]) < np.mean(clipped_image[boundary+1:,:,:])
    # add = 1

    return separability, center + (boundary + add - int(rect_size[0]/2)) * normal * w

def update_evalpts_seg(evalpts,img,seg,rect_size,div,weight,w=1.0):
    sep_pts = np.array([update_sample_point_seg(evalpts[u,v],
            [evalpts[u-1,v],evalpts[u,v-1],evalpts[u+1,v],evalpts[u,v+1]],
            img,seg,rect_size,w,weight) for u in range(1,div+1) for v in range(1,div+1)])
    return np.mean(sep_pts[:,0]), np.array([sep_pts[i,1] for i in range(div*div)])

def cul_contour_seg(img,seg,init,rect_size,restraint=False,P=None,div=20,N_limit=30,dif_limit=0.001,dif_abs=True,increasing_limit=0.01,ctrlpts_increasing=False,c=0.95,ctrlpts_size=8,w=0.95,weight=0.5,debug=False):
    prev_separability = 0
    mindim = 6
    maxdim = 14
    challenge = True
    
    for _ in range(N_limit):
        evalpts = evalpts_uv(init,div,0.00001)
        mean_separability, updated_evalpts = update_evalpts_seg(evalpts,img,seg,rect_size.astype(np.int64),div,weight,w)

        if debug:
            # evalpts = np.reshape(evalpts[1:div+1,1:div+1,:],(div*div,3))
            # u_evalpts = np.array(updated_evalpts)

            # for i in range(0,evalpts.shape[0],10):
            #     mask = np.zeros(img.shape)
            #     mask[int(evalpts[i,0]),int(evalpts[i,1]),int(evalpts[i,2])] = 1
            #     mask[int(u_evalpts[i,0]),int(u_evalpts[i,1]),int(u_evalpts[i,2])] = 1

            #     print(evalpts[i,:])
            #     print(u_evalpts[i,:])                
                
            #     color_img = draw_segmentation(seg, mask,mark_val=1)
            #     show_ct_image(color_img)

            evalpts = np.array(updated_evalpts)

            evalpts = np.reshape(evalpts,(div*div,3))
            img_mask = get_image_mask_points(img,evalpts)

            color_img = draw_segmentation(seg, img_mask,mark_val=255)
            show_ct_image(color_img)


        print(mean_separability)
        if restraint:
            if dif_abs :
                if math.fabs(mean_separability - prev_separability) < dif_limit:
                    break
            else:
                if mean_separability - prev_separability < dif_limit:
                    if challenge:
                        print('challenge!!')
                        escape = init
                        init = projection_surf(P[init.ctrlpts_size_u - mindim],init)
                        challenge = False
                        continue
                    else:
                        init = escape
                        break
        else:
            if dif_abs :
                if math.fabs(mean_separability - prev_separability) < dif_limit:
                    break
            else:
                if mean_separability - prev_separability < dif_limit:
                    break
        
        init = approximate_surface(updated_evalpts.tolist(),div,div,3,3,ctrlpts_size_u=ctrlpts_size,ctrlpts_size_v=ctrlpts_size)

        if ctrlpts_increasing:
            if math.fabs(mean_separability - prev_separability) < increasing_limit and ctrlpts_size < maxdim:
                ctrlpts_size += 1
        
        print(ctrlpts_size)

        prev_separability = mean_separability

        print(rect_size.astype(np.int64))
        rect_size = np.array([rs * c if 1 <= rs * c else 1 for rs in rect_size])
        challenge = True


    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot(updated_evalpts[:,0],updated_evalpts[:,1],updated_evalpts[:,2],'o',markersize=2)
    # ax.plot(evalpts.reshape([(div+2)**2,3])[:,0],evalpts.reshape([(div+2)**2,3])[:,1],evalpts.reshape([(div+2)**2,3])[:,2],'x',markersize=2)
    # plt.show()

    return init

def max_cul_contour_seg(img,seg,init,rect_size,restraint=False,P=None,div=20,N_limit=30,dif_limit=0.001,dif_abs=True,increasing_limit=0.01,ctrlpts_increasing=False,c=0.95,ctrlpts_size=8,w=0.95,weight=0.5,debug=False):
    prev_separability = 0
    mindim = 6
    maxdim = 14
    challenge = True

    separabilities = []
    surfs = []
    
    for _ in range(N_limit):
        evalpts = evalpts_uv(init,div,0.00001)
        mean_separability, updated_evalpts = update_evalpts_seg(evalpts,img,seg,rect_size.astype(np.int64),div,weight,w)

        separabilities.append(mean_separability)

        print(mean_separability)
        if restraint:
            if dif_abs :
                if math.fabs(mean_separability - prev_separability) < dif_limit:
                    break
            else:
                if mean_separability - prev_separability < dif_limit:
                    if challenge:
                        print('challenge!!')
                        escape = init
                        init = projection_surf(P[init.ctrlpts_size_u - mindim],init)
                        challenge = False
                        continue
                    else:
                        init = escape
                        break
        else:
            if dif_abs :
                if math.fabs(mean_separability - prev_separability) < dif_limit:
                    break
            else:
                if mean_separability - prev_separability < dif_limit:
                    break

        
        init = approximate_surface(updated_evalpts.tolist(),div,div,3,3,ctrlpts_size_u=ctrlpts_size,ctrlpts_size_v=ctrlpts_size)
        surfs.append(init)

        if ctrlpts_increasing:
            if math.fabs(mean_separability - prev_separability) < increasing_limit and ctrlpts_size < maxdim:
                ctrlpts_size += 1
        
        print(ctrlpts_size)

        prev_separability = mean_separability

        print(rect_size.astype(np.int64))
        rect_size = np.array([rs * c if 1 <= rs * c else 1 for rs in rect_size])
        challenge = True


    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot(updated_evalpts[:,0],updated_evalpts[:,1],updated_evalpts[:,2],'o',markersize=2)
    # ax.plot(evalpts.reshape([(div+2)**2,3])[:,0],evalpts.reshape([(div+2)**2,3])[:,1],evalpts.reshape([(div+2)**2,3])[:,2],'x',markersize=2)
    # plt.show()
     
    return surfs[np.argmax(separabilities)]

def get_full_case_id(cid):
    try:
        cid = int(cid)
        case_id = "case_{:05d}".format(cid)
    except ValueError:
        case_id = cid

    return case_id


def get_case_path(cid):
    # Resolve location where data should be living
    data_path = Path('/Users/shomakitamiya/Documents/python/snake3D/data/ValidationData')
    if not data_path.exists():
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )

    # Get case_id from provided cid
    case_id = get_full_case_id(cid)

    # Make sure that case_id exists under the data_path
    case_path = data_path / case_id
    if not case_path.exists():
        raise ValueError(
            "Case could not be found \"{}\"".format(case_path.name)
        )

    return case_path


def load_volume(cid):
    case_path = get_case_path(cid)
    vol = nib.load(str(case_path / "imaging.nii.gz"))
    return vol


def load_segmentation(cid):
    case_path = get_case_path(cid)
    seg = nib.load(str(case_path / "segmentation.nii.gz"))
    return seg

def evaluate(case_id, predictions):
    # Handle case of softmax output
    if len(predictions.shape) == 4:
        predictions = np.argmax(predictions, axis=-1)

    # Check predictions for type and dimensions
    if not isinstance(predictions, (np.ndarray, nib.Nifti1Image)):
        raise ValueError("Predictions must by a numpy array or Nifti1Image")
    if isinstance(predictions, nib.Nifti1Image):
        predictions = predictions.get_data()

    if not np.issubdtype(predictions.dtype, np.integer):
        predictions = np.round(predictions)
    predictions = predictions.astype(np.uint8)

    # Load ground truth segmentation
    gt = load_segmentation(case_id).get_data()

    # Make sure shape agrees with case
    if not predictions.shape == gt.shape:
        raise ValueError(
            ("Predictions for case {} have shape {} "
            "which do not match ground truth shape of {}").format(
                case_id, predictions.shape, gt.shape
            )
        )

    try:
        # Compute tumor+kidney Dice
        tk_pd = np.greater(predictions, 0)
        tk_gt = np.greater(gt, 0)
        tk_dice = 2*np.logical_and(tk_pd, tk_gt).sum()/(
            tk_pd.sum() + tk_gt.sum()
        )
    except ZeroDivisionError:
        return 0.0, 0.0

    try:
        # Compute tumor Dice
        tu_pd = np.greater(predictions, 1)
        tu_gt = np.greater(gt, 1)
        tu_dice = 2*np.logical_and(tu_pd, tu_gt).sum()/(
            tu_pd.sum() + tu_gt.sum()
        )
    except ZeroDivisionError:
        return tk_dice, 0.0

    return tk_dice, tu_dice

def load_case(cid):
    vol = load_volume(cid)
    seg = load_segmentation(cid)
    return vol, seg

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

def cul_dice(gt, prediction):
    try:
        # Compute tumor Dice
        dice = 2*np.logical_and(gt, prediction).sum()/(
            gt.sum() + prediction.sum()
        )
    except ZeroDivisionError:
        return 0.0

    return dice 

def load_projection_matrix(method,pca_dim,mindim=6,maxdim=14):
    P = []
    for i in range(mindim,maxdim+1):
        Pi = np.load('/Users/shomakitamiya/Documents/python/snake3D/data/numpy/' + method + str(i) + '_dim' + str(pca_dim) + '_projection.npy')
        P.append(Pi)
    return P

def load_projection_matrix_path(path,mindim=6,maxdim=20):
    P = []
    for i in range(mindim,maxdim+1):
        Pi = np.load(path)
        P.append(Pi)
    return P

def window_function(img,center,width):
    wmax = center + width
    wmin = center - width

    img = np.where(wmax < img,wmax,img)
    img = np.where(img < wmin,wmin,img)

    img = (img - wmin) / (wmax - wmin) * 255
    img = img.astype(np.uint8)

    return img

def eval_contour_error(pre, gt):
    # データ読み込み
    pre_con = get_full_pts(pre)
    gt_con = get_contour_pts(gt)

    tree = BallTree(gt_con)              
    _, ind = tree.query(pre_con, k=1)
    ind = np.reshape(ind,(ind.shape[0]))

    dist_sum = 0
    for i in range(pre_con.shape[0]):
        dist_sum += np.linalg.norm(pre_con[i,:] - gt_con[ind[i],:])

    dist_ave = dist_sum / pre_con.shape[0]

    return dist_ave

def eval_contour_error_seg(pre, gt):
    # データ読み込み
    pre_con = get_contour_pts(pre)
    gt_con = get_contour_pts(gt)

    tree = BallTree(gt_con)              
    _, ind = tree.query(pre_con, k=1)
    ind = np.reshape(ind,(ind.shape[0]))

    dist_sum = 0
    for i in range(pre_con.shape[0]):
        dist_sum += np.linalg.norm(pre_con[i,:] - gt_con[ind[i],:])

    dist_ave = dist_sum / pre_con.shape[0]

    return dist_ave

def remove_left_kidney(seg):
    contour_pts = get_full_pts(seg)
    kmeans = KMeans(n_clusters=2,random_state=173).fit(contour_pts)
    pred = np.array(kmeans.labels_)
    selected_clu = int(kmeans.cluster_centers_[0,2] < kmeans.cluster_centers_[1,2])
    not_selected = contour_pts[pred != selected_clu,:]

    for i in range(not_selected.shape[0]):
        seg[not_selected[i,0],not_selected[i,1],not_selected[i,2]] = 0

    return seg

def save_image_slice(img,slices,name):
    pil_img = Image.fromarray(img[slices,:,:])
    output_path = '/Users/shomakitamiya/Documents/python/snake3D/data/exp/'
    pil_img.save(output_path + name + '.png')

def manual_init(points, radius, pn=20, qn=8):
    trace = interpolate_curve(points, 3)
    rad_curve = interp1d([p[0] for p in points], radius)

    params = np.linspace(0,1,pn).tolist()
    trace_binormal = np.array(trace.binormal(params))
    trace_tangent = np.array(trace.tangent(params))

    trace_points = []

    for tbi, tt in zip(trace_binormal, trace_tangent):
        for angle in np.linspace(0,2*np.pi,pn).tolist():
            r = R.from_rotvec(angle*tt[1])
            p = tbi[0] + rad_curve(tbi[0][0]) * r.apply(tbi[1])
            trace_points.append(p.tolist())

    return approximate_surface(trace_points,pn,pn,3,3,ctrlpts_size_u=qn,ctrlpts_size_v=qn)

def separability_membrane(img,init,rect_size,div=40,N_limit=20,dif_limit=0.001,dif_abs=True,increasing_limit=0.01,ctrlpts_increasing=True,c=0.95,ctrlpts_size=8,w=0.95,debug=True):
    margin = rect_size[0]
    img = add_margin(img, margin)

    newcp = np.array(init.ctrlpts) + margin
    init.ctrlpts = newcp.tolist()
    
    prev_separability = 0
    for _ in range(N_limit):
        evalpts = evalpts_uv(init,div,0.00001)
        mean_separability, updated_evalpts = update_evalpts(evalpts,img,rect_size.astype(np.int64),div,w)
  
        if debug:
            print(mean_separability)

        if dif_abs :
            if math.fabs(mean_separability - prev_separability) < dif_limit:
                break
        else:
            if mean_separability - prev_separability < dif_limit:
                break

        init = approximate_surface(updated_evalpts.tolist(),div,div,3,3,ctrlpts_size_u=ctrlpts_size,ctrlpts_size_v=ctrlpts_size)

        if ctrlpts_increasing:
            if math.fabs(mean_separability - prev_separability) < increasing_limit:
                ctrlpts_size = ctrlpts_size + 1
        
        if debug:
            print(ctrlpts_size)

        prev_separability = mean_separability

        if debug:
            print(rect_size.astype(np.int64))

        rect_size = np.array([rs * c if 1 <= rs * c else 1 for rs in rect_size])

    newcp = np.array(init.ctrlpts) - margin
    init.ctrlpts = newcp.tolist()
        
    return init

def separabilities_func(inside, outside):
    N = len(inside) + len(outside)
    whole = np.concatenate([inside, outside])

    variance_all = N * np.var(whole)
    average_all = np.mean(whole)
    variance_boundary = len(inside)*(np.mean(inside)-average_all) ** 2 \
                        + len(outside)*(np.mean(outside)-average_all) ** 2
                
    return variance_boundary/variance_all

def separability_via_bb(surf, img, bb):
    con_img = get_image_mask(img, surf)
    fill_img = fill_contour(con_img)
    clipped_fill = fill_img[bb[0],bb[1],bb[2]]
    
    clipped_img = img[bb[0],bb[1],bb[2]]

    inside = clipped_img[clipped_fill == 255]
    outside = clipped_img[clipped_fill != 255]
                
    return separabilities_func(inside, outside)

def eval_surf(surf, gt):
    con_img = get_image_mask(gt, surf)
    fill_img = fill_contour(con_img)
    prediction = np.where(fill_img==255,1,0)

    gt = np.ravel(gt)
    prediction = np.ravel(prediction)
                
    return f1_score(gt, prediction)

def bb_from_surface(surf, img, c = 1.1):

    res_pts = np.array(surf.evalpts)
    res_max = np.max(res_pts,axis=0)
    res_min = np.min(res_pts,axis=0)
    res_cen = (res_max + res_min) / 2

    bb_min = res_cen - c * (res_cen - res_min)
    bb_min = np.where(bb_min < 0, 0 ,bb_min).astype(np.int64)

    bb_max = res_cen + c * (res_cen - res_min)
    bb_max = np.where(img.shape <= bb_max, img.shape ,bb_max).astype(np.int64)

    bb = [slice(bb_min[0],bb_max[0]),slice(bb_min[1],bb_max[1]),slice(bb_min[2],bb_max[2])]

    return bb

def make_surfaces_by_snd(original_surf, num_of_surfs=10):
    save_surfs = []
    for s in range(num_of_surfs):
        if s == 0:
            save_surfs.append(original_surf)
        else:
            oscnp = np.array(original_surf.ctrlpts)
            # oscnp += np.random.standard_normal(oscnp.shape)
            
            moved_surf = copy.deepcopy(original_surf)
            moved_surf.set_ctrlpts(oscnp.tolist(), original_surf.ctrlpts_size_u, original_surf.ctrlpts_size_v)
            save_surfs.append(moved_surf)

    return save_surfs