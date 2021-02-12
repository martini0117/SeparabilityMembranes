import sys; sys.path.append('/Users/shomakitamiya/Documents/python/snake3D/src/Toolbox')
import math

import geomdl.visualization.VisMPL as VisMPL
import matplotlib.pyplot as plt
import numpy as np
from geomdl import BSpline
from geomdl.fitting import approximate_surface
from geomdl.knotvector import generate
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import Toolbox as tb

# 球のパラメータ
res = 300
center = res/2
radius_true = 40

# ノイズを乗せる
sphere = tb.make_sphere_voxel(res,res/2,radius_true)
sphere += tb.make_sphere_voxel(res,res/2+20,radius_true-20)
sphere += 0.6*np.random.rand(res,res,res)
sphere = np.where(sphere >= 1,1,sphere)
tb.show_ct_image(sphere)


# 初期輪郭
radius_init = 50

points = np.array([[radius_init*math.cos(u)*math.cos(v), radius_init*math.cos(v)*math.sin(u),radius_init*math.sin(v)] 
                    for u in np.linspace(0,2*math.pi,num=20) 
                    for v in np.linspace(-math.pi/2+0.01,math.pi/2-0.01,num=20)])

points += center + 0.5

# 制御点補完
surf = approximate_surface(points.tolist(),20,20,3,3,ctrlpts_size_u=8,ctrlpts_size_v=8)

img = tb.draw_contour(sphere,surf)
tb.show_ct_image(img)

div = 100
evalpts = tb.evalpts_uv(surf,div,0)

ave = []
for evalp in evalpts:
    for e in evalp:
        ave.append(np.linalg.norm(np.array(e)-np.array([center,center,center]))) 
        # print(np.array(e)-np.array([center,center,center]))
        # print(np.linalg.norm(np.array(e)-np.array([center,center,center])))

# print(np.mean(ave))
# print(np.var(ave))
# print(np.max(evalpts))
# print(np.min(evalpts))


rect_size = np.array([30,6,6])

surf = tb.cul_contour(sphere,surf,rect_size,div=20,N_limit=10,c=0.95,ctrlpts_size=8,w=1,dif_limit=-0.01,dif_abs=False,ctrlpts_increasing=True,increasing_limit=0.01)
# tb.surf_render(surf)
print('evalpts')
# for evalp in evalpts:
#     print(np.linalg.norm(np.array(evalp)-np.array([center,center,center])))

div = 100
evalpts = tb.evalpts_uv(surf,div,0)

ave = []
for evalp in evalpts:
    for e in evalp:
        ave.append(np.linalg.norm(np.array(e)-np.array([center,center,center]))) 
        # print(np.array(e)-np.array([center,center,center]))
        # print(np.linalg.norm(np.array(e)-np.array([center,center,center])))

# print(np.mean(ave))
# print(np.var(ave))
# print(np.max(ave))
# print(np.min(ave))

# print(np.max(evalpts))
# print(np.min(evalpts))


img = tb.draw_contour(sphere,surf)
tb.show_ct_image(img)