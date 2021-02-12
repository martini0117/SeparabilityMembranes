from geomdl import BSpline
from geomdl.knotvector import generate
from geomdl.fitting import approximate_surface
import geomdl.visualization.VisMPL as VisMPL
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# points = np.array([[u, v,math.sin(v)+math.sin(u)] 
#                     for u in np.linspace(0,2*math.pi,num=30) 
#                     for v in np.linspace(0,2*math.pi,num=30)])

points = np.array([[math.cos(u)*math.cos(v), math.cos(v)*math.sin(u),math.sin(v)] 
                    for u in np.linspace(0,2*math.pi,num=30) 
                    for v in np.linspace(-math.pi/2,math.pi/2,num=30)])

# print(points)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot(points[:,0],points[:,1],points[:,2],'o',markersize=2)
# plt.show()

surf = approximate_surface(points.tolist(),30,30,3,3,ctrlpts_size_u=10,ctrlpts_size_v=10)

surf.delta = 0.05

surf.vis = VisMPL.VisSurfWireframe()
# surf.render()

div = 100
evalpts = np.array([surf.evaluate_list([(u,v) for v in np.linspace(0,1,div)]) for u in np.linspace(0,1,div)])
print(evalpts.shape)


fig = plt.figure()
ax = Axes3D(fig)

# all_points = np.reshape(evalpts,(div*div,3))
# ax.plot(all_points[:,0],all_points[:,1],all_points[:,2],'o',markersize=2)

# points33 = np.reshape(evalpts[10:13,10:13,:],(9,3))
# ax.plot(points33[:,0],points33[:,1],points33[:,2],'o',markersize=2)

# plt.show()

def cul_normal_vector(center, neighborhood):
    nbv = [neighborhood[i]-center for i in range(4)]
    nbv_norm = [nbv[i] / np.linalg.norm(nbv[i]) for i in range(4)]
    normals = np.array([np.cross(nbv_norm[i],nbv[(i + 1) % 4]) for i in range(4)])
    normal = np.mean(normals,axis=0) 
    return normal / np.linalg.norm(normal)
    # return normal


u = 80
v = 10

center = evalpts[u,v,:]
neighborhood = np.array([evalpts[u-1,v,:],evalpts[u,v-1,:],evalpts[u+1,v,:],evalpts[u,v+1,:]])

# center = np.array([0,0,1])
# neighborhood = np.array([[-1,0,0.2],[0,-1,0.1],[1,0,-0.4],[0,1,-0.3]])

normal = cul_normal_vector(center, neighborhood)
print(normal)
# normals = np.array([center, normal + center,nc[0] + center,nc[1] + center,nc[2] + center,nc[3] + center])

# print(np.dot(normals[2],neighborhood[1]-center))
# print(np.dot(normals[2],neighborhood[2]-center))

nbc = neighborhood[3]-center
nbc = nbc / np.linalg.norm(nbc)

nbv = [neighborhood[i]-center for i in range(4)]
nbv_norm = [nbv[i] / np.linalg.norm(nbv[i]) for i in range(4)]
normals = np.array([np.cross(nbv_norm[i],nbv[(i + 1) % 4]) for i in range(4)])

print(np.dot(normals[3],nbc))

ax.plot(neighborhood[:,0],neighborhood[:,1],neighborhood[:,2],'o',markersize=2)
ax.plot(normals[:,0],normals[:,1],normals[:,2],'o',markersize=2)

plt.show()
