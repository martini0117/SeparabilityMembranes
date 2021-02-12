from geomdl import BSpline
from geomdl.knotvector import generate
from geomdl.fitting import approximate_curve
import geomdl.visualization.VisMPL as VisMPL
import math
import numpy as np
import matplotlib.pyplot as plt

points = np.array([[math.cos(i),math.sin(i)] for i in np.linspace(0,2*math.pi,num=100)])

# print(points)
# plt.plot(points[:,0],points[:,1])
# plt.show()

curve = approximate_curve(points.tolist(),3,ctrlpts_size=6)

curve.delta = 0.01
curve.vis = VisMPL.VisCurve2D()
curve.render()