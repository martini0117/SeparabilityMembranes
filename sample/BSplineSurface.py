from geomdl import BSpline
import geomdl.visualization.VisMPL as VisMPL
from geomdl.knotvector import generate

# Create a BSpline surface instance (Bezier surface)
surf = BSpline.Surface()

# Set degrees
surf.degree_u = 2
surf.degree_v = 2

# Set control points
control_points = [[0, 0, 0], [1, 0, 0],[2,0,0],
                 [0, 1, 1], [1, 1, 1],[2,1,1],
                 [0, 2, 0], [1, 2, 0], [2, 2, 0]]

# control_points = [[0, 0, 0], [0, 1, 1],[0, 2, 0],[0, 0, 0], [0, 1, 1],
#                   [1, 0, 0], [1, 1, 1],[1, 2, 0],[1, 0, 0], [1, 1, 1],
#                   [2, 0, 0], [2, 1, 1],[2, 2, 0],[2, 0, 0], [2, 1, 1],
#                   [0, 0, 0], [0, 1, 1],[0, 2, 0],[0, 0, 0], [0, 1, 1],
#                   [1, 0, 0], [1, 1, 1],[1, 2, 0],[1, 0, 0], [1, 1, 1]
#                   ]

# control_points = [[0, 0, 0], [0, 1, 1],[0,2,0],[0, 0, 0], [0, 1, 1],
#                   [1, 0, 0], [1, 1, 1],[1,2,0],[1, 0, 0], [1, 1, 1],
#                   [2, 0, 0], [2, 1, 1], [2, 2, 0],[2, 0, 0], [2, 1, 1]
#                   ]

# control_points = [[0, 0, 0], [0, 1, 1],[0, 2, 0],
#                   [1, 0, 0], [1, 1, 1],[1, 2, 0],
#                   [2, 0, 0], [2, 1, 1],[2, 2, 0],
#                   [0, 0, 0], [0, 1, 1],[0, 2, 0],
#                   [1, 0, 0], [1, 1, 1],[1, 2, 0]
#                   ]

surf.set_ctrlpts(control_points, 3, 3)

# Set knot vectors
#surf.knotvector_u = [0, 0, 0, 0, 1, 1, 1, 1]
#surf.knotvector_v = [0, 0, 0, 1, 1, 1]
surf.knotvector_u = generate(surf.degree_u,surf.ctrlpts_size_u,clamped=True)
surf.knotvector_v = generate(surf.degree_v,surf.ctrlpts_size_v,clamped=False)
print(surf.knotvector_u)

surf.evaluate(  start_u = surf.knotvector_u[surf.degree_u],
                 stop_u = surf.knotvector_u[surf.ctrlpts_size_u],
                 start_v = surf.knotvector_v[surf.degree_v],
                 stop_v = surf.knotvector_v[surf.ctrlpts_size_v])

# Set evaluation delta (control the number of surface points)
surf.delta = 0.05

# Get surface points (the surface will be automatically evaluated)
surface_points = surf.evalpts



#surf.vis = VisMPL.VisSurface()
surf.vis = VisMPL.VisSurfWireframe()
surf.render()