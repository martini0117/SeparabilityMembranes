from geomdl import BSpline
from geomdl.knotvector import generate
import geomdl.visualization.VisMPL as VisMPL

# Create a 3-dimensional B-spline Curve
curve = BSpline.Curve()

# Set degree
curve.degree = 3

# Set control points
curve.ctrlpts = [[1,2], [1,0], [0,-3], [3,2]]
curve.ctrlpts.extend([curve.ctrlpts[i] for i in range(curve.degree)])
print(curve.ctrlpts)

# Set knot vector
#curve.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
curve.knotvector = generate(curve.degree,len(curve.ctrlpts),clamped=False)

# Set evaluation delta (controls the number of curve points)
curve.delta = 0.01

# Get curve points (the curve will be automatically evaluated)
curve_points = curve.evalpts

curve.vis = VisMPL.VisCurve2D()
curve.render()