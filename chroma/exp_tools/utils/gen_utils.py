import numpy as np
import pyvista as pv

def circle_params(points, normal):
    v01 = (points[1] - points[0]) / np.linalg.norm(points[1] - points[0])
    v12 = (points[2] - points[1]) / np.linalg.norm(points[2] - points[1])
    mid01 = points[0] + (points[1] - points[0])/2
    mid12 = points[1] + (points[2] - points[1])/2
    
    A = np.array([normal, v01, v12])
    b = np.array([np.dot(normal, points[0]), np.dot(v01, mid01), np.dot(v12, mid12)])
    center = np.linalg.solve(A, b)
    radius = np.linalg.norm(center - points[0])
    return center, radius


def convert_to_pyvista(mesh):
    return pv.make_tri_mesh(mesh.vertices, mesh.triangles)
