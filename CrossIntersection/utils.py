import numpy as np


def poly_axes(pts):
    axes = []
    n = len(pts)
    for i in range(n):
        p1, p2 = pts[i], pts[(i+1) % n]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])  
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            axes.append(normal / norm)
    return axes


def project_poly(pts, axis):
    
    proj = np.dot(pts, axis) 
    return proj.min(), proj.max()

def sat_collision(poly1, poly2):
    
    axes = poly_axes(poly1) + poly_axes(poly2)
    for axis in axes:
        min1, max1 = project_poly(poly1, axis)
        min2, max2 = project_poly(poly2, axis)
        if max1 < min2 or max2 < min1:    
            return False
    return True

def vehicle_poly_np(vehicle):
    return np.array(vehicle.get_world_vertices(), dtype=float)


def get_reference_road(x, y, x_r, y_r, tree, interval=20, num_points=5):
    
    _, nearest_idx = tree.query([x, y])

    indices = nearest_idx + np.arange(num_points) * interval
    indices = np.clip(indices, 0, len(x_r) - 1)

    ref_x = x_r[indices]
    ref_y = y_r[indices]

    return ref_x, ref_y
