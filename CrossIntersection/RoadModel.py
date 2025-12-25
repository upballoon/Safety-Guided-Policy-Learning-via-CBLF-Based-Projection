import numpy as np

def circle_point(R, theta, d_theta, x, y):
    theta_rad = np.deg2rad(theta)
    delta_theta_rad = np.deg2rad(d_theta)
    _angle = theta_rad + delta_theta_rad

    xc = x - R * np.cos(theta_rad)
    yc = y - R * np.sin(theta_rad)

    t = np.linspace(theta_rad, _angle, 1000)
    x_points = xc + R * np.cos(t)
    y_points = yc + R * np.sin(t)

    return x_points, y_points

def rotation_point(x, y, theta):
    r_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = np.dot(r_matrix, np.array([x, y]))
    return rotated[0, :], rotated[1, :]

def calculate_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3 / 2)
    return curvature


def road_model(num_lane=3):
    """_summary_

    Args:
        xe, ye is road egde
        xc, yc is road refercne
        
    Returns:
        _type_: _description_
    """
    road_length = 50
    road_width = 4
    road_radius = 20

    x_l0 = [road_width, road_width]
    y_l0 = [road_width + road_radius + road_length, road_width + road_radius]
    x_c1, y_c1 = circle_point(road_radius, 180, 90, road_width, road_width + road_radius)
    x_l1 = [road_width + road_radius, road_width + road_radius + road_length]
    y_l1 = [road_width, road_width]

    x_e0p1 = np.concatenate(([0], x_l0, x_c1, x_l1))
    y_e0p1 = np.concatenate(([road_width + road_radius + road_length], y_l0, y_c1, y_l1))

    route1_x1 = np.linspace(road_width + road_radius + road_length, -road_width - road_radius - road_length, 3000)
    route1_y1 = np.full_like(route1_x1, road_width * 0.5)

    route_x2c, route_y2c = circle_point(road_radius + road_width * 0.5, 270, -90, road_width + road_radius, road_width * 0.5)
    route2_x1 = np.concatenate([
        np.linspace(road_width + road_radius + road_length, road_width + road_radius, 1000),
        route_x2c,
        np.linspace(road_width * 0.5, road_width * 0.5, 1000)])
    route2_y1 = np.concatenate([
        np.full(1000, road_width * 0.5),
        route_y2c,
        np.linspace(road_width + road_radius, road_width + road_radius + road_length, 1000)])

    route_x3c, route_y3c = circle_point(road_radius + road_width * 1.5, 90, 90, road_width + road_radius, road_width * 0.5)
    route3_x1 = np.concatenate([
        np.linspace(road_width + road_radius + road_length, road_width + road_radius, 1000),
        route_x3c,
        np.linspace(-road_width * 0.5, -road_width * 0.5, 1000)])
    route3_y1 = np.concatenate([
        np.full(1000, road_width * 0.5),
        route_y3c,
        np.linspace(-road_width - road_radius, -road_width - road_radius - road_length, 1000)])

    x_e0p2, y_e0p2 = rotation_point(x_e0p1, y_e0p1, np.pi / 2)
    x_e0p3, y_e0p3 = rotation_point(x_e0p2, y_e0p2, np.pi / 2)
    x_e0p4, y_e0p4 = rotation_point(x_e0p3, y_e0p3, np.pi / 2)

    route1_x2, route1_y2 = rotation_point(route1_x1, route1_y1, np.pi / 2)
    route1_x3, route1_y3 = rotation_point(route1_x2, route1_y2, np.pi / 2)
    route1_x4, route1_y4 = rotation_point(route1_x3, route1_y3, np.pi / 2)

    route2_x2, route2_y2 = rotation_point(route2_x1, route2_y1, np.pi / 2)
    route2_x3, route2_y3 = rotation_point(route2_x2, route2_y2, np.pi / 2)
    route2_x4, route2_y4 = rotation_point(route2_x3, route2_y3, np.pi / 2)

    route3_x2, route3_y2 = rotation_point(route3_x1, route3_y1, np.pi / 2)
    route3_x3, route3_y3 = rotation_point(route3_x2, route3_y2, np.pi / 2)
    route3_x4, route3_y4 = rotation_point(route3_x3, route3_y3, np.pi / 2)

    xe0 = np.concatenate([x_e0p1, x_e0p4, x_e0p3, x_e0p2])
    ye0 = np.concatenate([y_e0p1, y_e0p4, y_e0p3, y_e0p2])
    
    xc = np.zeros((4,3000,3))
    yc = np.zeros((4,3000,3))
    xc[:,:,0] = [route3_x1, route3_x2, route3_x3, route3_x4]
    yc[:,:,0] = [route3_y1, route3_y2, route3_y3, route3_y4]
    xc[:,:,1] = [route1_x1, route1_x2, route1_x3, route1_x4]
    yc[:,:,1] = [route1_y1, route1_y2, route1_y3, route1_y4]
    xc[:,:,2] = [route2_x1, route2_x2, route2_x3, route2_x4]
    yc[:,:,2] = [route2_y1, route2_y2, route2_y3, route2_y4]

    xe = xe0
    ye = ye0
    curvature = np.zeros((4,3000,3))
    heading_direction = np.zeros((4,3000,3))
    for i in range(num_lane):
        for j in range(4):
            curvature[j, :, i] = calculate_curvature(xc[j, :, i], yc[j, :, i])
            dx = np.gradient(xc[j, :, i])
            dy = np.gradient(yc[j, :, i])
            norm_factor = np.sqrt(dx**2 + dy**2)
            nx = -dy/ norm_factor
            ny = dx / norm_factor
            heading_direction[j, :, i] = np.arctan2(ny, nx)-np.pi/2
            
    centerline_segment_x = np.linspace(road_width + road_radius, road_width + road_radius + road_length, 1000)
    centerline_segment_y = np.zeros_like(centerline_segment_x)

    c_x_east,  c_y_east  = centerline_segment_x, centerline_segment_y
    c_x_south, c_y_south = rotation_point(centerline_segment_x, centerline_segment_y, np.pi / 2)
    c_x_west,  c_y_west  = rotation_point(centerline_segment_x, centerline_segment_y, np.pi)
    c_x_north, c_y_north = rotation_point(centerline_segment_x, centerline_segment_y, 3 * np.pi / 2)

    center_lines_x = [c_x_east, c_x_south, c_x_west, c_x_north]
    center_lines_y = [c_y_east, c_y_south, c_y_west, c_y_north]

    return xe, ye, xc, yc, curvature, heading_direction, center_lines_x, center_lines_y


