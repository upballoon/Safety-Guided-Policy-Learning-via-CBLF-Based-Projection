import numpy as np
import scipy.io as scio
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

K_PATH = os.path.join(CURRENT_DIR, "LQR_K.mat")

class Controller():
    def __init__(self, xc, yc, curv, hdg):
        self.K = scio.loadmat(K_PATH)  
        self.curvature = curv
        self.heading = hdg
        self.x_rc = xc
        self.y_rc = yc
    
    def IDM(self, ego, front, v_target = 10):
        if ego.X**2+ego.Y**2<=ego.d_inter**2:
            s0 = min(max(ego.v*3, 10), 20) 
            if ego.X**2+ego.Y**2<=(ego.d_inter+10)**2:
                v_target = 8
            else:
                v_target = 6
        else:
            s0 = 8
            v_target = max(v_target, 0.1)
        T = 1.6 
        a_max = 3 
        brakes = 10
        delta = 4 
        if front:  
            delta_v = ego.v - front.v*np.cos(ego.psi-front.psi)
            s_star = s0 + max(0, ego.v * T + (ego.v * delta_v) / (2 * np.sqrt(a_max * brakes)))
            acc = a_max * (1 - (ego.v / v_target) ** delta - (s_star / np.linalg.norm([front.X-ego.X, front.Y-ego.Y])) ** 2)
        else:
            acc = a_max * (1 - (ego.v / v_target) ** delta)
        acc = np.clip(acc, -brakes, a_max)
        return acc

    def lqr_control(self, car, best_lane):
        best_lane -= 1
        start_phase = car.start_phase - 1
        xr = self.x_rc[start_phase, :, best_lane]
        yr = self.y_rc[start_phase, :, best_lane]
        thetar = self.heading[start_phase, :, best_lane]
        kappar = self.curvature[start_phase, :, best_lane]
        d_all = (car.X - xr) ** 2 + (car.Y - yr) ** 2 
        smin = np.argmin(d_all)
        dmin = smin
        tor = np.array([np.cos(thetar[dmin]), np.sin(thetar[dmin])])
        nor = np.array([-np.sin(thetar[dmin]), np.cos(thetar[dmin])])
        d_err = np.array([car.X - xr[dmin], car.Y - yr[dmin]])
        ed = np.dot(nor, d_err)
        es = np.dot(tor, d_err)
        projection_point_thetar = thetar[dmin] + kappar[dmin] * es
        ded = car.vy * np.cos(car.psi - projection_point_thetar) + car.vx * np.sin(car.psi - projection_point_thetar)
        epsi = np.sin(car.psi - projection_point_thetar)
        s_dot = car.vx * np.cos(car.psi - projection_point_thetar) - car.vy * np.sin(car.psi - projection_point_thetar)
        s_dot /= (1 - kappar[dmin] * ed)
        ephi_dot = car.dpsi - kappar[dmin] * s_dot
        kr = kappar[dmin]
        err = np.array([ed, ded, epsi, ephi_dot])
        Vx = np.arange(0.01, 40.01, 0.01)
        idx = (np.abs(Vx - car.vx)).argmin()
        k = self.K['K'][:, idx].T
        forword_angle = kr * (car.lf + car.lr - car.lr * k[2] - (car.m * car.vx ** 2 / (car.lf + car.lr)) *
                            ((car.lr / car.Caf) + (car.lf / car.Car) * k[2] - (car.lf / car.Car)))
        angle = -np.dot(k, err) + forword_angle
        return angle

    
    def cmd(self, ego, vehicles):
        front_vehicle = ego.vehicle_detect(vehicles)
        if front_vehicle:
            ego.id_f = front_vehicle.id
        else:
            ego.id_f = 0
        accel = self.IDM(ego, front_vehicle)
        best_lane = ego.dir_dic[ego.dir] 
        steer = self.lqr_control(ego, best_lane)
        steer = np.clip(steer, -0.3, 0.3)
        return accel, steer
    









