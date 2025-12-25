import numpy as np

class SimpleVehicle:
    """自行车运动学模型"""
    def __init__(self, id, x, y, vx, vy, yaw, X_rc, Y_rc, dir=2,
                 delta_limit=np.radians(30)):
        self.id = id 
        self.id_f = None 
        self.X, self.Y, self.psi, self.v = x, y, np.deg2rad(yaw), np.sqrt(vx**2+vy**2)
        self.dpsi = 0
        self.vx = vx
        self.vy = vy
        self.L = 2.875
        self.w, self.h = 4.875, 2.1
        self.delta_limit = delta_limit
        self.d_inter = 24  # 十字路口半径
        self.LANE_WIDTH = 4
        
        self.Caf = 78268  # Cornering stiffness front [N/rad]
        self.Car = 133321  # Cornering stiffness rear [N/rad]
        self.m = 1911  # Vehicle mass [kg]
        self.lf = 1.35  # Distance from CG to front axle [m]
        self.lr = 1.525  # Distance from CG to rear axle [m]
        self.Iz = 2410
        
        self.color = (0,120,255)

        self.dir = dir
        self.dir_dic = {'l': 1, 's': 2, 'r': 3}
        
        self.start_phase = find_quadrant(x, y)

        self.X_ref = X_rc[self.start_phase-1, :, self.dir_dic[self.dir]-1]
        self.Y_ref = Y_rc[self.start_phase-1, :, self.dir_dic[self.dir]-1]
        self.traj_predict = predict_traj(0, self.X, self.Y, self.psi)
        
        self.target_phase = cal_target_phase(self.start_phase, self.dir_dic[self.dir])
        self.org = cal_turn_origin(self.start_phase, self.dir_dic[self.dir])
        self.tp = call_turn_point(self.start_phase)

    def step(self, a_cmd, delta_cmd, dt):
        self.v = max(self.v + a_cmd * dt, 0)
        Vx = self.v * np.cos(self.psi)
        Vy = self.v * np.sin(self.psi)
        self.dpsi = self.v * np.tan(delta_cmd) / self.L
        self.X += Vx * dt
        self.Y += Vy * dt
        self.psi += self.dpsi * dt
        self.vx = self.v
        self.vy = 0
        self.ax = a_cmd
        self.ay = 0
    
    def call_turn_point(self):
        if self.start_phase==1:
            return [24, 2]
        elif self.start_phase==2:
            return [-2, 24]
        elif self.start_phase==3:
            return [-24, 2]
        elif self.start_phase==4:
            return [2, -24]
        
    def cal_turn_origin(self): 
        if (self.start_phase==1 and self.dir=='r') or (self.start_phase==2 and self.dir=='l'):
            return [24, 24]
        elif (self.start_phase==2 and self.dir=='r') or (self.start_phase==3 and self.dir=='l'):
            return [-24, 24]
        elif (self.start_phase==3 and self.dir=='r') or (self.start_phase==4 and self.dir=='l'):
            return [-24, -24]
        elif (self.start_phase==4 and self.dir=='r') or (self.start_phase==1 and self.dir=='l'):
            return [24, -24]
    
    def cal_target_phase(self):
        if (self.start_phase == 1 and self.dir=='l') or \
            (self.start_phase == 2 and self.dir=='s') or \
            (self.start_phase == 3 and self.dir=='r'):
            return 3
        elif (self.start_phase == 1 and self.dir=='s') or \
            (self.start_phase == 2 and self.dir=='r') or \
            (self.start_phase == 4 and self.dir=='l'):
            return 2
        elif (self.start_phase == 2 and self.dir=='l') or \
            (self.start_phase == 3 and self.dir=='s') or \
            (self.start_phase == 4 and self.dir=='r'):
            return 4
        elif (self.start_phase == 1 and self.dir=='r') or \
            (self.start_phase == 3 and self.dir=='l') or \
            (self.start_phase == 4 and self.dir=='s'):
            return 1
        else:
            return 0

    def find_quadrant(self, x, y):
        if x >= 0 and y > 0:
            return 1  
        elif x < 0 and y > 0:
            return 2    
        elif x < 0 and y <= 0:
            return 3    
        elif x >= 0 and y < 0:
            return 4    
        else:
            return 0

    def predict_traj(self):
        num_points = 50
        trajectory = np.zeros((num_points, 2))
        x, y = self.X, self.Y
        ds = 0.5
        theta = self.psi
        for i in range(num_points):
            x += ds * np.cos(theta)
            y += ds * np.sin(theta)
            theta += ds * np.tan(self.steer) / self.L
            trajectory[i, :] = [x, y]
        return trajectory
    
    def vehicle_detect(self, vehicles):
        vehicles = [veh for veh in vehicles if veh.id != self.id] 
        possible_vehicles = []
        
        vehicle = [veh for veh in vehicles if self.start_phase == veh.start_phase]  
        possible_vehicle = self.Radar(vehicle, l=20, angle=30)
        if possible_vehicle != None:
            possible_vehicles.append(possible_vehicle)
        
        vehicle = [veh for veh in vehicles if self.target_phase == veh.target_phase]   
        possible_vehicle = self.Radar(vehicle, l=20, angle=30)
        if possible_vehicle != None:
            possible_vehicles.append(possible_vehicle)

        if self.X**2 + self.Y**2 <= self.d_inter**2:  
            if self.dir=='l':   
                if self.start_phase == 1:
                    x1_tar, y1_tar = 6, 0  # 14, 0
                    x2_tar, y2_tar = 6, 0  # 10.14359, -2
                    x3_tar, y3_tar = 0, -6  # 2, -10.14359 
                    x4_tar, y4_tar = 0, -6  # 0, -14
                    x5_tar, y5_tar = -6, -24  # -2, -24
                elif self.start_phase == 2:
                    x1_tar, y1_tar = 0, 6  # 0, 14
                    x2_tar, y2_tar = 0, 6  # 2, 10.14359
                    x3_tar, y3_tar = 6, 0   # 10.14359, 2   
                    x4_tar, y4_tar = 6, 0   # 14, 0
                    x5_tar, y5_tar = 24, -6   # 24, -2
                elif self.start_phase == 3:
                    x1_tar, y1_tar = -6, 0    # -14, 0
                    x2_tar, y2_tar = -6, 0    # -10.14359, 2  
                    x3_tar, y3_tar = 0, 6    # -2, 10.14359  
                    x4_tar, y4_tar = 0, 6    # 0, 14
                    x5_tar, y5_tar = 6, 24    # 2, 24
                elif self.start_phase == 4:
                    x1_tar, y1_tar = 0, -6  #   0, -14
                    x2_tar, y2_tar = 0, -6  #   -2, -10.14359  
                    x3_tar, y3_tar = -6, 0  #   -10.14359, -2  
                    x4_tar, y4_tar = -6, 0  #   -14, 0
                    x5_tar, y5_tar = -24, 6  #   -24, 2
                for veh in vehicles:
                    if veh.X**2+veh.Y**2<self.d_inter**2: 
                        if ((self.start_phase == 1 and veh.start_phase == 4)  \
                        or (self.start_phase == 2 and veh.start_phase == 1)  \
                        or (self.start_phase == 3 and veh.start_phase == 2)  \
                        or (self.start_phase == 4 and veh.start_phase == 3)): 
                            if veh.dir == 'l':    
                                veh_A = arc_length(self.X, self.Y, x4_tar, y4_tar, self.org[0], self.org[1])
                                veh_B = arc_length(veh.X, veh.Y, x4_tar, y4_tar, veh.org[0], veh.org[1])
                                if veh_B >=-4 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 's':  
                                veh_A = arc_length(self.X, self.Y, x1_tar, y1_tar, self.org[0], self.org[1])
                                veh_B = str_length(veh.X, veh.Y, x1_tar, y1_tar, veh.tp[0], veh.tp[1])
                                if veh_B >=-4 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                        elif ((self.start_phase == 1 and veh.start_phase == 3) \
                        or (self.start_phase == 2 and veh.start_phase == 4) \
                        or (self.start_phase == 3 and veh.start_phase == 1)\
                        or (self.start_phase == 4 and veh.start_phase == 2)):
                            if veh.dir == 'r':
                                veh_A = arc_length(self.X, self.Y, x5_tar, y5_tar, self.org[0], self.org[1])
                                veh_B = -arc_length(veh.X, veh.Y, x5_tar, y5_tar, veh.org[0], veh.org[1], R=18)
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 's':
                                veh_A = arc_length(self.X, self.Y, x3_tar, y3_tar, self.org[0], self.org[1])
                                veh_B = str_length(veh.X, veh.Y, x3_tar, y3_tar, veh.tp[0], veh.tp[1])
                                if veh_B >=-4 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                        elif ((self.start_phase == 1 and veh.start_phase == 2) \
                        or (self.start_phase == 2 and veh.start_phase == 3) \
                        or (self.start_phase == 3 and veh.start_phase == 4)\
                        or (self.start_phase == 4 and veh.start_phase == 1)): 
                            if veh.dir == 'l':
                                veh_A = arc_length(self.X, self.Y, x1_tar, y1_tar, self.org[0], self.org[1])
                                veh_B = arc_length(veh.X, veh.Y, x1_tar, y1_tar, veh.org[0], veh.org[1])
                                if veh_B >=-4 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 's':
                                veh_A = arc_length(self.X, self.Y, x5_tar, y5_tar, self.org[0], self.org[1])
                                veh_B = str_length(veh.X, veh.Y, x5_tar, y5_tar, veh.tp[0], veh.tp[1])
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                closest_vehicle = min(possible_vehicles, key=lambda veh: np.sqrt((veh.X-self.X)**2 + (veh.Y-self.Y)**2)) if possible_vehicles != [] else None
                return closest_vehicle
                
            elif self.dir=='s':  
                if self.start_phase == 1:
                    x1_tar, y1_tar = 6, 6  # 10.14359, 6  
                    x2_tar, y2_tar = 0, 6  # 6, 6  
                    x3_tar, y3_tar = 0, 6  #-2, 2 
                    x4_tar, y4_tar = -6, 6 #-10.14359, 61
                    x5_tar, y5_tar = -24, 6 #-24, 6  
                elif self.start_phase == 2:
                    x1_tar, y1_tar = -6, 6   # -2, 10.14359
                    x2_tar, y2_tar = -6, 0   # -2, 2
                    x3_tar, y3_tar = -6, 0   # -2, -2
                    x4_tar, y4_tar = -6, -6  # -2, -10.14359
                    x5_tar, y5_tar = -6, -24   # -2, -24
                elif self.start_phase == 3:
                    x1_tar, y1_tar = -6, -6   #  -10.14359, -2
                    x2_tar, y2_tar = 0, -6   #  -2, -2
                    x3_tar, y3_tar = 0, -6   #  2, -2
                    x4_tar, y4_tar = 6, -6   #  10.14359, -2
                    x5_tar, y5_tar = 24, -6   #  24, -2
                elif self.start_phase == 4:
                    x1_tar, y1_tar = 6, -6  # 2, -10.14359
                    x2_tar, y2_tar = 6, 0  # 2, -2
                    x3_tar, y3_tar = 6, 0  # 2, 2
                    x4_tar, y4_tar = 6, 6  # 2, 10.14359
                    x5_tar, y5_tar = 6, 24  # 2, 24
                for veh in vehicles:
                    if veh.X**2+veh.Y**2<self.d_inter**2: 
                        if ((self.start_phase == 1 and veh.start_phase == 4) \
                         or (self.start_phase == 2 and veh.start_phase == 1) \
                         or (self.start_phase == 3 and veh.start_phase == 2)\
                         or (self.start_phase == 4 and veh.start_phase == 3)): 
                            if veh.dir == 'l':
                                veh_A = str_length(self.X, self.Y, x5_tar, y5_tar, self.tp[0], self.tp[1])
                                veh_B = arc_length(veh.X, veh.Y, x5_tar, y5_tar, veh.org[0], veh.org[1])
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 's':
                                veh_A = str_length(self.X, self.Y, x1_tar, y1_tar, self.tp[0], self.tp[1])
                                veh_B = str_length(veh.X, veh.Y, x1_tar, y1_tar, veh.tp[0], veh.tp[1])
                                if veh_B >=-3 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                        elif ((self.start_phase == 1 and veh.start_phase == 3) \
                           or (self.start_phase == 2 and veh.start_phase == 4) \
                           or (self.start_phase == 3 and veh.start_phase == 1)\
                           or (self.start_phase == 4 and veh.start_phase == 2)):
                            if veh.dir == 'l':
                                veh_A = str_length(self.X, self.Y, x2_tar, y2_tar, self.tp[0], self.tp[1])
                                veh_B = arc_length(veh.X, veh.Y, x2_tar, y2_tar, veh.org[0], veh.org[1])
                                if veh_B >=-4 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                        elif ((self.start_phase == 1 and veh.start_phase == 2) \
                           or (self.start_phase == 2 and veh.start_phase == 3) \
                           or (self.start_phase == 3 and veh.start_phase == 4) \
                           or (self.start_phase == 4 and veh.start_phase == 1)): 
                            if veh.dir == 'l':
                                veh_A = str_length(self.X, self.Y, x2_tar, y2_tar, self.tp[0], self.tp[1])
                                veh_B = arc_length(veh.X, veh.Y, x2_tar, y2_tar, veh.org[0], veh.org[1])
                                if veh_B >=-4 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 's':
                                veh_A = str_length(self.X, self.Y, x4_tar, y4_tar, self.tp[0], self.tp[1])
                                veh_B = str_length(veh.X, veh.Y, x4_tar, y4_tar, veh.tp[0], veh.tp[1])
                                if veh_B >=-3 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 'r':
                                veh_A = str_length(self.X, self.Y, x5_tar, y5_tar, self.tp[0], self.tp[1])
                                veh_B = -arc_length(veh.X, veh.Y, x5_tar, y5_tar, veh.org[0], veh.org[1], R=18)
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                closest_vehicle = min(possible_vehicles, key=lambda veh: np.sqrt((veh.X-self.X)**2 + (veh.Y-self.Y)**2)) if possible_vehicles != [] else None
                return closest_vehicle
            
            elif self.dir=='r': 
                if self.start_phase == 1:
                    x1_tar, y1_tar = 6, 24  # 2, 24 
                elif self.start_phase == 2:
                    x1_tar, y1_tar = -24, 6 #-24, 2
                elif self.start_phase == 3:
                    x1_tar, y1_tar = -6, -24 # -2, -24
                elif self.start_phase == 4:
                    x1_tar, y1_tar = 24, -6 # 24, -2
                for veh in vehicles:
                    if veh.X**2+veh.Y**2<self.d_inter**2: 
                        if ((self.start_phase == 1 and veh.start_phase == 4) \
                        or (self.start_phase == 2 and veh.start_phase == 1) \
                        or (self.start_phase == 3 and veh.start_phase == 2)\
                        or (self.start_phase == 4 and veh.start_phase == 3)): 
                            if veh.dir == 's':
                                veh_A = -arc_length(self.X, self.Y, x1_tar, y1_tar, self.org[0], self.org[1], R=18)
                                veh_B = np.sqrt((veh.X-x1_tar)**2 + (veh.Y-y1_tar)**2)
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                        elif ((self.start_phase == 1 and veh.start_phase == 3) \
                        or (self.start_phase == 2 and veh.start_phase == 4) \
                        or (self.start_phase == 3 and veh.start_phase == 1)\
                        or (self.start_phase == 4 and veh.start_phase == 2)): 
                            if veh.dir == 'l':
                                veh_A = -arc_length(self.X, self.Y, x1_tar, y1_tar, self.org[0], self.org[1], R=18)
                                veh_B = arc_length(veh.X, veh.Y, x1_tar, y1_tar, veh.org[0], veh.org[1])
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                closest_vehicle = min(possible_vehicles, key=lambda veh: np.sqrt((veh.X-self.X)**2 + (veh.Y-self.Y)**2)) if possible_vehicles != [] else None
                return closest_vehicle
            else:
                return None
        return min(possible_vehicles, key=lambda veh: np.sqrt((veh.X-self.X)**2 + (veh.Y-self.Y)**2)) if possible_vehicles != [] else None
    
    def Radar(self, vehicles, l = 100, angle=60):
        omega1 = np.radians(angle)  
        x = np.array([veh.X for veh in vehicles])
        y = np.array([veh.Y for veh in vehicles])
        dx = x - self.X
        dy = y - self.Y
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) - self.psi
        angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi
        in_range = (distance <= l) & (np.abs(angle) <= omega1 / 2) 
        if np.any(in_range):  
            closest_idx = np.where(in_range)[0][np.argmin(distance[in_range])]
            closest_vehicle = vehicles[closest_idx]
        else:
            closest_vehicle = None
        return closest_vehicle

    def isterminal(self):
        dist = np.sqrt((self.X - self.X_ref[-1])**2 + (self.Y - self.Y_ref[-1])**2)
        return dist <= 2

    def Filter(self, current, target, tau, dt):
        return current + (target - current) * dt / tau

    def get_world_vertices(self):
        half_w, half_h = self.w / 2, self.h / 2
        local = [(-half_w,  half_h), ( half_w,  half_h),
                 ( half_w, -half_h), (-half_w, -half_h)]
        cos_y, sin_y = np.cos(self.psi), np.sin(self.psi)
        return [(self.X + dx * cos_y - dy * sin_y,
                 self.Y + dx * sin_y + dy * cos_y) for dx, dy in local]


class Vehicle:
    def __init__(self, id, x, y, vx, vy, psi, X_rc, Y_rc, dir=2, color=(0,120,255)):
        self.id = id
        self.id_f = 0  
        self.X = x
        self.Y = y
        self.vx = vx
        self.vy = vy
        self.v = np.sqrt(vx**2 + vy**2)
        self.dpsi = 0
        self.psi = np.deg2rad(psi)
        self.ax = 0
        self.ay = 0
        self.dir = dir
        self.dir_dic = {'l': 1, 's': 2, 'r': 3}
        self.L = 2.875
        self.accel = 0
        self.steer = 0
        self.isFilter = False
        self.LANE_WIDTH = 4
        self.Caf = 78268  # Cornering stiffness front [N/rad]
        self.Car = 133321  # Cornering stiffness rear [N/rad]
        self.m = 1911  # Vehicle mass [kg]
        self.lf = 1.35  # Distance from CG to front axle [m]
        self.lr = 1.525  # Distance from CG to rear axle [m]
        self.Iz = 2410
        self.w = 4.8
        self.h = 2.1
        self.d_inter = 24  
        self.num_lane = 3
        self.X_rc = X_rc
        self.Y_rc = Y_rc
        self.ispredict = True
        self.t_p = 0.1 
        
        self.color = color
        self.start_phase = find_quadrant(x, y)

        self.X_ref = X_rc[self.start_phase-1, :, self.dir_dic[self.dir]-1]
        self.Y_ref = Y_rc[self.start_phase-1, :, self.dir_dic[self.dir]-1]
        self.traj_predict = predict_traj(0, self.X, self.Y, self.psi)
        
        self.target_phase = cal_target_phase(self.start_phase, self.dir_dic[self.dir])
        self.org = cal_turn_origin(self.start_phase, self.dir_dic[self.dir])
        self.tp = call_turn_point(self.start_phase)

    def step(self, accel, steer, dt=0.1):
        if self.isFilter:
            self.accel = self.Filter(self.accel, accel, 0.5, dt)
            self.steer = self.Filter(self.steer, steer, 0.2, dt)
        else:
            self.accel = accel
            self.steer = steer

        if self.vx > 5:
            dot_X = self.vx * np.cos(self.psi) - self.vy * np.sin(self.psi)
            dot_Y = self.vx * np.sin(self.psi) + self.vy * np.cos(self.psi)
            dot_vx = self.accel + self.vy * self.dpsi
            dot_vy = (
                -2 * (self.Car + self.Caf) * self.vy / (self.m * self.vx)
                - (self.vx + 2 * (self.Caf * self.lf - self.Car * self.lr) / (self.m * self.vx)) * self.dpsi
                + 2 * self.Caf * steer / self.m
            )
            ddpsi = (
                -2 * (self.Caf * self.lf - self.Car * self.lr) * self.vy / (self.Iz * self.vx)
                - 2 * (self.Caf * self.lf**2 + self.Car * self.lr**2) * self.dpsi / (self.Iz * self.vx)
                + 2 * self.Caf * self.lf * steer / self.Iz
            )

            self.dpsi += ddpsi * dt
            self.psi += self.dpsi * dt
            self.vx += dot_vx * dt
            self.vy += dot_vy * dt
            self.X += dot_X * dt
            self.Y += dot_Y * dt
            self.v = np.sqrt(self.vx**2 + self.vy**2)
            self.ax = dot_vx
            self.ay = dot_vy
        else:
            self.v = max(self.v + self.accel * dt, 0)
            Vx = self.v * np.cos(self.psi)
            Vy = self.v * np.sin(self.psi)
            self.dpsi = self.v * np.tan(self.steer) / self.L
            self.X += Vx * dt
            self.Y += Vy * dt
            self.psi += self.dpsi * dt
            self.vx = self.v
            self.vy = 0
            self.ax = self.accel
            self.ay = 0

        if self.ispredict:
            self.traj_predict = predict_traj(steer, self.X, self.Y, self.psi)
    
    def vehicle_detect(self, vehicles):
        vehicles = [veh for veh in vehicles if veh.id != self.id] 
        possible_vehicles = []
        
        vehicle = [veh for veh in vehicles if self.start_phase == veh.start_phase] 
        possible_vehicle = self.Radar(vehicle, l=20, angle=30)
        if possible_vehicle != None:
            possible_vehicles.append(possible_vehicle)
        
        vehicle = [veh for veh in vehicles if self.target_phase == veh.target_phase] 
        possible_vehicle = self.Radar(vehicle, l=20, angle=30)
        if possible_vehicle != None:
            possible_vehicles.append(possible_vehicle)

        if self.X**2 + self.Y**2 <= self.d_inter**2:   
            if self.dir=='l':   
                if self.start_phase == 1:
                    x1_tar, y1_tar = 6, 0  # 14, 0
                    x2_tar, y2_tar = 6, 0  # 10.14359, -2
                    x3_tar, y3_tar = 0, -6  # 2, -10.14359 
                    x4_tar, y4_tar = 0, -6  # 0, -14
                    x5_tar, y5_tar = -6, -24  # -2, -24
                elif self.start_phase == 2:
                    x1_tar, y1_tar = 0, 6  # 0, 14
                    x2_tar, y2_tar = 0, 6  # 2, 10.14359
                    x3_tar, y3_tar = 6, 0   # 10.14359, 2   
                    x4_tar, y4_tar = 6, 0   # 14, 0
                    x5_tar, y5_tar = 24, -6   # 24, -2
                elif self.start_phase == 3:
                    x1_tar, y1_tar = -6, 0    # -14, 0
                    x2_tar, y2_tar = -6, 0    # -10.14359, 2  
                    x3_tar, y3_tar = 0, 6    # -2, 10.14359  
                    x4_tar, y4_tar = 0, 6    # 0, 14
                    x5_tar, y5_tar = 6, 24    # 2, 24
                elif self.start_phase == 4:
                    x1_tar, y1_tar = 0, -6  #   0, -14
                    x2_tar, y2_tar = 0, -6  #   -2, -10.14359  
                    x3_tar, y3_tar = -6, 0  #   -10.14359, -2  
                    x4_tar, y4_tar = -6, 0  #   -14, 0
                    x5_tar, y5_tar = -24, 6  #   -24, 2
                for veh in vehicles:
                    if veh.X**2+veh.Y**2<self.d_inter**2: 
                        if ((self.start_phase == 1 and veh.start_phase == 4) \
                         or (self.start_phase == 2 and veh.start_phase == 1) \
                         or (self.start_phase == 3 and veh.start_phase == 2)\
                         or (self.start_phase == 4 and veh.start_phase == 3)):
                            if veh.dir == 'l':
                                veh_A = str_length(self.X, self.Y, x5_tar, y5_tar, self.tp[0], self.tp[1])
                                veh_B = arc_length(veh.X, veh.Y, x5_tar, y5_tar, veh.org[0], veh.org[1])
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 's':
                                veh_A = str_length(self.X, self.Y, x2_tar, y2_tar, self.tp[0], self.tp[1])
                                veh_B = str_length(veh.X, veh.Y, x2_tar, y2_tar, veh.tp[0], veh.tp[1])
                                if veh_B >=-3 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                        elif ((self.start_phase == 1 and veh.start_phase == 3) \
                           or (self.start_phase == 2 and veh.start_phase == 4) \
                           or (self.start_phase == 3 and veh.start_phase == 1)\
                           or (self.start_phase == 4 and veh.start_phase == 2)):
                            if veh.dir == 'l':
                                veh_A = str_length(self.X, self.Y, x4_tar, y4_tar, self.tp[0], self.tp[1])
                                veh_B = arc_length(veh.X, veh.Y, x4_tar, y4_tar, veh.org[0], veh.org[1])
                                if veh_B >=-4 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                        elif ((self.start_phase == 1 and veh.start_phase == 2) \
                           or (self.start_phase == 2 and veh.start_phase == 3) \
                           or (self.start_phase == 3 and veh.start_phase == 4) \
                           or (self.start_phase == 4 and veh.start_phase == 1)):
                            if veh.dir == 'l':
                                veh_A = str_length(self.X, self.Y, x1_tar, y1_tar, self.tp[0], self.tp[1])
                                veh_B = arc_length(veh.X, veh.Y, x1_tar, y1_tar, veh.org[0], veh.org[1])
                                if veh_B >=-4 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 's':
                                veh_A = str_length(self.X, self.Y, x3_tar, y3_tar, self.tp[0], self.tp[1])
                                veh_B = str_length(veh.X, veh.Y, x3_tar, y3_tar, veh.tp[0], veh.tp[1])
                                if veh_B >=-3 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 'r':
                                veh_A = str_length(self.X, self.Y, x5_tar, y5_tar, self.tp[0], self.tp[1])
                                veh_B = -arc_length(veh.X, veh.Y, x5_tar, y5_tar, veh.org[0], veh.org[1], R=18)
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                closest_vehicle = min(possible_vehicles, key=lambda veh: np.sqrt((veh.X-self.X)**2 + (veh.Y-self.Y)**2)) if possible_vehicles != [] else None
                return closest_vehicle
                
            elif self.dir=='s': 
                if self.start_phase == 1:
                    x1_tar, y1_tar = 6, 6  # 10.14
                    x2_tar, y2_tar = 0, 6  # 6, 6 
                    x3_tar, y3_tar = 0, 6  #-2, 2 
                    x4_tar, y4_tar = -6, 6 #-10.14
                    x5_tar, y5_tar = -24, 6 #-24, 
                elif self.start_phase == 2:
                    x1_tar, y1_tar = -6, 6   # -2, 10.14359
                    x2_tar, y2_tar = -6, 0   # -2, 2
                    x3_tar, y3_tar = -6, 0   # -2, -2
                    x4_tar, y4_tar = -6, -6  # -2, -10.14359
                    x5_tar, y5_tar = -6, -24   # -2, -24
                elif self.start_phase == 3:
                    x1_tar, y1_tar = -6, -6   #  -10.14359, -2
                    x2_tar, y2_tar = 0, -6   #  -2, -2
                    x3_tar, y3_tar = 0, -6   #  2, -2
                    x4_tar, y4_tar = 6, -6   #  10.14359, -2
                    x5_tar, y5_tar = 24, -6   #  24, -2
                elif self.start_phase == 4:
                    x1_tar, y1_tar = 6, -6  # 2, -10.14359
                    x2_tar, y2_tar = 6, 0  # 2, -2
                    x3_tar, y3_tar = 6, 0  # 2, 2
                    x4_tar, y4_tar = 6, 6  # 2, 10.14359
                    x5_tar, y5_tar = 6, 24  # 2, 24
                for veh in vehicles:
                    if veh.X**2+veh.Y**2<self.d_inter**2: 
                        if ((self.start_phase == 1 and veh.start_phase == 4) \
                         or (self.start_phase == 2 and veh.start_phase == 1) \
                         or (self.start_phase == 3 and veh.start_phase == 2)\
                         or (self.start_phase == 4 and veh.start_phase == 3)): 
                            if veh.dir == 'l':
                                veh_A = str_length(self.X, self.Y, x5_tar, y5_tar, self.tp[0], self.tp[1])
                                veh_B = arc_length(veh.X, veh.Y, x5_tar, y5_tar, veh.org[0], veh.org[1])
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 's':
                                veh_A = str_length(self.X, self.Y, x1_tar, y1_tar, self.tp[0], self.tp[1])
                                veh_B = str_length(veh.X, veh.Y, x1_tar, y1_tar, veh.tp[0], veh.tp[1])
                                if veh_B >=-3 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                        elif ((self.start_phase == 1 and veh.start_phase == 3) \
                           or (self.start_phase == 2 and veh.start_phase == 4) \
                           or (self.start_phase == 3 and veh.start_phase == 1)\
                           or (self.start_phase == 4 and veh.start_phase == 2)):
                            if veh.dir == 'l':
                                veh_A = str_length(self.X, self.Y, x2_tar, y2_tar, self.tp[0], self.tp[1])
                                veh_B = arc_length(veh.X, veh.Y, x2_tar, y2_tar, veh.org[0], veh.org[1])
                                if veh_B >=-4 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                        elif ((self.start_phase == 1 and veh.start_phase == 2) \
                           or (self.start_phase == 2 and veh.start_phase == 3) \
                           or (self.start_phase == 3 and veh.start_phase == 4) \
                           or (self.start_phase == 4 and veh.start_phase == 1)): 
                            if veh.dir == 'l':
                                veh_A = str_length(self.X, self.Y, x2_tar, y2_tar, self.tp[0], self.tp[1])
                                veh_B = arc_length(veh.X, veh.Y, x2_tar, y2_tar, veh.org[0], veh.org[1])
                                if veh_B >=-4 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 's':
                                veh_A = str_length(self.X, self.Y, x4_tar, y4_tar, self.tp[0], self.tp[1])
                                veh_B = str_length(veh.X, veh.Y, x4_tar, y4_tar, veh.tp[0], veh.tp[1])
                                if veh_B >=-3 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                            elif veh.dir == 'r':
                                veh_A = str_length(self.X, self.Y, x5_tar, y5_tar, self.tp[0], self.tp[1])
                                veh_B = -arc_length(veh.X, veh.Y, x5_tar, y5_tar, veh.org[0], veh.org[1], R=18)
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                closest_vehicle = min(possible_vehicles, key=lambda veh: np.sqrt((veh.X-self.X)**2 + (veh.Y-self.Y)**2)) if possible_vehicles != [] else None
                return closest_vehicle
            
            elif self.dir=='r':  
                if self.start_phase == 1:
                    x1_tar, y1_tar = 6, 24  # 2, 24  
                elif self.start_phase == 2:
                    x1_tar, y1_tar = -24, 6 #-24, 2
                elif self.start_phase == 3:
                    x1_tar, y1_tar = -6, -24 # -2, -24
                elif self.start_phase == 4:
                    x1_tar, y1_tar = 24, -6 # 24, -2
                for veh in vehicles:
                    if veh.X**2+veh.Y**2<self.d_inter**2: 
                        if ((self.start_phase == 1 and veh.start_phase == 4) \
                        or (self.start_phase == 2 and veh.start_phase == 1) \
                        or (self.start_phase == 3 and veh.start_phase == 2)\
                        or (self.start_phase == 4 and veh.start_phase == 3)): 
                            if veh.dir == 's':
                                veh_A = -arc_length(self.X, self.Y, x1_tar, y1_tar, self.org[0], self.org[1], R=18)
                                veh_B = np.sqrt((veh.X-x1_tar)**2 + (veh.Y-y1_tar)**2)
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                        elif ((self.start_phase == 1 and veh.start_phase == 3) \
                        or (self.start_phase == 2 and veh.start_phase == 4) \
                        or (self.start_phase == 3 and veh.start_phase == 1)\
                        or (self.start_phase == 4 and veh.start_phase == 2)): 
                            if veh.dir == 'l':
                                veh_A = -arc_length(self.X, self.Y, x1_tar, y1_tar, self.org[0], self.org[1], R=18)
                                veh_B = arc_length(veh.X, veh.Y, x1_tar, y1_tar, veh.org[0], veh.org[1])
                                if veh_B >=-5 and veh_A>=veh_B:
                                    possible_vehicles.append(veh)
                closest_vehicle = min(possible_vehicles, key=lambda veh: np.sqrt((veh.X-self.X)**2 + (veh.Y-self.Y)**2)) if possible_vehicles != [] else None
                return closest_vehicle
            else:
                return None
        return min(possible_vehicles, key=lambda veh: np.sqrt((veh.X-self.X)**2 + (veh.Y-self.Y)**2)) if possible_vehicles != [] else None
    
    def Radar(self, vehicles, l = 100, angle=60, islane=True):
        omega1 = np.radians(angle) 
        x = np.array([veh.X for veh in vehicles])
        y = np.array([veh.Y for veh in vehicles])
        dx = x - self.X
        dy = y - self.Y
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) - self.psi
        angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi
        in_range = (distance <= l) & (np.abs(angle) <= omega1 / 2) 
        if np.any(in_range):  
            closest_idx = np.where(in_range)[0][np.argmin(distance[in_range])] 
            closest_vehicle = vehicles[closest_idx]
        else:
            closest_vehicle = None
        return closest_vehicle

    def isterminal(self):
        dist = np.sqrt((self.X - self.X_ref[-1])**2 + (self.Y - self.Y_ref[-1])**2)
        return dist <= 2

    def Filter(self, current, target, tau, dt):
        return current + (target - current) * dt / tau
    
    def initialize(self, x, y, vx, vy, psi):
        self.id_f = 0 
        self.X = x
        self.Y = y
        self.vx = vx
        self.vy = vy
        self.dpsi = 0
        self.v = np.sqrt(vx**2 + vy**2)
        self.psi = np.deg2rad(psi)

        self.start_phase = find_quadrant(x, y)

        self.X_ref = self.X_rc[self.start_phase, :, self.dir_dic[self.dir]]
        self.Y_ref = self.Y_rc[self.start_phase, :, self.dir_dic[self.dir]]
        self.traj_predict = predict_traj(0, self.X, self.Y, self.psi)
    
    def get_world_vertices(self):
        half_w, half_h = self.w / 2, self.h / 2
        local = [(-half_w,  half_h), ( half_w,  half_h),
                 ( half_w, -half_h), (-half_w, -half_h)]
        cos_y, sin_y = np.cos(self.psi), np.sin(self.psi)
        return [(self.X + dx * cos_y - dy * sin_y,
                 self.Y + dx * sin_y + dy * cos_y) for dx, dy in local]

def arc_length(x, y, a, b, x0=24, y0=-24, R=30):
    theta_A = np.arctan2(b - y0, a - x0)
    theta_P = np.arctan2(y - y0, x - x0)

    delta_theta = theta_A - theta_P  
    delta_theta %= 2 * np.pi  

    if delta_theta > np.pi:
        delta_theta -= 2 * np.pi  

    arc = R * delta_theta  
    return arc

def str_length(x, y, a, b, x0, y0):
    if np.sqrt((x-x0)**2 + (y-y0)**2) > np.sqrt((a-x0)**2 + (b-y0)**2):
        return -np.sqrt((x-a)**2+(y-b)**2)
    else:
        return np.sqrt((x-a)**2+(y-b)**2)
    
    
def call_turn_point(start_phase):
    if start_phase==1:
        return [24, 2]
    elif start_phase==2:
        return [-2, 24]
    elif start_phase==3:
        return [-24, 2]
    elif start_phase==4:
        return [2, -24]
    
def cal_turn_origin(start_phase, dir):
    if (start_phase==1 and dir==3) or (start_phase==2 and dir==1):
        return [24, 24]
    elif (start_phase==2 and dir == 3) or (start_phase==3 and dir==1):
        return [-24, 24]
    elif (start_phase==3 and dir == 3) or (start_phase==4 and dir==1):
        return [-24, -24]
    elif (start_phase==4 and dir == 3) or (start_phase==1 and dir==1):
        return [24, -24]

def cal_target_phase(start_phase, dir):
    if (start_phase == 1 and dir == 1) or \
        (start_phase == 2 and dir == 2) or \
        (start_phase == 3 and dir == 3):
        return 3
    elif (start_phase == 1 and dir == 2) or \
            (start_phase == 2 and dir == 3) or \
            (start_phase == 4 and dir == 1):
        return 2
    elif (start_phase == 2 and dir == 1) or \
            (start_phase == 3 and dir == 2) or \
            (start_phase == 4 and dir == 3):
        return 4
    elif (start_phase == 1 and dir == 3) or \
            (start_phase == 3 and dir == 1) or \
            (start_phase == 4 and dir == 2):
        return 1
    else:
        return 0

def find_quadrant(x, y):
    if x >= 0 and y > 0:
        return 1    
    elif x < 0 and y > 0:
        return 2    
    elif x < 0 and y <= 0:
        return 3    
    elif x >= 0 and y < 0:
        return 4    
    else:
        return 0

def predict_traj(steer, x, y, theta, num_points=50, ds=0.5, L=2.875):
    trajectory = np.zeros((num_points, 2))
    for i in range(num_points):
        x += ds * np.cos(theta)
        y += ds * np.sin(theta)
        theta += ds * np.tan(steer) / L
        trajectory[i, :] = [x, y]
    return trajectory