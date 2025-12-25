"""
    simulation actors.
"""

from .Vehicle import Vehicle, SimpleVehicle
import numpy as np
from .Controller import Controller
import random


class Actors:
    def __init__(self, xc, yc, curv, hdg, lane_1_max=3,lane_2_max=3,lane_3_max=3):
        self.xc = xc
        self.yc = yc
        self.vx = 10
        self.vy = 0
        
        self.x_ini = [45, -6, -45]
        self.y_ini = [6, 45, -6]
        self.h_ini = [180, -90, 0]
        
        self.d_ram = 30  
        self.random_dist = 20  
        self.lane_1_max = lane_1_max
        self.lane_2_max = lane_2_max
        self.lane_3_max = lane_3_max
        
        self.ctl = Controller(xc, yc, curv, hdg)

        self.dir_dic = {1:'l', 2:'s', 3:'r'}
        
        self.lane_1_vehicles = []
        self.lane_2_vehicles = []
        self.lane_3_vehicles = []
        
        self.vehicle_id_counter = 1 
        self.add_vehicles()
        

    def add_vehicles(self):
        # Lane 1:
        if self.lane_1_max != 0:
            if len(self.lane_1_vehicles) < self.lane_1_max and len(self.lane_1_vehicles)!=0 :
                if max(veh.X for veh in self.lane_1_vehicles) < self.d_ram - np.random.randint(self.random_dist):
                    self.lane_1_vehicles.append(
                        SimpleVehicle(
                            self.vehicle_id_counter,  
                            65, 6, self.vx, self.vy, self.h_ini[0], self.xc, self.yc, 
                            dir = self.dir_dic[random.randint(1,3)]
                        )
                    )
                    self.vehicle_id_counter += 1 
            elif len(self.lane_1_vehicles)==0:
                self.lane_1_vehicles.append(
                        SimpleVehicle(
                            self.vehicle_id_counter,  
                            self.x_ini[0], self.y_ini[0], self.vx, self.vy, self.h_ini[0], self.xc, self.yc, 
                            dir = self.dir_dic[random.randint(1,3)]
                        )
                    )
                self.vehicle_id_counter += 1
        
        # Lane 2:
        if self.lane_2_max != 0:
            if len(self.lane_2_vehicles) < self.lane_2_max and len(self.lane_2_vehicles)!=0:
                if max(veh.Y for veh in self.lane_2_vehicles) < self.d_ram - np.random.randint(self.random_dist):
                    self.lane_2_vehicles.append(
                        SimpleVehicle(
                            self.vehicle_id_counter,  
                            -6, 65, self.vx, self.vy, self.h_ini[1], self.xc, self.yc, 
                            dir = self.dir_dic[random.randint(1,3)]
                        )
                    )
                    self.vehicle_id_counter += 1
            elif len(self.lane_2_vehicles)==0:
                self.lane_2_vehicles.append(
                        SimpleVehicle(
                            self.vehicle_id_counter, 
                            self.x_ini[1], self.y_ini[1], self.vx, self.vy, self.h_ini[1], self.xc, self.yc, 
                            dir = self.dir_dic[random.randint(1,3)]
                        )
                    )
                self.vehicle_id_counter += 1
        
        # Lane 3:
        if self.lane_3_max != 0:
            if len(self.lane_3_vehicles) < self.lane_3_max and len(self.lane_3_vehicles)!=0:
                if min(veh.X for veh in self.lane_3_vehicles) > -self.d_ram + np.random.randint(self.random_dist):
                    self.lane_3_vehicles.append(
                        SimpleVehicle(
                            self.vehicle_id_counter,  
                            -65, -6, self.vx, self.vy, self.h_ini[2], self.xc, self.yc, 
                            dir = self.dir_dic[random.randint(1,3)]
                        )
                    )
                    self.vehicle_id_counter += 1
            elif len(self.lane_3_vehicles)==0:
                self.lane_3_vehicles.append(
                        SimpleVehicle(
                            self.vehicle_id_counter,  
                            self.x_ini[2], self.y_ini[2], self.vx, self.vy, self.h_ini[2], self.xc, self.yc, 
                            dir = self.dir_dic[random.randint(1,3)]
                        )
                    )
                self.vehicle_id_counter += 1

    def del_vehicles(self):
        if self.lane_1_vehicles:
            self.lane_1_vehicles = [
                veh for veh in self.lane_1_vehicles
                if not (
                    (min(v.Y for v in self.lane_1_vehicles) < -70 and veh.Y == min(v.Y for v in self.lane_1_vehicles)) or
                    (min(v.X for v in self.lane_1_vehicles) < -70 and veh.X == min(v.X for v in self.lane_1_vehicles)) or
                    (max(v.Y for v in self.lane_1_vehicles) > 70 and veh.Y == max(v.Y for v in self.lane_1_vehicles))
                )
            ]
        
        if self.lane_2_vehicles:
            self.lane_2_vehicles = [
                veh for veh in self.lane_2_vehicles
                if not (
                    (min(v.X for v in self.lane_2_vehicles) < -70 and veh.X == min(v.X for v in self.lane_2_vehicles)) or
                    (min(v.Y for v in self.lane_2_vehicles) < -70 and veh.Y == min(v.Y for v in self.lane_2_vehicles)) or
                    (max(v.X for v in self.lane_2_vehicles) > 70 and veh.X == max(v.X for v in self.lane_2_vehicles))
                )
            ]
        
        if self.lane_3_vehicles:
            self.lane_3_vehicles = [
                veh for veh in self.lane_3_vehicles
                if not (
                    (max(v.Y for v in self.lane_3_vehicles) > 70 and veh.Y == max(v.Y for v in self.lane_3_vehicles)) or
                    (max(v.X for v in self.lane_3_vehicles) > 70 and veh.X == max(v.X for v in self.lane_3_vehicles)) or
                    (min(v.Y for v in self.lane_3_vehicles) < -70 and veh.Y == min(v.Y for v in self.lane_3_vehicles))
                )
            ]

    def control_vehilces(self, dt=0.02, ego_car=None):
        self.add_vehicles()
        self.del_vehicles()
        if ego_car:
            vehicles = [ego_car] + self.lane_1_vehicles + self.lane_2_vehicles + self.lane_3_vehicles
            for i, veh in enumerate(vehicles):
                if i != 0:
                    accel, steer = self.ctl.cmd(veh, vehicles)
                    veh.step(accel, steer, dt)
        else:
            vehicles = self.lane_1_vehicles + self.lane_2_vehicles + self.lane_3_vehicles
            for i, veh in enumerate(vehicles):
                accel, steer = self.ctl.cmd(veh, vehicles)
                veh.step(accel, steer, dt)
            
        
    
    def get_vehicle_data(self):
        return {
            'x_others': self.x_others,
            'y_others': self.y_others,
            'v_others': self.v_others,
            'psi_others': self.psi_others,
            'lane': self.lane,
            'num_car': self.num_car
        }
        
    
    
        