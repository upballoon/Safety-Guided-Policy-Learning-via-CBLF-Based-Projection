from RoadModel import road_model
import numpy as np
from GUI import GUI_Visualizer
from Vehicle import Vehicle
from Controller import Controller
from Actor import Actors
from utils import vehicle_poly_np, sat_collision


is_render = True

def main():
    xe, ye, xc, yc, curv, hdg, ctr_x, ctr_y = road_model()
    if is_render:
        gui = GUI_Visualizer(xe, ye, xc, yc, ctr_x, ctr_y)
    ctl = Controller(xc, yc, curv, hdg)
    lane_1_max, lane_2_max, lane_3_max = 3, 3, 3
    
    actor = Actors(xc, yc, curv, hdg, lane_1_max, lane_2_max, lane_3_max)
    ego_car = Vehicle(0, 2, -50, 0, 0, 90, xc, yc, 1)
    
    color_blue = (0,120,255)
    color_red = (255,0,0)
    color_org = (255,180,0)
    ego_car.color = color_org
    
    step = 0
    running = True
    while running:
        dt = 0.02
        if lane_1_max+lane_2_max+lane_3_max != 0:
            actor.control_vehilces(dt, ego_car)
        
        accel, steer = 0, 0 # ctl.cmd(0, ego_car, vehicles)
        ego_car.step(accel, steer, dt)
        
        vehicles = actor.lane_1_vehicles + actor.lane_2_vehicles + actor.lane_3_vehicles
        if is_render:
            for veh in actor.lane_1_vehicles + actor.lane_2_vehicles + actor.lane_3_vehicles:
                veh.color = color_blue
            ego_car.color = color_org

            for i in range(len(vehicles)):
                if sat_collision(vehicle_poly_np(vehicles[i]), vehicle_poly_np(ego_car)):
                    ego_car.color = color_red  
                    vehicles[i].color = color_red 
                
            vehicles += [ego_car]
            gui.render(vehicles, accel, steer)
            running = gui.check_quit_event()
        else:
            print(f"Step {step}\t(X, Y)=({ego_car.X:.2f}, {ego_car.Y})\tv={ego_car.v:.2f}\tpsi={ego_car.psi:.2f}")
        
        step += 1
        
        # TODO: isdone条件
        if ego_car.color==color_red:
            running = False
            print("Simulation ended due to collision.")
            
    if is_render:
        gui.close()

if __name__ == "__main__":
    main()