import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .RoadModel3 import road_model
from .GUI import GUI_Visualizer
from .Vehicle import Vehicle
from .Controller import Controller
from .Actor import Actors
from scipy.spatial import KDTree
from .utils import vehicle_poly_np, sat_collision, get_reference_road

class CrossEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}
    
    @classmethod
    def default_config(self) -> dict:
        
        return {
            "frequence": 10,
            "dt": 50,  
            "duration": 40, 
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "lane_1_max": 3,
            "lane_2_max": 3,
            "lane_3_max": 3,
            "dir": 'l',       # 1-Left, 2-Stright, 3-Right
            "v_ref":6.0,
            "level": 0,
            "normalize_reward": True,
            "offroad_terminal": True,
            "safe_distance": 5.0,    
            "collision_cost": 10.0,  
            "distance_cost_mode": "binary",  
            "num_lane": 3, 
            "max_obs_vehicles": 5  
        }
        
    def __init__(self, config=None, render_mode="rgb_array"):
        super(CrossEnv, self).__init__()
        
        self.config = self.default_config()
        if config:
            self.config.update(config)
        
        self.render_mode = render_mode
        
        self.dir = self.config["dir"]
        self.dir_dic = {'l': 1, 's': 2, 'r': 3}
        self.color_blue = (0, 120, 255)
        self.color_red = (255, 0, 0)
        self.color_org = (255, 180, 0)
        self.lane_1_max, self.lane_2_max, self.lane_3_max = self.config["lane_1_max"], self.config["lane_2_max"], self.config["lane_3_max"]
        
        self.ctl_frq = 1/self.config["frequence"]
        self.dt = 1/self.config["dt"]
        self.duration = self.config["duration"]

        self.safe_distance = self.config.get("safe_distance", 5.0)
        self.collision_cost = self.config.get("collision_cost", 10.0)
        self.distance_cost_mode = self.config.get("distance_cost_mode", "binary")
        self.level = self.config.get("level", 0)
        self.num_lane = self.config.get("num_lane", 1)

        self.max_obs_vehicles = self.config.get("max_obs_vehicles", 5)

        self.preview_interval = 20   
        self.num_preview = 5
        
        # Action space: acceleration and steering
        self.action_space = spaces.Box(
            low=np.array([-3.0, -0.5]), high=np.array([3.0, 0.5]), dtype=np.float64
        )
        self.obs_dim = 4 + 2 * self.num_preview + self.other_feat_dim * min(self.level * 3, self.max_obs_vehicles)
        high = np.ones(self.obs_dim, dtype=np.float64) * 100.0  # 粗略上界，可按实际调整
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)


        self._setup_env()
        
    def _setup_env(self) -> None:
        xe, ye, self.xc, self.yc, self.curv, self.hdg, ctr_x, ctr_y, x_w, y_w = road_model(num_lane=self.num_lane)

        if self.render_mode == "human":
            self.gui = GUI_Visualizer(xe, ye, self.xc, self.yc, ctr_x, ctr_y, x_w, y_w)
        
        self.ctl = Controller(self.xc, self.yc, self.curv, self.hdg)
        self.actor = Actors(self.xc, self.yc, self.curv, self.hdg, self.lane_1_max, self.lane_2_max, self.lane_3_max)

        self.ego_car = Vehicle(0, 6, -50, 0, 0, 90, self.xc, self.yc, self.dir, self.color_org) # id, x, y, vx, vy, heading, xref, yref, target, color
        
        self.x_r = np.asarray(self.xc[self.ego_car.start_phase-1, :, self.dir_dic[self.ego_car.dir]-1])
        self.y_r = np.asarray(self.yc[self.ego_car.start_phase-1, :, self.dir_dic[self.ego_car.dir]-1])
        self.hdg_r = np.asarray(self.hdg[self.ego_car.start_phase-1, :, self.dir_dic[self.ego_car.dir]-1])
        self.curv_r = np.asarray(self.curv[self.ego_car.start_phase-1, :, self.dir_dic[self.ego_car.dir]-1])
        # 构建 KD-Tree 查找最近轨迹点
        trajectory_points = np.stack([self.x_r, self.y_r], axis=1)
        self.tree = KDTree(trajectory_points)
        
        self.terminated = False 
        self.truncated = False  
        self.runout = False
        self.collision = False
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_env()
        self.step_count = 0
        obs = self._get_obs()         
        self.prev_s_norm = self.s_norm

        return obs, {}

    def step(self, action, with_cost=None):
        if self.terminated:
            obs = self._get_obs()
            reward = 0.0
            self.truncated = self._get_truncated()
            info = {}
            if with_cost:
                cost = self._get_distance_cost()
                return obs, reward, cost, self.terminated, self.truncated, info
            else:
                return obs, reward, self.terminated, self.truncated, info

        self.accel, self.steer = action[0], action[1]
        step = int(self.ctl_frq/self.dt) 
        for _ in range(step):  
            self.actor.control_vehilces(self.dt, self.ego_car)
            self.ego_car.step(self.accel, self.steer, self.dt)
        
        self.step_count += step
        self.terminated = self._get_done()
        
        self.truncated = self._get_truncated()
        obs = self._get_obs()
        reward = self._get_reward()
        info = {"X": self.ego_car.X, 
                "Y": self.ego_car.Y,
                "vx": self.ego_car.vx, 
                "vy": self.ego_car.vy,
                "heading": self.ego_car.psi,
                "x_ref": self.ref_x,
                "y_ref": self.ref_y,
                "hdg_ref": self.psi_ref,
                "curv_ref": self.curv_ref,
                }
        if with_cost:
            cost = self._get_distance_cost()
            return obs, reward, cost, self.terminated, self.truncated, info
        else:
            return obs, reward, self.terminated, self.truncated, info

    def _get_obs(self):
        ego_x   = self.ego_car.X
        ego_y   = self.ego_car.Y
        ego_vx  = self.ego_car.vx
        ego_vy  = self.ego_car.vy
        ego_psi = self.ego_car.psi

        cos_psi = np.cos(ego_psi)
        sin_psi = np.sin(ego_psi)

        _, nearest_idx = self.tree.query([ego_x, ego_y])
        nearest_idx = int(nearest_idx)
        N = len(self.x_r)

        idxs = nearest_idx + np.arange(self.num_preview) * self.preview_interval
        idxs = np.clip(idxs, 0, N - 1)

        self.ref_x = self.x_r[idxs]
        self.ref_y = self.y_r[idxs]

        self.psi_ref = self.hdg_r[nearest_idx]
        self.curv_ref = self.curv_r[nearest_idx]
        self.s_norm = nearest_idx / (N - 1)

        self.e_psi = _wrap_to_pi(ego_psi - self.psi_ref)

        ref_x0 = self.x_r[nearest_idx]
        ref_y0 = self.y_r[nearest_idx]
        ex = ego_x - ref_x0
        ey = ego_y - ref_y0
        n_ref_x = -np.sin(self.psi_ref)
        n_ref_y =  np.cos(self.psi_ref)
        self.e_y = ex * n_ref_x + ey * n_ref_y

        dx_ref = self.ref_x - ego_x
        dy_ref = self.ref_y - ego_y
        x_rel_ref =  cos_psi * dx_ref + sin_psi * dy_ref
        y_rel_ref = -sin_psi * dx_ref + cos_psi * dy_ref
        obs_preview = np.stack([x_rel_ref, y_rel_ref], axis=-1).flatten()

        obs_core = np.array([self.s_norm, self.e_y, self.e_psi, ego_vx], dtype=np.float64)

        vehicles = (
            self.actor.lane_1_vehicles
            + self.actor.lane_2_vehicles
            + self.actor.lane_3_vehicles
        )

        max_k = min(self.level*3, self.max_obs_vehicles)
        other_feats = np.zeros((max_k, self.other_feat_dim), dtype=np.float64)

        temp_list = []
        for veh in vehicles:
            dx = veh.X - ego_x
            dy = veh.Y - ego_y
            dvx = veh.vx - ego_vx
            dvy = veh.vy - ego_vy

            x_rel =  cos_psi * dx + sin_psi * dy
            y_rel = -sin_psi * dx + cos_psi * dy

            vx_rel =  cos_psi * dvx + sin_psi * dvy
            vy_rel = -sin_psi * dvx + cos_psi * dvy

            temp_list.append((veh.id, x_rel, y_rel, vx_rel, vy_rel))

        for i, (id, x_rel, y_rel, vx_rel, vy_rel) in enumerate(temp_list[:max_k]):
            other_feats[i, :] = [id, x_rel, y_rel, vx_rel, vy_rel]

        obs_others = other_feats.flatten()   # 长度 = max_k * 4
        
        obs = np.concatenate([obs_core, obs_preview, obs_others/10], dtype=np.float64)
        return obs
    
    def _get_reward(self) -> None:
        R_crash = -200.0 if self.collision else 0.0
        R_out   = -200.0  if self.runout   else 0.0
        
        k_y, k_psi = 1.0, 0.5
        R_follow = -k_y * self.e_y**2 - k_psi * self.e_psi**2
        R_follow = np.clip(R_follow, -1.0, 1.0)

        v_ref = self.config.get('v_ref', 6.0)
        k_v   = 0.2
        R_speed = -k_v * (self.ego_car.vx - v_ref)**2

        ds = self.s_norm - self.prev_s_norm
        self.prev_s_norm = self.s_norm
        k_s = 5.0
        R_prog = k_s * ds
        
        reward = R_follow + R_speed + R_prog + R_out + R_crash

        return reward
    
    def _get_distance_cost(self) -> float:
        ego_x = self.ego_car.X
        ego_y = self.ego_car.Y

        vehicles = (
            self.actor.lane_1_vehicles
            + self.actor.lane_2_vehicles
            + self.actor.lane_3_vehicles
        )

        if len(vehicles) == 0:
            base_cost = 0.0
        else:
            d_min = np.inf
            for veh in vehicles:
                dx = veh.X - ego_x
                dy = veh.Y - ego_y
                d = np.hypot(dx, dy)
                if d < d_min:
                    d_min = d

            if d_min >= self.safe_distance:
                base_cost = 0.0
            else:
                if self.distance_cost_mode == "binary":
                    base_cost = 1.0
                else:
                    base_cost = (self.safe_distance - d_min) / max(self.safe_distance, 1e-6)

        return float(base_cost)
    
    def _get_truncated(self) -> None:
        truncated = True if self.step_count*self.dt >= self.duration else False
        if truncated == True:
            self.terminated = True
        return truncated
        
    def _get_done(self) -> None:
        """ collision check """
        self.collision = False
        vehicles = self.actor.lane_1_vehicles + self.actor.lane_2_vehicles + self.actor.lane_3_vehicles
        for veh in vehicles:
            if sat_collision(vehicle_poly_np(veh), vehicle_poly_np(self.ego_car)):
                self.collision = True
                break
        
        """ run out check """
        X, Y = self.ego_car.X, self.ego_car.Y
        stright_1 = X>=24 and X<=70 and Y>-4*self.num_lane and Y<4*self.num_lane
        stright_2 = X>-4*self.num_lane and X<4*self.num_lane and Y>24 and Y<70
        stright_3 = X>=-70 and X<=-24 and Y>-4*self.num_lane and Y<4*self.num_lane
        stright_4 = X>-4*self.num_lane and X<4*self.num_lane and Y>-70 and Y<-24
        circle_1 = (X-24)**2 + (Y-24)**2 > 12**2 and X>=0 and X<24 and Y>=0 and Y<24
        circle_2 = (X+24)**2 + (Y-24)**2 > 12**2 and X>-24 and X<=0 and Y>=0 and Y<24
        circle_3 = (X+24)**2 + (Y+24)**2 > 12**2 and X>-24 and X<=0 and Y>-24 and Y<=0
        circle_4 = (X-24)**2 + (Y+24)**2 > 12**2 and X>=0 and X<24 and Y>-24 and Y<=0
        
        if stright_1 or stright_2 or stright_3 or stright_4 or circle_1 or circle_2 or circle_3 or circle_4:
            self.runout = False
        else:
            self.runout = True

        isend_phase_1 = self.ego_car.target_phase == 1 and self.ego_car.Y >= 50 and self.ego_car.X > 0 and self.ego_car.X < 4*self.num_lane
        isend_phase_2 = self.ego_car.target_phase == 2 and self.ego_car.X <=-50 and self.ego_car.Y > 0 and self.ego_car.Y < 4*self.num_lane
        isend_phase_3 = self.ego_car.target_phase == 3 and self.ego_car.Y <=-50 and self.ego_car.X >-4*self.num_lane and self.ego_car.X < 0
        isend_phase_4 = self.ego_car.target_phase == 4 and self.ego_car.X >= 50 and self.ego_car.Y >-4*self.num_lane and self.ego_car.Y < 0
        
        done = True if self.runout or self.collision or isend_phase_1 or isend_phase_2 or isend_phase_3 or isend_phase_4 else False
        
        return done

    def render(self) -> None:
        if self.render_mode == "human":
            vehicles = [self.ego_car] + self.actor.lane_1_vehicles + self.actor.lane_2_vehicles + self.actor.lane_3_vehicles
            self.gui.render(vehicles, self.accel, self.steer)
            if not self.gui.check_quit_event():
                self.terminated = True

    def close(self) -> None:
        if self.render_mode == "human":
            self.gui.close()



def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi