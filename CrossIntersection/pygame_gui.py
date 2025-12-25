
import pygame
import pygame.gfxdraw
import math
from RoadModel import road_model2      

class Vehicle:
    """自行车运动学模型"""
    def __init__(self, x=0, y=0, yaw=0, v=0,
                 wheelbase=2.5,  w=4.8, h=2.1,
                 delta_limit=math.radians(30), a_limit=3.0):
        self.x, self.y, self.yaw, self.v = x, y, yaw, v
        self.L = wheelbase
        self.w, self.h = w, h
        self.delta_limit, self.a_limit = delta_limit, a_limit

    def step(self, a_cmd, delta_cmd, dt):
        a     = max(-self.a_limit,   min(self.a_limit,   a_cmd))
        delta = max(-self.delta_limit, min(self.delta_limit, delta_cmd))
        self.x   += self.v * math.cos(self.yaw) * dt
        self.y   += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / self.L * math.tan(delta) * dt
        self.v   += a * dt

    def get_world_vertices(self):
        half_w, half_h = self.w / 2, self.h / 2
        local = [(-half_w,  half_h), ( half_w,  half_h),
                 ( half_w, -half_h), (-half_w, -half_h)]
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        return [(self.x + dx * cos_y - dy * sin_y,
                 self.y + dx * sin_y + dy * cos_y) for dx, dy in local]

def world_to_screen(x, y, scale, offx, offy):
    return int(x * scale + offx), int(-y * scale + offy)

def draw_vehicle(screen, vehicle, scale, offx, offy, color):
    pts = [world_to_screen(px, py, scale, offx, offy)
           for px, py in vehicle.get_world_vertices()]
    pygame.gfxdraw.filled_polygon(screen, pts, color)
    pygame.gfxdraw.aapolygon  (screen, pts, (0, 0, 0))  

def draw_road(screen, xe, ye, xc, yc, center_x, center_y,
              scale, offx, offy):
    road_poly = [world_to_screen(xe[i], ye[i], scale, offx, offy)
                 for i in range(len(xe))]
    pygame.draw.polygon(screen, (47, 47, 47), road_poly)

    for i in range(len(road_poly)):
        pygame.draw.line(screen, (255, 255, 255),
                         road_poly[i], road_poly[(i+1) % len(road_poly)], 2)

    for x_line, y_line in zip(center_x, center_y):
        pts = [world_to_screen(x_line[i], y_line[i], scale, offx, offy)
               for i in range(len(x_line))]
        pygame.draw.lines(screen, (255, 255, 0), False, pts, 2)

    white = (255, 255, 255)
    for d in range(4):
        for lane in range(3):
            pts = [world_to_screen(xc[d, i, lane], yc[d, i, lane], scale, offx, offy)
                   for i in range(0, 3000, 10)]
            for i in range(0, len(pts)-1, 4):
                pygame.draw.line(screen, white, pts[i], pts[i+1], 1)


def draw_debug_texts(screen, font, texts, clear_rect=None):
    if clear_rect:
        screen.fill((0, 50, 0), clear_rect) 
    for text, x, y in texts:
        txt_surf = font.render(text, True, (255, 255, 255))
        screen.blit(txt_surf, (x, y))
        
        
def main():
    SCALE, OFFX, OFFY = 10, 500, 500
    pygame.init()
    W, H = OFFX * 2, OFFY * 2
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Multi‑Vehicle Demo")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 20)

    xe, ye, xc, yc, curv, hdg, ctr_x, ctr_y = road_model2()
    road_layer = pygame.Surface((W, H), pygame.SRCALPHA)
    draw_road(road_layer, xe, ye, xc, yc, ctr_x, ctr_y, SCALE, OFFX, OFFY)

    vehicles = [
        Vehicle(x=  2,   y=-40, yaw= math.pi/2, v=10),  
        Vehicle(x=  2,   y=-50, yaw= math.pi/2, v=10),  
        Vehicle(x=  2,   y=-60, yaw= math.pi/2, v=10),  
        Vehicle(x=  2,   y=-70, yaw= math.pi/2, v=10),  
        Vehicle(x=  2,   y=-80, yaw= math.pi/2, v=10)   
    ]
    colors = [(255,200,  0), ( 30,144,255), ( 30,144,255),
              ( 30,144,255), ( 30,144,255)]            

    step = 0
    running = True
    while running:
        dt = clock.tick(60) / 1000.0    
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        acc_cmds   = [0.0, 0.0, 0.0, 0.0, 0.0]               
        steer_cmds = [0.3 * math.sin(step*0.03), 0,0,0,0]    

        for i, veh in enumerate(vehicles):
            veh.step(acc_cmds[i], steer_cmds[i], dt)

        screen.fill((0, 50, 0))
        screen.blit(road_layer, (0, 0))
        for veh, col in zip(vehicles, colors):
            draw_vehicle(screen, veh, SCALE, OFFX, OFFY, col)

        lead = vehicles[0]
        debug_texts = [
            (f"x={lead.x:5.2f}", 10, 10),
            (f"y={lead.y:5.2f}", 100, 10),
            (f"yaw={math.degrees(lead.yaw):6.1f}°", 200, 10),
            (f"v={lead.v:4.2f}", 10, 30),
            (f"accel={acc_cmds[0]:5.2f}", 10, 50),
            (f"steer={steer_cmds[0]:5.2f}", 100, 50)
        ]
        draw_debug_texts(screen, font, debug_texts, clear_rect=(0, 0, 400, 100))
        
        pygame.display.flip()
        step += 1
        if step >= 500:
            running = False

    pygame.quit()

if __name__ == "__main__":
    main()
