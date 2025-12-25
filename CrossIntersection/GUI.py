import pygame
import pygame.gfxdraw
import math
import os

WINDOW_WIDTH = 1000 
WINDOW_HEIGHT = 1000
# os.environ['SDL_VIDEO_WINDOW_POS'] = "{},{}".format(-1000, 400)
SCALE, OFFX, OFFY = 8, 500, 500
W, H = OFFX * 2, OFFY * 2
WINDOW_TITLE = "Cross Intersection"
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

class GUI_Visualizer:
    def __init__(self, xe, ye, xc, yc, x_yel, y_yel, x_white, y_white):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(WINDOW_TITLE)
        self.font = pygame.font.SysFont(None, 20)
    
        self.clock = pygame.time.Clock()  
        self.colors = [(255, 200, 0), (30, 144, 255), (30, 144, 255), (30, 144, 255), (30, 144, 255)]
        self.road_layer = pygame.Surface((W, H), pygame.SRCALPHA)
        self.draw_road(xe, ye, xc, yc, x_yel, y_yel, x_white, y_white)
    
    def render(self, vehicles, acc_cmd, steer_cmd):
        self.screen.fill((0, 50, 0))
        self.screen.blit(self.road_layer, (0, 0))
        for veh in vehicles:
            self.draw_vehicle(veh)
            
        lead = vehicles[0]
        debug_texts = [
            (f"x={lead.X:5.2f}", 10, 10),
            (f"y={lead.Y:5.2f}", 100, 10),
            (f"yaw={math.degrees(lead.psi):6.1f}°", 200, 10),
            (f"v={lead.v:4.2f}", 10, 30),
            (f"accel={acc_cmd:5.2f}", 10, 50),
            (f"steer={steer_cmd:5.2f}", 100, 50)
        ]
        
        self.draw_debug_texts(debug_texts, clear_rect=(0, 0, 400, 100))
        pygame.display.flip()
       
    def draw_road(self, xe, ye, xc, yc, center_x, center_y, x_white, y_white):
        road_poly = [world_to_screen(xe[i], ye[i], SCALE, OFFX, OFFY)
                    for i in range(len(xe))]
        pygame.draw.polygon(self.road_layer, (47, 47, 47), road_poly)

        for i in range(len(road_poly)):
            pygame.draw.line(self.road_layer, WHITE,
                            road_poly[i], road_poly[(i+1) % len(road_poly)], 2)
        for x_line, y_line in zip(center_x, center_y):
            pts = [world_to_screen(x_line[i], y_line[i], SCALE, OFFX, OFFY)
                for i in range(len(x_line))]
            pygame.draw.lines(self.road_layer, YELLOW, False, pts, 2)
        # for x_line, y_line in zip(x_white, y_white):
        #     pts = [world_to_screen(x_line[i], y_line[i], SCALE, OFFX, OFFY)
        #         for i in range(len(x_line))]
        #     pygame.draw.lines(self.road_layer, WHITE, False, pts, 2)
        # 车道线（改成虚线）
        for x_line, y_line in zip(x_white, y_white):
            pts = [world_to_screen(x_line[i], y_line[i], SCALE, OFFX, OFFY)
                for i in range(0, len(x_line), 10)]
            for i in range(0, len(pts) - 1, 4):
                pygame.draw.line(self.road_layer, WHITE, pts[i], pts[i + 1], 1)
        # for d in range(4):
        #     for lane in range(3):
        #         pts = [world_to_screen(xc[d, i, lane], yc[d, i, lane], SCALE, OFFX, OFFY)
        #             for i in range(0, 3000, 10)]
        #         for i in range(0, len(pts)-1, 4):
        #             pygame.draw.line(self.road_layer, WHITE, pts[i], pts[i+1], 1)
    
    def draw_vehicle(self, vehicle): # 绘制车辆
        pts = [world_to_screen(px, py, SCALE, OFFX, OFFY)
            for px, py in vehicle.get_world_vertices()]
        pygame.gfxdraw.filled_polygon(self.screen, pts, vehicle.color)
        pygame.gfxdraw.aapolygon(self.screen, pts, (0, 0, 0))  
    
    def draw_debug_texts(self, texts, clear_rect=None):
        if clear_rect: 
            self.screen.fill((0, 50, 0), clear_rect) 
        for text, x, y in texts:
            txt_surf = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(txt_surf, (x, y))
            
    def check_quit_event(self):
        running = True
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        return running        
    
    def close(self):
        pygame.quit()

def world_to_screen(x, y, scale, offx, offy):
    return int(x * scale + offx), int(-y * scale + offy)

