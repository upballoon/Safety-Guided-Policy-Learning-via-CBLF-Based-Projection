import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from CrossIntersection.CrossIntersections_v0 import CrossEnv
import argparse
import json
import os
import copy
import time
from datetime import datetime
import numpy as np
np.set_printoptions(suppress=True, precision=2)
import warnings
warnings.filterwarnings("ignore")
from cbf_utils import safe_action

def eval(args, level, config):
    
    writer = SummaryWriter(log_dir=args.log_dir)
    params = vars(args)
    param_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
    writer.add_text('CommandLineParameters', param_text, global_step=0)
    with open(os.path.join(args.log_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(params, f, indent=4)


    env = CrossEnv(render_mode="human", config=config)
    ep_rewards, ep_costs = [], []

    for episode in range(1000):
        ep_reward, ep_cost = 0, 0
        env.reset()
        
        for step in range(args.max_steps):
            action = safe_action(env, args, 
                                 k0_cbf = 1.0,
                                 k1_cbf = 2.0,
                                 k3_cbf = 1.0,
                                 k4_cbf = 2.0,
                                 k_y    = 500,
                                 k_psi  = 0.1,
                                 k_v    = 0.1,)
            next_state, reward, cost, done, terminated, info = env.step(action, True)
            
            ep_reward += reward
            ep_cost += cost

            env.render()
            
            if done:
                break
        ep_rewards.append(ep_reward)
        ep_costs.append(ep_cost)
        avg_reward = float(np.mean(ep_rewards[-50:]))
        avg_cost = float(np.mean(ep_costs[-50:]))
        writer.add_scalar('Reward/reward', ep_reward, episode)
        writer.add_scalar('Reward/avg', avg_reward, episode)
        writer.add_scalar('Cost/cost', ep_cost, episode)
        writer.add_scalar('Cost/avg_cost', avg_cost, episode)
        writer.add_scalar('Info/collision', env.collision, episode)
        writer.add_scalar('Info/step', step, episode)
        print('episode:',episode+1,
              '\tsteps:',step+1,
              '\tscore:%.1f'%ep_reward)
    env.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for SAC-continuous")
    parser.add_argument("--max_steps", type=int, default=300, help="300")
    args = parser.parse_args()
    
    args.agent_name = "CBF"
    args.env_name = "CrossInter"

    level = 2
    args.v_ref = 6.0
    config = {
        "lane_1_max": level,
        "lane_2_max": level,
        "lane_3_max": level,
        "v_ref":args.v_ref,
    }
    
    start_date = datetime.now().strftime("%Y.%m.%d %H:%M")
    args.date = datetime.now().strftime("%Y%m%d%H%M")
    args.save_dir = "agent/{}/{}/{}".format(args.env_name, args.agent_name, args.date)
    args.log_dir = 'logs/{}/{}/level{}_{}'.format(args.env_name, args.agent_name, level, args.date)

    eval(args, level, config)