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
from cbf_utils import safe_action, solve_cbf_qp, push_buffer, CBFContext

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = torch.tensor(args.max_action, dtype=torch.float)
        self.l1 = nn.Linear(args.state_dim, args.hidden_size)
        self.l2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.mean_layer = nn.Linear(args.hidden_size, args.action_dim)
        self.log_std_layer = nn.Linear(args.hidden_size, args.action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x) 
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mean, std) 
        if deterministic: 
            action = mean
        else:
            action = dist.rsample() 
        if with_logprob:
            log_pi = dist.log_prob(action).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=1, keepdim=True)
        else:
            log_pi = None
        action = self.max_action * torch.tanh(action)
        return action, log_pi


class Critic(nn.Module):  
    def __init__(self, args):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(args.state_dim + args.action_dim, args.hidden_size)
        self.l2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.l3 = nn.Linear(args.hidden_size, 1)
        # Q2
        self.l4 = nn.Linear(args.state_dim + args.action_dim, args.hidden_size)
        self.l5 = nn.Linear(args.hidden_size, args.hidden_size)
        self.l6 = nn.Linear(args.hidden_size, 1)

    def forward(self, state, action):
        s_a = torch.cat([state, action], 1)
        q1 = torch.tanh(self.l1(s_a))
        q1 = torch.tanh(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.tanh(self.l4(s_a))
        q2 = torch.tanh(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    

class ReplayBuffer(object):
    def __init__(self, args):
        self.max_size = int(args.batch_size)
        self.count = 0
        self.size = 0
        self.state = np.zeros((self.max_size, args.state_dim))
        self.action = np.zeros((self.max_size, args.action_dim))
        self.a_safe = np.zeros((self.max_size, args.action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.cbfs = np.zeros((self.max_size, args.cbf_nums))
        self.next_state = np.zeros((self.max_size, args.state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, state, action, a_safe, r, next_state, dw, cbfs):
        self.state[self.count] = state
        self.action[self.count] = action
        self.a_safe[self.count] = a_safe
        self.r[self.count] = r
        self.cbfs[self.count][0:len(cbfs)] = cbfs
        self.next_state[self.count] = next_state
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  
        self.size = min(self.size + 1, self.max_size)  

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.state[index], dtype=torch.float).to(args.device)
        batch_a = torch.tensor(self.action[index], dtype=torch.float).to(args.device)
        batch_a_safe = torch.tensor(self.a_safe[index], dtype=torch.float).to(args.device)
        batch_r = torch.tensor(self.r[index], dtype=torch.float).to(args.device)
        batch_s_ = torch.tensor(self.next_state[index], dtype=torch.float).to(args.device)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float).to(args.device)
        batch_cbfs = torch.tensor(self.cbfs[index], dtype=torch.float).to(args.device)
        return batch_s, batch_a, batch_a_safe, batch_r, batch_s_, batch_dw, batch_cbfs


class SAC(object):
    def __init__(self, args):
        self.mini_batch   = args.mini_batch
        self.gamma        = args.gamma
        self.TAU          = args.TAU
        self.lr           = args.lr_a
        self.chkpt_dir    = args.save_dir
        self.date         = args.date
        self.device       = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.max_action      = args.max_action
        self.adaptive_lambda = True
        self.use_lagrangian  = True

        self.actor         = Actor(args).to(self.device)
        self.critic        = Critic(args).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer       = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer      = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.adaptive_alpha  = True
        if self.adaptive_alpha:   
            self.target_entropy = -args.action_dim
            self.log_alpha = torch.zeros(1).to(self.device) 
            self.log_alpha.requires_grad = True
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2

        if self.adaptive_lambda:
            self.log_lambda = torch.zeros(1).to(self.device)
            self.log_lambda.requires_grad = True
            self.lam_lag = self.log_lambda.exp().detach()
            self.lambda_optimizer = torch.optim.Adam([self.log_lambda], lr=self.lr)
        else:
            self.lam_lag = 10.0
        
        self.log = {'loss_safe':np.nan, 'violation_mean': 0}        

    def evalueate(self, state):
        state  = torch.as_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state, deterministic = True)
        return action.cpu().numpy().flatten()

    def choose_action(self, state):
        state  = torch.as_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state, with_logprob=False)
            
        return action.cpu().numpy().flatten()

    def update(self, buffer):
        batch_s, batch_a, batch_a_safe, batch_r, batch_s_, batch_dw, batch_cbfs = buffer.sample(self.mini_batch)

        with torch.no_grad():
            a_raw_next, log_pi_next = self.actor(batch_s_)
            target_Q1, target_Q2 = self.critic_target(batch_s_, a_raw_next)
            target_Q_min = torch.min(target_Q1, target_Q2)
            TD_target = batch_r + self.gamma * (1 - batch_dw) * (target_Q_min - self.alpha * log_pi_next)
        
        # Update critic
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        critic_loss = F.mse_loss(current_Q1, TD_target) + F.mse_loss(current_Q2, TD_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic parameter
        for params in self.critic.parameters():
            params.requires_grad = False
        
        # Compute actor loss
        a_raw, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a_raw)
        Q = torch.min(Q1, Q2)
        actor_loss = torch.mean(self.alpha * log_pi - Q)

        if self.use_lagrangian:
            violation = torch.clamp(batch_cbfs, min=-10.0, max=1.0)
            loss_safe = F.mse_loss(a_raw, batch_a_safe)
            actor_loss = actor_loss + self.lam_lag * loss_safe
            self.log['loss_safe'] = loss_safe.detach().cpu().flatten()
            self.log['violation_mean'] = violation.mean().detach().cpu().flatten()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True
                
        # Update alpha
        if self.adaptive_alpha:
            alpha_loss = -torch.mean(self.log_alpha.exp() * (log_pi + self.target_entropy).detach())
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha += self.lr * log_pi.exp().mean().item() - self.target_entropy
        
        # Update lambda
        if self.use_lagrangian:
            if self.adaptive_lambda:
                penalty = torch.mean(violation)
                # penalty = torch.min(violation)
                lagrangian_loss = torch.mean(self.log_lambda.exp() * penalty.detach())
                self.lambda_optimizer.zero_grad()
                lagrangian_loss.backward()
                self.lambda_optimizer.step()
                self.lam_lag = self.log_lambda.exp().detach()
            else:
                self.lam_lag += self.lr * penalty.detach().mean().item()
                self.lam_lag = max(self.lam_lag, 0.0)

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
        

    def save_models(self):
        print("---saving models---")
        agent_actor = os.path.join(self.chkpt_dir, f"actor_{self.date}.pl")
        torch.save(self.actor.state_dict(), agent_actor)

    def load_models(self, date):
        print("---loading model---")
        agent_actor = os.path.join(self.chkpt_dir, f"actor_{date}.pl")
        self.actor.load_state_dict(torch.load(agent_actor))


def train(args, env):
    start_time = time.time()
    writer = SummaryWriter(log_dir=args.log_dir)
    params = vars(args)
    param_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
    writer.add_text('CommandLineParameters', param_text, global_step=0)
    with open(os.path.join(args.log_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    agent = SAC(args)
    buffer = ReplayBuffer(args)

    ep_rewards, ep_costs = [], []
    avg_reward, avg_cost = 0, 0
    best_reward = -np.inf
    total_steps = 0  

    for episode in range(args.max_episodes):
        state, _ = env.reset()
        ep_reward, ep_cost = 0, 0
        for step in range(args.max_steps):
            if total_steps < args.mini_batch:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)

            a_safe, CBFs = safe_action(env, args, action, k0_cbf=1.0, k1_cbf=2.0, k3_cbf=1.0, k4_cbf=2.0, k_y=500, k_psi=0.1, k_v=0.1)

            next_state, reward, cost, done, terminated, info = env.step(action, True)
            buffer.store(state, action, a_safe, reward, next_state, done, CBFs)
            
            state = next_state
            ep_reward  += reward
            ep_cost += cost
            if total_steps >= args.mini_batch:
                agent.update(buffer)
            total_steps += 1
            if done:
                break
            env.render()
        ep_rewards.append(ep_reward)
        ep_costs.append(ep_cost)
        avg_reward = np.mean(ep_rewards[-50:])
        avg_cost = np.mean(ep_costs[-50:])
        if avg_reward > best_reward:
            best_reward = avg_reward
            if episode > 100:
                folder_path = args.save_dir
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                agent.save_models()
        writer.add_scalar('Reward/reward', ep_reward, episode)
        writer.add_scalar('Reward/avg', avg_reward, episode)
        writer.add_scalar('Info/step', step+1, episode)
        writer.add_scalar('Cost/cost', ep_cost, episode)
        writer.add_scalar('Cost/avg_cost', avg_cost, episode)
        writer.add_scalar('Cost/lam', agent.lam_lag, episode)
        writer.add_scalar('Loss/safe', agent.log['loss_safe'], episode)
        writer.add_scalar('Loss/vio', agent.log['violation_mean'], episode)

        print('episode:',episode+1,
              '\tsteps:',step+1,
              '\tscore:%.1f'%ep_reward,
              '\tavg:%.1f' % avg_reward,
              '\tbest:%.1f'%best_reward,
              '\tcost:%.1f'%ep_cost,
              '\tavg_cost:%.1f'%avg_cost,
              )
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    print(f"{hours} h {minutes} m {seconds:.2f} s")
    writer.close()
    env.close()

    end_date = datetime.now().strftime("%Y.%m.%d %H:%M")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print(f"Training agent: {args.agent_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for SAC-continuous")
    parser.add_argument("--max_episodes", type=int, default=int(2000), help=" Maximum number of training episodes")
    parser.add_argument("--max_steps", type=int, default=300, help="300")
    parser.add_argument("--batch_size", type=int, default=1e6, help="Batch size")
    parser.add_argument("--mini_batch", type=int, default=300, help="mini Batch size")
    parser.add_argument("--hidden_size", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--TAU", type=int, default=0.005, help="ddpg parameter")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    level = 1
    args.v_ref = 6.0
    config = {
        "lane_1_max": level,
        "lane_2_max": level,
        "lane_3_max": level,
        "v_ref":args.v_ref,
        "level": level,
    }
    env = CrossEnv(config=config)   # render_mode="human", 
    args.config = env.config
    args.cbf_nums = level*4*3 + 2 
    
    args.agent_name = "SAC_S"
    args.env_name = "CrossInter"
    start_date = datetime.now().strftime("%Y.%m.%d %H:%M")
    args.date = datetime.now().strftime("%Y%m%d%H%M")
    args.save_dir = "agent/{}/{}/{}".format(args.env_name, args.agent_name, args.date)
    args.log_dir = 'logs/{}/{}/SACC_level{}_{}'.format(args.env_name, args.agent_name, level, args.date)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = list(env.action_space.high)
    
    train(args, env)
    