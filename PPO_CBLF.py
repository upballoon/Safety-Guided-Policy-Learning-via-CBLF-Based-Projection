
from torch.utils.tensorboard import SummaryWriter
# import safety_gymnasium as gym
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
import os
import time
from datetime import datetime
import numpy as np
import json
from CrossIntersection.CrossIntersections_v0 import CrossEnv
from cbf_utils import safe_action, solve_cbf_qp, push_buffer, CBFContext

class RunningMeanStd:
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = torch.tensor(args.max_action, dtype=torch.float32)
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh] 

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_continuous():
    def __init__(self, args):
        self.max_action = torch.tensor(args.max_action, dtype=torch.float32)
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_episodes *args.max_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.lr_l = args.lr_a
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.use_lagrangian = True
        self.adaptive_lambda = True
        
        self.chkpt_dir = args.save_dir
        self.date = args.date
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        self.actor = Actor(args)
        self.critic = Critic(args)

        if self.adaptive_lambda:
            self.log_lambda = torch.zeros(1).to(self.device)
            self.log_lambda.requires_grad = True
            self.lam_lag = self.log_lambda.exp().detach()
            self.lambda_optimizer = torch.optim.Adam([self.log_lambda], lr=self.lr_l)
        else:
            self.lam_lag = 10.0

        if self.set_adam_eps:  
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        
        self.log = {'loss_safe':np.nan, 'violation_mean': 0}

    def evaluate(self, s):  # When evaluating the policy, we only use the mean  
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, replay_buffer, total_steps):
        batch_s, batch_a, batch_a_log, batch_a_safe, batch_r, batch_s_, batch_done, batch_cbfs = replay_buffer.sample()
        adv = []
        gae = 0
        with torch.no_grad(): 
            vs = self.critic(batch_s)
            vs_ = self.critic(batch_s_)
            deltas = batch_r + self.gamma * (1.0 - batch_done) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(batch_done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(batch_s[index])
                a_now_raw = dist_now.rsample()
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)
                a_logprob_now = dist_now.log_prob(batch_a[index])
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - batch_a_log[index].sum(1, keepdim=True))

                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy 

                if self.use_lagrangian:
                    violation = torch.clamp(batch_cbfs[index], min=-10.0, max=1.0)
                    loss_safe = F.mse_loss(a_now_raw, batch_a_safe[index])
                    actor_loss = actor_loss + self.lam_lag * loss_safe
                    self.log['loss_safe'] = loss_safe.detach().cpu().flatten()
                    self.log['violation_mean'] = violation.mean().detach().cpu().flatten()
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(batch_s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                # Update lambda
                if self.use_lagrangian:
                    if self.adaptive_lambda:
                        penalty = torch.mean(violation)
                        lagrangian_loss = torch.mean(self.log_lambda.exp() * penalty.detach())
                        self.lambda_optimizer.zero_grad()
                        lagrangian_loss.backward()
                        self.lambda_optimizer.step()
                        self.lam_lag = self.log_lambda.exp().detach()
                    else:
                        self.lam_lag += self.lr_l * penalty.detach().mean().item()
                        self.lam_lag = max(self.lam_lag, 0.0)

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def save_models(self):
        print("---saving models---")
        agent_actor = os.path.join(self.chkpt_dir, f"actor_{self.date}.pl")
        torch.save(self.actor.state_dict(), agent_actor)

    def load_models(self, date):
        print("---loading model---")
        agent_actor = os.path.join(self.chkpt_dir, f"actor_{date}.pl")
        self.actor.load_state_dict(torch.load(agent_actor))


class ReplayBuffer:
    def __init__(self, args):
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.cbf_nums = args.cbf_nums
        self.batch_size = int(args.batch_size)

        self.state = np.zeros((self.batch_size, self.state_dim))
        self.action = np.zeros((self.batch_size, self.action_dim))
        self.action_log = np.zeros((self.batch_size, self.action_dim))
        self.reward = np.zeros((self.batch_size, 1))
        self.next_state = np.zeros((self.batch_size, self.state_dim))
        self.done = np.zeros((self.batch_size, 1))
        self.a_safe = np.zeros((self.batch_size, self.action_dim))
        self.cbfs = np.zeros((self.batch_size, self.cbf_nums))
        self.count = 0

    def store(self, state, action, action_log, a_safe, reward, next_state, done, CBFs):
        self.state[self.count]      = state
        self.action[self.count]     = action
        self.action_log[self.count] = action_log
        self.reward[self.count]     = reward
        self.next_state[self.count] = next_state
        self.done[self.count]       = done
        self.a_safe[self.count]     = a_safe
        self.cbfs[self.count][0:len(CBFs)] = CBFs
        self.count += 1

    def sample(self):
        batch_s      = torch.tensor(self.state, dtype=torch.float)
        batch_a      = torch.tensor(self.action, dtype=torch.float)
        batch_a_log  = torch.tensor(self.action_log, dtype=torch.float)
        batch_a_safe = torch.tensor(self.a_safe, dtype=torch.float)
        batch_r      = torch.tensor(self.reward, dtype=torch.float)
        batch_s_     = torch.tensor(self.next_state, dtype=torch.float)
        batch_done   = torch.tensor(self.done, dtype=torch.float)
        batch_cbfs   = torch.tensor(self.cbfs, dtype=torch.float)
        self.count   = 0
        self.reset() 
        return batch_s, batch_a, batch_a_log, batch_a_safe, batch_r, batch_s_, batch_done, batch_cbfs
    
    def reset(self): 
        self.state      = np.zeros((self.batch_size, self.state_dim))
        self.action     = np.zeros((self.batch_size, self.action_dim))
        self.action_log = np.zeros((self.batch_size, self.action_dim))
        self.reward     = np.zeros((self.batch_size, 1))
        self.next_state = np.zeros((self.batch_size, self.state_dim))
        self.done       = np.zeros((self.batch_size, 1))
        self.a_safe = np.zeros((self.batch_size, self.action_dim))
        self.cbfs = np.zeros((self.batch_size, self.cbf_nums))
        self.count      = 0


def train(args, agent, env):
    start_time = time.time()
    writer = SummaryWriter(log_dir=args.log_dir)
    params = vars(args)
    param_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
    writer.add_text('CommandLineParameters', param_text, global_step=0)
    with open(os.path.join(args.log_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    avg_reward, avg_cost, best_reward = 0, 0, float("-inf")
    ep_rewards, ep_costs = [], []
    total_steps = 0
    for episode in range(args.max_episodes):
        state = env.reset()[0]
        ep_reward, ep_cost = 0, 0
        for step in range(args.max_steps):
            action, action_log = agent.choose_action(state) 

            a_safe, CBFs = safe_action(env, args, action, k0_cbf=1.0, k1_cbf=2.0, k3_cbf=1.0, k4_cbf=2.0, k_y=500, k_psi=0.1, k_v=0.1)

            next_state, reward, cost, done, terminated, info = env.step(action, True)
            
            ep_reward += reward
            ep_cost += cost

            replay_buffer.store(state, action, action_log, a_safe, reward, next_state, done, CBFs)
            
            state = next_state
            total_steps += 1
            env.render()
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
        ep_rewards.append(ep_reward)
        ep_costs.append(ep_cost)
        avg_reward = float(np.mean(ep_rewards[-50:]))
        avg_cost = float(np.mean(ep_costs[-50:]))
        writer.add_scalar('Reward/reward', ep_reward, episode)
        writer.add_scalar('Reward/avg', avg_reward, episode)
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
        if avg_reward > best_reward:
            best_reward = avg_reward
            if episode > 100:
                folder_path = args.save_dir
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                agent.save_models()
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
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--env_name", type=str, default="CrossInter")
    parser.add_argument("--agent_name", type=str, default="PPO")
    parser.add_argument("--render_mode", type=str, default="rgb_array")
    parser.add_argument("--max_episodes", type=int, default=int(2000), help=" Maximum number of training episodes")
    parser.add_argument("--max_steps", type=int, default=int(300), help=" Maximum number of training episodes")
    parser.add_argument("--batch_size", type=int, default=int(2048), help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=bool, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    level = 1
    args.v_ref = 6.0
    config = {
        "level": level,
        "lane_1_max": level,
        "lane_2_max": level,
        "lane_3_max": level,
        "v_ref":args.v_ref,
    }
    args.cbf_nums = level*4*3 + 2 
    env = CrossEnv(config=config) 
    args.config = config
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = list(env.action_space.high)
    
    args.agent_name = "PPO_S"
    args.env_name = "CrossInter"

    args.date = datetime.now().strftime("%Y%m%d%H%M")
    args.save_dir = "agent/{}/{}/{}".format(args.env_name, args.agent_name, args.date)
    args.log_dir = "logs/{}/{}/PPOC_level{}_{}".format(args.env_name, args.agent_name, level, args.date)
    start_date = datetime.now().strftime("%Y.%m.%d %H:%M")

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)
    
    train(args, agent, env)