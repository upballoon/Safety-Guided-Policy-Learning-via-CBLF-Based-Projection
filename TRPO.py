import os
import json
import argparse
import math
from CrossIntersection.CrossIntersections_v0 import CrossEnv
import scipy.optimize
import time
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.std_layer = nn.Linear(args.hidden_width, args.action_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, log_std, std

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.value_head = nn.Linear(args.hidden_width, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        state_values = self.value_head(x)
        return state_values


class ReplayBuffer(object):
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.Transition = namedtuple(
            "Transition", ("state", "action", "mask", "next_state", "reward")
        )
        self.memory = []

    def store(self, state, action, mask, next_state, reward):
        self.memory.append(self.Transition(state, action, mask, next_state, reward))

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

    def sample_tensors(self):
        batch = self.Transition(*zip(*self.memory))
        s   = torch.as_tensor(np.array(batch.state),      dtype=torch.float64, device=self.device)
        a   = torch.as_tensor(np.array(batch.action),     dtype=torch.float64, device=self.device)
        m   = torch.as_tensor(np.array(batch.mask),       dtype=torch.float64, device=self.device)  # 1 - done
        s_  = torch.as_tensor(np.array(batch.next_state), dtype=torch.float64, device=self.device)
        r   = torch.as_tensor(np.array(batch.reward),     dtype=torch.float64, device=self.device)
        self.clear()
        return s, a, m, s_, r


class TRPO:
    def __init__(self, args):
        self.max_action =  torch.tensor(args.max_action, dtype=torch.float)
        self.l2_reg    = args.l2_reg
        self.damping   = args.damping
        self.max_kl    = args.max_kl
        self.gamma     = args.gamma  # Discount factor
        self.lamda     = args.lamda
        self.chkpt_dir = args.save_dir
        self.date      = args.date
        self.device    = torch.device(args.device if torch.cuda.is_available() else "cpu")

        self.actor = Actor(args)
        self.critic = Critic(args)
    
    def choose_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float64, device=self.device).unsqueeze(0)
        action_mean, _, action_std = self.actor(Variable(state))
        action = torch.normal(action_mean, action_std)
        action = self.max_action * torch.tanh(action)
        return action.detach().numpy().flatten()

    def conjugate_gradients(self, Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def linesearch(self, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
        fval = f(True).data
        # print("fval before", fval.item())
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            set_flat_params_to(self.actor, xnew)
            newfval = f(True).data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                # print("fval after", newfval.item())
                return True, xnew
        return False, x

    def update(self, memory):
        states, actions, masks, s_, rewards = memory.sample_tensors()

        returns = torch.Tensor(actions.size(0),1)
        deltas = torch.Tensor(actions.size(0),1)
        advantages = torch.Tensor(actions.size(0),1)

        values = self.critic(Variable(states))

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + self.gamma * self.lamda * prev_advantage * masks[i]

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        targets = Variable(returns)

        # Original code uses the same LBFGS to optimize the value loss
        def get_value_loss(flat_params):
            set_flat_params_to(self.critic, torch.Tensor(flat_params))
            for param in self.critic.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = self.critic(Variable(states))
            value_loss = (values_ - targets).pow(2).mean()

            # weight decay
            for param in self.critic.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            value_loss.backward()
            return (value_loss.data.double().numpy(), get_flat_grad_from(self.critic).data.double().numpy())

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(self.critic).double().numpy(), maxiter=25)
        set_flat_params_to(self.critic, torch.Tensor(flat_params))

        advantages = (advantages - advantages.mean()) / advantages.std()

        action_means, action_log_stds, action_stds = self.actor(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = self.actor(Variable(states))
            else:
                action_means, action_log_stds, action_stds = self.actor(Variable(states))
                    
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()

        def get_kl():
            mean1, log_std1, std1 = self.actor(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)
        
        loss = get_loss()
        grads = torch.autograd.grad(loss, self.actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        def Fvp(v):
            kl = get_kl()
            kl = kl.mean()

            grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, self.actor.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * self.damping

        stepdir = self.conjugate_gradients(Fvp, -loss_grad, 10)
        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm[0]
        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        # print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))
        prev_params = get_flat_params_from(self.actor)
        success, new_params = self.linesearch(get_loss, prev_params, fullstep, neggdotstepdir / lm[0])
        set_flat_params_to(self.actor, new_params)

        return loss
    
    def save_models(self):
        print("---saving models---")
        agent_actor = os.path.join(self.chkpt_dir, f"actor_{self.date}.pl")
        torch.save(self.actor.state_dict(), agent_actor)

    def load_models(self, date):
        print("---loading model---")
        agent_actor = os.path.join(self.chkpt_dir, f"actor_{date}.pl")
        self.actor.load_state_dict(torch.load(agent_actor))


def train(args, env, agent):
    start_time = time.time()
    writer = SummaryWriter(log_dir=args.log_dir)
    # 定义超参数
    params = vars(args)
    param_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
    writer.add_text('CommandLineParameters', param_text, global_step=0)
    with open(os.path.join(args.log_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # running_state = ZFilter((args.state_dim,), clip=5.0)
    memory = ReplayBuffer(device=args.device)
    
    avg_reward, avg_cost, best_score = 0, 0, float("-inf")
    ep_rewards, ep_costs = [], []

    steps_this_iter = 0      
    ep_rewards = []
    for episode in range(args.max_episodes):
        state, _ = env.reset()
        # state = running_state(state)
        ep_reward, ep_cost = 0.0, 0.0

        for step in range(args.max_steps):
            action = agent.choose_action(state)         

            next_state, reward, cost, terminated, truncated, info = env.step(action, True)
            done = terminated or truncated
            ep_reward += reward
            ep_cost += cost
            # next_state = running_state(next_state)

            mask = 0.0 if terminated else 1.0

            memory.store(state, action, mask, next_state, reward)
            steps_this_iter += 1
            state = next_state

            if steps_this_iter >= args.batch_size:
                agent.update(memory)
                steps_this_iter = 0
                break
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
        
        print('episode:',episode+1,
              '\tsteps:',step+1,
              '\tscore:%.1f'%ep_reward,
              '\tavg:%.1f' % avg_reward,
              '\tbest:%.1f'%best_score,
              '\tcost:%.1f'%ep_cost,
              '\tavg_cost:%.1f'%avg_cost,
              )
        if avg_reward > best_score:
            best_score = avg_reward
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



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.995)')
    parser.add_argument('--env_name', default="CrossInter", metavar='G', help='name of the environment to run')
    parser.add_argument('--agent_name', default="TRPO", metavar='G', help='name of the environment to run')
    parser.add_argument('--max_episodes', type=int, default=2000, metavar='G', help='name of the environment to run')
    parser.add_argument('--max_steps', type=int, default=300, metavar='G', help='name of the environment to run')
    parser.add_argument('--lamda', type=float, default=0.97, metavar='G', help='gae (default: 0.97)')
    parser.add_argument('--l2_reg', type=float, default=1e-3, metavar='G', help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--max_kl', type=float, default=1e-1, metavar='G', help='max kl value (default: 1e-2)')
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument('--damping', type=float, default=1e-1, metavar='G', help='damping (default: 1e-1)')
    parser.add_argument('--batch_size', type=int, default=2048, metavar='N', help='random seed (default: 1)')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--log_interval', type=int, default=1, metavar='N', help='interval between training status logs (default: 10)')
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.v_ref = 6.0
    level = 2
    config = {
        "level": level,
        "lane_1_max": level,
        "lane_2_max": level,
        "lane_3_max": level,
        "v_ref":args.v_ref,
    }

    env = CrossEnv(config=config)   # render_mode="human", 
    args.config = env.config
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = list(env.action_space.high)
    start_date = datetime.now().strftime("%Y.%m.%d %H:%M")
    args.date = datetime.now().strftime("%Y%m%d%H%M")
    args.save_dir = "agent/{}/{}/{}".format(args.env_name, args.agent_name, args.date)
    args.log_dir = "logs/{}/{}/TRPO_level{}_{}".format(args.env_name, args.agent_name, level, args.date)
    
    agent = TRPO(args)

    train(args, env, agent)
