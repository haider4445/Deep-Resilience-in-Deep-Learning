import torch
import torch.nn as nn        # Pytorch neural network package
import torch.optim as optim  # Pytorch optimization package
import gym
import gym.spaces
import warnings
import time
import numpy as np
import collections
from dqn import DQN, ExperienceReplay, Agent
from wrappers import make_env
from torch.utils.tensorboard import SummaryWriter
import datetime
import argparse
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description = "Fast Undetectable Attack")
parser.add_argument('-mp','--Path', metavar = 'path', nargs = "?", default = '', type = str, help = 'Complete path to model folder')
parser.add_argument('-e','--env', type = str, nargs = "?", default = "PongNoFrameskip-v4", help = 'Environment name like PongNoFrameskip-v4')
args = parser.parse_args()

DirectoryPath = args.Path 
DEFAULT_ENV_NAME = args.env 
device = torch.device("cuda")
MEAN_REWARD_BOUND = 1100         
gamma = 0.99                   
batch_size = 32                
replay_size = 10000            
learning_rate = 1e-4           
sync_target_frames = 1000      
replay_start_size = 10000      
eps_start=1.0
eps_decay=.999985
eps_min=0.02

if __name__ == '__main__':
	print(">>>Training starts at ",datetime.datetime.now())

	env = make_env(DEFAULT_ENV_NAME)
	net = DQN(env.observation_space.shape, env.action_space.n).to(device)
	target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
	writer = SummaryWriter(comment="-" + DEFAULT_ENV_NAME)
	 
	buffer = ExperienceReplay(replay_size)
	agent = Agent(env, buffer)

	epsilon = eps_start
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)
	total_rewards = []
	frame_idx = 0  
	best_mean_reward = None

	while True:
	        frame_idx += 1
	        epsilon = max(epsilon*eps_decay, eps_min)

	        reward = agent.play_step(net, epsilon, device=device)
	        if reward is not None:
	            total_rewards.append(reward)

	            mean_reward = np.mean(total_rewards[-100:])

	            print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (
	                frame_idx, len(total_rewards), mean_reward, epsilon))
	            
	            writer.add_scalar("epsilon", epsilon, frame_idx)
	            writer.add_scalar("reward_100", mean_reward, frame_idx)
	            writer.add_scalar("reward", reward, frame_idx)

	            if best_mean_reward is None or best_mean_reward < mean_reward:
	                torch.save(net.state_dict(), DirectoryPath + DEFAULT_ENV_NAME + "-best.dat")
	                best_mean_reward = mean_reward
	                if best_mean_reward is not None:
	                    print("Best mean reward updated %.3f" % (best_mean_reward))

	            if mean_reward > MEAN_REWARD_BOUND:
	                print("Solved in %d frames!" % frame_idx)
	                break

	        if len(buffer) < replay_start_size:
	            continue

	        batch = buffer.sample(batch_size)
	        states, actions, rewards, dones, next_states = batch

	        states_v = torch.tensor(states).to(device)
	        next_states_v = torch.tensor(next_states).to(device)
	        actions_v = torch.tensor(actions).to(device)
	        rewards_v = torch.tensor(rewards).to(device)
	        done_mask = torch.ByteTensor(dones).to(device)

	        state_action_values = net(states_v).gather(1, actions_v.type(torch.int64).unsqueeze(-1)).squeeze(-1)

	        next_state_values = target_net(next_states_v).max(1)[0]

	        next_state_values[done_mask] = 0.0

	        next_state_values = next_state_values.detach()

	        expected_state_action_values = next_state_values * gamma + rewards_v

	        loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

	        optimizer.zero_grad()
	        loss_t.backward()
	        optimizer.step()

	        if frame_idx % sync_target_frames == 0:
	            target_net.load_state_dict(net.state_dict())
	       
	writer.close()
	print(">>>Training ends at ",datetime.datetime.now())



