import random
import itertools
import numpy as np
import torch
import gym
import copy

class Strategy:

	def __init__(self, Totalsteps):
		self.Totalsteps = Totalsteps

	def randomStrategy(self):
		if random.randint(0,100) % 2 == 0:
			return False
		else:
			return True
	def leastStepsStrategy(self):
		pass
	
	def CriticalPointStrategy(self, M,n,net, state, env_orig,delta = 10, acts_mask = [], repeat_adv_act = 1, dam = "pong", domain = True, fullSearch = False ):

		adv_acts = []
		attack = False
		bestdiff = -1
		best_acts = []

		#deep copy env
		if isinstance(env_orig, gym.wrappers.Monitor):
			env = copy.deepcopy(env_orig.env)  # copy the environment
		else:
			env = copy.deepcopy(env_orig)       

		#calculate baseline state
		for i in range(M):
			state_v = torch.tensor(np.array([state], copy=False))
			q_vals = net(state_v).data.numpy()[0]
			orig_action = np.argmax(q_vals)
			adv_acts.append(orig_action)
			state, reward, done, _ = env.step(orig_action)
			#print("Baseline Calculation reward: ", reward)
		baselineState = state

		#print("Baseline state shape: ", baselineState.shape)

		#find attack strategies
		action_shape = env.action_space.shape or env.action_space.n
		action_shape = np.prod(action_shape)
		if acts_mask:
			actions = [a for a in range(int(action_shape)) if a in self.acts_mask]
		else:
			actions = [a for a in range(int(action_shape))]
		atk_strategies = [p for p in itertools.product(actions, repeat=n // repeat_adv_act)]  # define attack strategies
		atk_strategies = np.repeat(atk_strategies, repeat_adv_act, axis=-1)

		#print("Attack strategies: ", atk_strategies)
		#print("Attack strategies shape: ", atk_strategies.shape)


		#find damage without attack
		if domain == True:
			if dam == "pong":
				#print(baselineState)
				std_dam = self.dam_pong(baselineState[0])
		else:
			std_dam = reward

		#find attack state
		for atk in atk_strategies:
			acts = list(atk)
			i = 0
			for adv_a in acts:
				if i < len(acts):
					adv_action = adv_a
				else:
					state_v = torch.tensor(np.array([state], copy=False))
					q_vals = net(state_v).data.numpy()[0]
					adv_action = np.argmax(q_vals)
					adv_acts.append(adv_action)
				state, reward, done, _ = env.step(adv_action)
				#print("Adversarial Calculation reward: ", reward)	

				if done:
					break
				i +=1
			attackState = state

			#find damage with attack
			if domain == True:
				if dam == "pong":
					atk_dam = self.dam_pong(attackState[0])
			else:
				#print(reward)
				atk_dam = reward

			#print("DAM Attack", atk_dam)
			#print("DAM without Attack", std_dam)


			difference = abs(atk_dam - std_dam)
			print("Actions before delta: ",adv_acts)
			print("its difference", difference)

			if difference > delta:# and atk_dam > std_dam:
				print("Actions so far: ",adv_acts)
				print("its difference", difference)

				attack = True
				adv_acts = acts
				if not fullSearch:
					return adv_acts, attack
				else:
					if difference >= bestdiff:
						bestdiff = difference
						best_acts = adv_acts
						print("Best actions so far",best_acts)
						print("its difference", difference)
			
		return best_acts, attack

	def dam_pong(self,obs):
		"""
		:param obs: (84 x 84) numpy array
		:return: int, pong observation dam value
		"""
		print(obs.shape)
		obs = obs.astype(int)
		ball_obs = obs[14:-7, 11:-11]
		right_bar_obs = obs[14:-7, -11:]
		right_bar = 117
		empty = 87

		try:
			if not np.all(ball_obs == empty):
				shape_ball = np.argwhere(ball_obs != empty)
			else:  # ball already passed left or right bar
				return np.inf
			pos_ball = shape_ball[-1]
			pos_ball_yx = [pos_ball[0], pos_ball[1]]
			shape_right_bar = np.argwhere(right_bar_obs == right_bar)
			pos_right_bar = shape_right_bar[len(shape_right_bar) // 2]
			pos_right_bar_yx = [pos_right_bar[0], pos_right_bar[1]]
			dam = abs(pos_ball_yx[0] - pos_right_bar_yx[0])
			return dam
		except:  # error
			return 0

