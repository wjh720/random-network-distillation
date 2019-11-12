import gym
import matplotlib.pyplot as plt
import numpy as np
import copy
import random


class Visualization:

	def __init__(self, size, n_agent, args):
		self.size = size
		self.n_agent = n_agent
		self.args = args
		self.visited = [np.zeros([args.size, args.size, 2]) for _ in range(self.n_agent)]
		self.visited_old = [np.zeros([args.size, args.size, 2]) for _ in range(self.n_agent)]

	def update(self, state, is_door_open):
		# For heat map
		for i in range(self.n_agent):
			self.visited[i][state[i][0]][state[i][1]][int(is_door_open)] += 1

	def show(self, path, e):
		figure = plt.figure(figsize=(16, 10))

		ax1 = figure.add_subplot(2, 6, 1)
		ax2 = figure.add_subplot(2, 6, 2)
		ax3 = figure.add_subplot(2, 6, 3)
		ax4 = figure.add_subplot(2, 6, 4)
		ax5 = figure.add_subplot(2, 6, 5)
		ax6 = figure.add_subplot(2, 6, 6)
		ax7 = figure.add_subplot(2, 6, 7)
		ax8 = figure.add_subplot(2, 6, 8)
		ax9 = figure.add_subplot(2, 6, 9)
		ax10 = figure.add_subplot(2, 6, 10)
		ax11 = figure.add_subplot(2, 6, 11)
		ax12 = figure.add_subplot(2, 6, 12)

		ax1.imshow(np.log(self.visited[0][:, :, 0] + 1))
		ax2.imshow(np.log(self.visited[0][:, :, 0] - self.visited_old[0][:, :, 0] + 1))
		ax3.imshow(np.log(self.visited[0][:, :, 1] + 1))
		ax4.imshow(np.log((self.visited[0][:, :, 1] - self.visited_old[0][:, :, 1] + 1)))
		ax5.imshow(np.log(np.sum(self.visited[0], axis=2) + 1))
		ax6.imshow(np.log(np.sum(self.visited[0], axis=2) - np.sum(self.visited_old[0], axis=2) + 1))

		ax7.imshow(np.log(self.visited[1][:, :, 0] + 1))
		ax8.imshow(np.log(self.visited[1][:, :, 0] - self.visited_old[1][:, :, 0] + 1))
		ax9.imshow(np.log(self.visited[1][:, :, 1] + 1))
		ax10.imshow(np.log((self.visited[1][:, :, 1] - self.visited_old[1][:, :, 1] + 1)))
		ax11.imshow(np.log(np.sum(self.visited[1], axis=2) + 1))
		ax12.imshow(np.log(np.sum(self.visited[1], axis=2) - np.sum(self.visited_old[1], axis=2) + 1))

		figure.savefig('%s/%i.png' % (path, e))
		plt.close(figure)

		self.visited_old = [v.copy() for v in self.visited]


class ThreePass:
	def __init__(self, args, rank):
		self.args = args
		self.rank = rank
		self.initialization(args)

	def get_discrete(self, shape):
		size = 1
		for item in shape:
			size *= item
		return gym.spaces.Discrete(size)

	def get_action_n(self, action_n):
		action_n = int(action_n)
		res = []
		for item in self.action_space_x:
			res.append(action_n % item)
			action_n //= item
		return res

	def initialization(self, args):

		self.seed = random.randint(0, 9999)
		np.random.seed(self.seed)

		self.is_print = self.rank == 0

		self.args = args
		self.size = args.size
		self.map = np.zeros([self.size, self.size])
		self.dec_int = args.gamma_dec != 0
		self.penalty = args.penalty

		if self.is_print:
			print(args.save_path)
			print('>>>>>>>>>>>>>dec_int', self.dec_int)

		self.map[:, self.size // 2] = -1
		self.map[self.size // 3, self.size // 2:] = -1
		self.map[self.size // 3 * 2, self.size // 2:] = -1

		# Left landmark
		self.map[int(self.size * 0.8), int(self.size * 0.2)] = 1

		# Right landmarks
		self.map[int(self.size / 6), int(self.size * 0.8)] = 1
		self.map[int(self.size / 2), int(self.size * 0.8)] = 1
		self.map[int(self.size / 6 * 5), int(self.size * 0.8)] = 1

		self.landmarks = np.array([[int(self.size * 0.8), int(self.size * 0.2)],
		                           [int(self.size / 6), int(self.size * 0.8)],
		                           [int(self.size / 2), int(self.size * 0.8)],
		                           [int(self.size / 6 * 5), int(self.size * 0.8)]])

		self.door_open_interval = args.doi

		self.door_open_n = [False, False, False]
		self.door_open_step_count_n = [0, 0, 0]
		self.door_position_n = [int(self.size / 6 * (m_di * 2 + 1)) for m_di in range(3)]

		self.n_agent = 2
		self.n_action = 4
		self.n_dim = 2

		self.state_n = [np.array([0, 0]) for _ in range(self.n_agent)]

		self.eye = np.eye(self.size)
		self.flag = np.eye(2)

		# Used by OpenAI baselines
		self.action_space_x = [self.n_action, self.n_action]
		self.action_space = self.get_discrete(self.action_space_x)
		self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[args.size * 4])
		self.num_envs = args.num_env
		self.metadata = {'render.modes': []}
		self.reward_range = (-100., 2000.)
		self.spec = 2

		self.t_step = 0

		# Visualization
		self.is_print = random.randint(0, 16) == 0
		if self.is_print:
			self.e_step = 0
			self.hot_map_save_path = self.args.save_path
			self.heat_map = Visualization(self.args.size, self.args.n_agent, self.args)

	def step(self, action_n, obs_a=False, obs_b=False, obs_c=False, obs_d=False):

		action_n = self.get_action_n(action_n)

		self.t_step += 1
		for i, action in enumerate(action_n):
			new_row = -1
			new_column = -1

			if action == 0:
				new_row = max(self.state_n[i][0] - 1, 0)
				new_column = self.state_n[i][1]
			elif action == 1:
				new_row = self.state_n[i][0]
				new_column = min(self.state_n[i][1] + 1, self.size - 1)
			elif action == 2:
				new_row = min(self.state_n[i][0] + 1, self.size - 1)
				new_column = self.state_n[i][1]
			elif action == 3:
				new_row = self.state_n[i][0]
				new_column = max(self.state_n[i][1] - 1, 0)

			if self.map[new_row][new_column] != -1:
				self.state_n[i] = np.array([new_row, new_column])

		for m_di in range(3):
			if self.door_open_n[m_di]:
				if self.door_open_step_count_n[m_di] >= self.door_open_interval:
					# print('>>>>>> Door Closed')
					self.door_open_n[m_di] = False
					# self.map[int(self.size * 0.45):int(self.size * 0.55), self.size // 2] = -1
					self.map[self.door_position_n[m_di] - 1:self.door_position_n[m_di] + 1, self.size // 2] = -1
					self.door_open_step_count_n[m_di] = 0
				else:
					self.door_open_step_count_n[m_di] += 1

		for m_di in range(3):
			if not self.door_open_n[m_di]:
				for landmark_id, landmark in enumerate(self.landmarks[[0, m_di + 1]]):
					for i, state in enumerate(self.state_n):
						if (landmark == state).all():
							# print('>>>>>>', i, 'Open the door.')
							self.door_open_n[m_di] = True
							self.map[self.door_position_n[m_di] - 1:self.door_position_n[m_di] + 1, self.size // 2] = 0
							self.door_open_step_count_n[m_di] = 0
							break

		# if obs_d:
		#     return self.observations_d()

		info = {'door': self.door_open_n, 'state': copy.deepcopy(self.state_n)}

		pre_t_step = self.t_step

		return_obs = self.obs_n()
		return_rew = self.reward()
		return_done = self.done()

		if return_done:
			info['episode'] = {'r': return_rew[0], 'l': pre_t_step}

		if self.is_print:
			self.heat_map.update(info['state'], np.array(info['door']).any())
			if return_done:
				self.e_step += 1
				if (self.e_step + 1) % 100 == 0:
					self.heat_map.show(self.hot_map_save_path, self.e_step + 1)

		return return_obs[0], return_rew[0], return_done, info

	def fix_reset(self):
		self.t_step = 0
		self.door_open_n = [False, False, False]
		self.door_open_step_count_n = [0, 0, 0]

		self.state_n = [np.array([0, 0]) for _ in range(self.n_agent)]

		self.map[:, self.size // 2] = -1

		return self.obs_n()

	def reset(self, obs_d=False):
		self.t_step = 0
		self.door_open_n = [False, False, False]
		self.door_open_step_count_n = [0, 0, 0]

		if self.args.fix_start:
			self.state_n = [np.array([0, 0]) for _ in range(self.n_agent)]
		else:
			for i in range(self.n_agent):
				self.state_n[i][1] = np.random.randint(self.size // 2)
				self.state_n[i][0] = np.random.randint(self.size)

		self.map[:, self.size // 2] = -1

		# if obs_d:
		#     return self.observations_d()

		return self.obs_n()[0]

	def random_reset(self, obs_d=False):
		self.t_step = 0
		self.door_open_n = [False, False, False]
		self.door_open_step_count_n = [0, 0, 0]

		for i in range(self.n_agent):
			self.state_n[i][1] = np.random.randint(self.size // 2)
			self.state_n[i][0] = np.random.randint(self.size)

		self.map[:, self.size // 2] = -1

		# if obs_d:
		#     return self.observations_d()

		return self.obs_n()

	# def local_state(self, i):
	#     return self.state_n[i]
	#
	# def local_states(self):
	#     return self.state_n
	#
	# def observation_a(self, i):
	#     return np.concatenate([self.state_n[i], np.array([int(self.door_open)])])
	#
	# def observations_a(self):
	#     return [self.observation_a(i) for i in range(self.n_agent)]
	#
	# def observation_b(self, i):
	#     return np.concatenate(self.state_n + [np.array([int(self.door_open)])])
	#
	# def observations_b(self):
	#     return [self.observation_b(i) for i in range(self.n_agent)]
	#
	# def observation(self, i):
	#     same_room = 0
	#     if self.state_n[i][1] < self.size // 2 and self.state_n[1 - i][1] < self.size // 2:
	#         same_room = 1
	#
	#     if self.state_n[i][1] >= self.size // 2 and self.state_n[1 - i][1] >= self.size // 2:
	#         same_room = 1
	#
	#     # indicator = same_room * 2 + self.door_open
	#
	#     return np.concatenate([self.state_n[i], np.array([int(same_room), int(self.door_open)])])
	#
	# def obs_c(self, i):
	#     same_room = 0
	#     if self.state_n[i][1] < self.size // 2 and self.state_n[1 - i][1] < self.size // 2:
	#         same_room = 1
	#
	#     if self.state_n[i][1] >= self.size // 2 and self.state_n[1 - i][1] >= self.size // 2:
	#         same_room = 1
	#
	#     # indicator = same_room * 2 + self.door_open
	#
	#     return np.concatenate([self.state_n[i], np.array([int(same_room), int(self.door_open)])])

	def obs_n(self):
		return [self.obs() for _ in range(self.n_agent)]

	# def observations_c(self):
	#     return [self.observation_c(i) for i in range(self.n_agent)]

	def obs(self):
		# same_room = 0
		# if self.state_n[i][1] < self.size // 2 and self.state_n[1 - i][1] < self.size // 2:
		#     same_room = 1
		#
		# if self.state_n[i][1] >= self.size // 2 and self.state_n[1 - i][1] >= self.size // 2:
		#     same_room = 1

		return np.concatenate([self.eye[self.state_n[0][0]],
		                       self.eye[self.state_n[0][1]],
		                       self.eye[self.state_n[1][0]],
		                       self.eye[self.state_n[1][1]]
		                       ]).copy()

	# self.flag[same_room],
	# self.flag[int(self.door_open)]]).copy()]

	# def observations_d(self):
	#     return [self.observation_d(i) for i in range(self.n_agent)]

	def reward(self):
		count = 0

		for i, state in enumerate(self.state_n):
			if state[1] > self.size // 2 and state[0] < self.size // 3:
				count += 1
		# print('>>>>>>', i, 'Pass.')

		return [(count >= 2) * 1000, (count >= 2) * 1000]

	def done(self):
		count = 0

		for state in self.state_n:
			if state[1] > self.size // 2 and state[0] < self.size // 3:
				count += 1

		if count >= 2 or self.t_step >= self.args.episode_length:
			self.reset()
			return 1

		return 0

	def close(self):
		self.reset()
