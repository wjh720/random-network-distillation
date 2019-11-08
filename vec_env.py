from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
from baselines import logger
from utils import tile_images
from Curiosity import Dec, Pushball_Dec, C_points, Cen, Appro_C_points, Pushball_Appro_C_points, \
	Island_Dec, Island_Appro_C_points, Island_Cen, Island_VI_Appro_C_points, Test_Island_Appro_C_points, x_Island_Cen, \
	Pushball_Cen
import pickle
import matplotlib.pyplot as plt
import copy


class AlreadySteppingError(Exception):
	"""
	Raised when an asynchronous step is running while
	step_async() is called again.
	"""

	def __init__(self):
		msg = 'already running an async step'
		Exception.__init__(self, msg)


class NotSteppingError(Exception):
	"""
	Raised when an asynchronous step is not running but
	step_wait() is called.
	"""

	def __init__(self):
		msg = 'not running an async step'
		Exception.__init__(self, msg)


class VecEnv(ABC):
	"""
	An abstract asynchronous, vectorized environment.
	"""

	def __init__(self, num_envs, observation_space, action_space):
		self.num_envs = num_envs
		self.observation_space = observation_space
		self.action_space = action_space

	@abstractmethod
	def reset(self):
		"""
		Reset all the environments and return an array of
		observations, or a tuple of observation arrays.

		If step_async is still doing work, that work will
		be cancelled and step_wait() should not be called
		until step_async() is invoked again.
		"""
		pass

	@abstractmethod
	def step_async(self, actions):
		"""
		Tell all the environments to start taking a step
		with the given actions.
		Call step_wait() to get the results of the step.

		You should not call this if a step_async run is
		already pending.
		"""
		pass

	@abstractmethod
	def step_wait(self):
		"""
		Wait for the step taken with step_async().

		Returns (obs, rews, dones, infos):
		 - obs: an array of observations, or a tuple of
				arrays of observations.
		 - rews: an array of rewards
		 - dones: an array of "episode done" booleans
		 - infos: a sequence of info objects
		"""
		pass

	@abstractmethod
	def close(self):
		"""
		Clean up the environments' resources.
		"""
		pass

	def step(self, actions):
		self.step_async(actions)
		return self.step_wait()

	def render(self, mode='human'):
		logger.warn('Render not defined for %s' % self)

	@property
	def unwrapped(self):
		if isinstance(self, VecEnvWrapper):
			return self.venv.unwrapped
		else:
			return self


class VecEnvWrapper(VecEnv):
	def __init__(self, venv, observation_space=None, action_space=None):
		self.venv = venv
		VecEnv.__init__(self,
						num_envs=venv.num_envs,
						observation_space=observation_space or venv.observation_space,
						action_space=action_space or venv.action_space)

	def step_async(self, actions):
		self.venv.step_async(actions)

	@abstractmethod
	def reset(self):
		pass

	@abstractmethod
	def step_wait(self):
		pass

	def close(self):
		return self.venv.close()

	def render(self):
		self.venv.render()


class CloudpickleWrapper(object):
	"""
	Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
	"""

	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		import cloudpickle
		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		import pickle
		self.x = pickle.loads(ob)


import numpy as np
from gym import spaces


class VecFrameStack(VecEnvWrapper):
	"""
	Vectorized environment base class
	"""

	def __init__(self, venv, nstack):
		self.venv = venv
		self.nstack = nstack
		wos = venv.observation_space  # wrapped ob space
		low = np.repeat(wos.low, self.nstack, axis=-1)
		high = np.repeat(wos.high, self.nstack, axis=-1)
		self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
		observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
		VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

	def step_wait(self):
		obs, rews, news, infos = self.venv.step_wait()
		self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
		for (i, new) in enumerate(news):
			if new:
				self.stackedobs[i] = 0
		self.stackedobs[..., -obs.shape[-1]:] = obs
		return self.stackedobs, rews, news, infos

	def reset(self):
		"""
		Reset all environments
		"""
		obs = self.venv.reset()
		self.stackedobs[...] = 0
		self.stackedobs[..., -obs.shape[-1]:] = obs
		return self.stackedobs

	def close(self):
		self.venv.close()


def worker(remote, parent_remote, env_fn_wrapper):
	parent_remote.close()
	env = env_fn_wrapper.x()
	while True:
		cmd, data = remote.recv()
		if cmd == 'step':
			ob, reward, done, info = env.step(data)
			if done:
				ob = env.reset()
			remote.send((ob, reward, done, info))
		elif cmd == 'reset':
			ob = env.reset()
			remote.send(ob)
		elif cmd == 'render':
			remote.send(env.render(mode='rgb_array'))
		elif cmd == 'close':
			remote.close()
			break
		elif cmd == 'get_spaces':
			remote.send((env.observation_space, env.action_space))
		else:
			raise NotImplementedError


class SubprocVecEnv(VecEnv):
	def __init__(self, env_fns, spaces=None):
		"""
		envs: list of gym environments to run in subprocesses
		"""
		self.waiting = False
		self.closed = False
		nenvs = len(env_fns)
		self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
		self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
				   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
		for p in self.ps:
			p.daemon = True  # if the main process crashes, we should not cause things to hang
			p.start()
		for remote in self.work_remotes:
			remote.close()

		self.remotes[0].send(('get_spaces', None))
		observation_space, action_space = self.remotes[0].recv()
		VecEnv.__init__(self, len(env_fns), observation_space, action_space)

	def step_async(self, actions):
		for remote, action in zip(self.remotes, actions):
			remote.send(('step', action))
		self.waiting = True

	def step_wait(self):
		results = [remote.recv() for remote in self.remotes]
		self.waiting = False
		obs, rews, dones, infos = zip(*results)
		return np.stack(obs), np.stack(rews), np.stack(dones), infos

	def reset(self):
		for remote in self.remotes:
			remote.send(('reset', None))
		return np.stack([remote.recv() for remote in self.remotes])

	def reset_task(self):
		for remote in self.remotes:
			remote.send(('reset_task', None))
		return np.stack([remote.recv() for remote in self.remotes])

	def close(self):
		if self.closed:
			return
		if self.waiting:
			for remote in self.remotes:
				remote.recv()
		for remote in self.remotes:
			remote.send(('close', None))
		for p in self.ps:
			p.join()
		self.closed = True

	def render(self, mode='human'):
		for pipe in self.remotes:
			pipe.send(('render', None))
		imgs = [pipe.recv() for pipe in self.remotes]
		bigimg = tile_images(imgs)
		if mode == 'human':
			import cv2
			cv2.imshow('vecenv', bigimg[:, :, ::-1])
			cv2.waitKey(1)
		elif mode == 'rgb_array':
			return bigimg
		else:
			raise NotImplementedError


class SubprocVecEnv_Pass(SubprocVecEnv):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv.__init__(self, env_fns)
		# For debugging
		self.hot_map_save_path = args.save_path
		self.ext_rewards = []
		self.args = args
		self.dec = Dec(self.args.size, self.args.n_agent, self.args)
		self.cen = Cen(self.args.size, self.args.n_agent, self.args)
		self.key_points = Appro_C_points(self.args.size, self.args.n_action, self.args, is_print=True)
		self.pre_state_n = [None for i in range(self.num_envs)]
		self.e_step = 0

	def dec_curiosity(self, state, i):
		return self.dec.output(state, i)

	def coor_curiosity(self, data_1, data_2, i):
		return self.key_points.output(data_1, data_2, i)

	def ma_reshape(self, obs, i):
		obs = np.array(obs)
		index = [i for i in range(len(obs.shape))]
		index[0] = i
		index[i] = 0
		return np.transpose(obs, index).copy()

	def smooth(self, data):

		smoothed = []

		for i in range(len(data)):
			smoothed.append(np.mean(data[max(0, i + 1 - 100): i + 1]))

		return smoothed

	def save_results(self, er2, name, arg):
		filename = '%s/%s_data.pkl' % (arg.save_path, name)
		with open(filename, 'wb') as file:
			pickle.dump(er2, file)

		# er3 = q_cen_for_pass()

		# er1_s = smooth(er1)
		er2_s = self.smooth(er2)
		# er3_s = smooth(er3)

		m_figure = plt.figure()
		m_ax1 = m_figure.add_subplot(3, 2, 1)
		m_ax2 = m_figure.add_subplot(3, 2, 2)
		m_ax3 = m_figure.add_subplot(3, 2, 3)
		m_ax4 = m_figure.add_subplot(3, 2, 4)
		m_ax5 = m_figure.add_subplot(3, 2, 5)
		m_ax6 = m_figure.add_subplot(3, 2, 6)

		# m_ax1.plot(er1_s)
		# m_ax2.plot(er1)
		m_ax3.plot(er2_s)
		m_ax4.plot(er2)
		# m_ax5.plot(er3_s)
		# m_ax6.plot(er3)

		m_ax1.legend(['epsilon-greedy'])
		m_ax2.legend(['epsilon-greedy (unsmoothed)'])
		m_ax3.legend(['dec-curiosity'])
		m_ax4.legend(['dec-curiosity (unsmoothed)'])
		m_ax5.legend(['cen-curiosity'])
		m_ax6.legend(['cen-curiosity (unsmoothed)'])

		m_figure.savefig('%s/%s_i Boltzmann.png' % (arg.save_path, name))
		plt.close(m_figure)

	def reset(self):
		for remote in self.remotes:
			remote.send(('reset', None))
		obs_n = _flatten_obs([remote.recv() for remote in self.remotes])
		return obs_n

	def step(self, actions):
		self.step_async(actions)
		obs, ext_rewards, dones, infos = self.step_wait()
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]

		# estimate coor
		for j in range(self.num_envs):
			if self.pre_state_n[j] != None:
				self.key_points.update([self.pre_state_n[j], actions[j]], [state_n[j], None])

		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		coor_rewards = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		cen_rewards = ext_rewards.copy()
		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				if self.pre_state_n[j] != None:
					coor_rewards[i, j] = self.coor_curiosity([self.pre_state_n[j], actions[j]], [state_n[j], None], i)
				else:
					coor_rewards[i, j] = 0.
				cen_rewards[i, j] = self.args.gamma_cen * self.cen.output(state_n[j])
				penalty_rewards[i, j] = self.args.penalty
		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j], infos[j]['door'])
			self.cen.update(state_n[j])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):
			if dones[i]:
				self.e_step += 1
				self.ext_rewards.append(ext_rewards[0, i])

				s_rate = self.args.t_save_rate
				if (self.e_step + 1) % (1000 * s_rate) == 0:
					print(self.e_step + 1, float(sum(self.ext_rewards[-1000 * s_rate:])) / (1000.0 * s_rate))
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs, ext_rewards, dones, infos


class SubprocVecEnv_Island(SubprocVecEnv_Pass):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv_Pass.__init__(self, env_fns, args)
		# For debugging
		self.dec = Island_Dec(self.args.size, self.args.island_agent_max_power,
							  self.args.n_agent, self.args)
		self.cen = Island_Cen(self.args.size, self.args.island_agent_max_power, self.args.n_agent, self.args)
		self.key_points = Island_Appro_C_points(self.args.size, self.args.n_action,
												self.args.island_agent_max_power,
												self.args.island_wolf_max_power,
												self.args, is_print=True)
		self.ext_rewards_list = [[] for _ in range(self.args.n_agent)]
		self.num_kill = []
		self.time_length = [[] for _ in range(self.args.n_agent)]
		self.death = [[] for _ in range(self.args.n_agent)]
		self.landmark = []

	def step(self, actions):
		self.step_async(actions)
		obs, ext_rewards, dones, infos = self.step_wait()
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]

		# estimate coor
		for j in range(self.num_envs):
			if self.pre_state_n[j] != None:
				self.key_points.update([self.pre_state_n[j], actions[j]], [state_n[j], None])

		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		coor_rewards = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		cen_rewards = ext_rewards.copy()
		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				if self.pre_state_n[j] != None:
					coor_rewards[i, j] = self.coor_curiosity([self.pre_state_n[j], actions[j]], [state_n[j], None], i)
				else:
					coor_rewards[i, j] = 0.
				cen_rewards[i, j] = self.args.gamma_cen * self.cen.output(state_n[j])
				penalty_rewards[i, j] = self.args.penalty
		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j])
			self.cen.update(state_n[j])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):

			info_r = infos[i]['rew']
			self.num_kill.append(info_r['kill'])
			self.landmark.append(info_r['landmark'])
			for j, death in enumerate(info_r['death']):
				self.ext_rewards_list[j].append(ext_rewards[j, i])
				self.death[j].append(int(death))
				self.time_length[j].append(info_r['time_length'][j])

			if dones[i]:
				self.e_step += 1

				s_rate = self.args.t_save_rate
				if (self.e_step + 1) % (1000 * s_rate) == 0:

					print(self.e_step + 1)
					for i in range(self.args.n_agent):
						print('agent_%d : ' % i,
							  'ext', round(float(sum(self.ext_rewards_list[i])) / (1000.0 * s_rate), 2),
							  'death', round(float(sum(self.death[i])) / (1000.0 * s_rate), 2),
							  'time_length', round(sum(self.time_length[i]) / (1000.0 * s_rate), 2))
					print('kill', float(sum(self.num_kill)) / (1000.0 * s_rate),
						  'landmark', float(sum(self.landmark)) / (1000.0 * s_rate))

					self.ext_rewards_list = [[] for _ in range(self.args.n_agent)]
					self.num_kill = []
					self.time_length = [[] for _ in range(self.args.n_agent)]
					self.death = [[] for _ in range(self.args.n_agent)]
					self.landmark = []
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs, ext_rewards, dones, infos


class SubprocVecEnv_x_Island(SubprocVecEnv_Pass):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv_Pass.__init__(self, env_fns, args)
		# For debugging
		self.dec = Island_Dec(self.args.size, self.args.x_island_agent_max_power,
							  self.args.n_agent, self.args)
		self.cen = x_Island_Cen(self.args.size, self.args.x_island_agent_max_power, self.args.n_agent, self.args)
		'''
		self.test_key_points = Test_Island_Appro_C_points(self.args.size, self.args.n_action,
														  self.args.x_island_agent_max_power,
														  self.args.x_island_wolf_max_power,
														  self.args, is_print=True)
		'''
		self.key_points = Island_VI_Appro_C_points(self.args.size, self.args.n_action,
												   self.args.x_island_agent_max_power,
												   self.args.x_island_wolf_max_power,
												   self.args, is_print=True)
		self.n_agent = self.args.n_agent
		self.ext_rewards_list = [[] for _ in range(self.args.n_agent)]
		self.num_kill = []
		self.time_length = [[] for _ in range(self.args.n_agent)]
		self.death = [[] for _ in range(self.args.n_agent)]
		self.landmark = []
		'''
		self.error_coor = [[] for _ in range(self.args.n_agent)]
		self.error_coor_p = [[] for _ in range(self.args.n_agent)]
		self.error_coor_t = [[] for _ in range(self.args.n_agent)]
		'''

	def step(self, actions):
		self.step_async(actions)
		obs, ext_rewards, dones, infos = self.step_wait()
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]
		'''
		# test estimate coor
		for j in range(self.num_envs):
			if self.pre_state_n[j] != None:
				self.test_key_points.update([self.pre_state_n[j], actions[j]], [state_n[j], None])
		'''
		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		cen_rewards = ext_rewards.copy()
		'''
		test_coor_rewards = ext_rewards.copy()
		test_coor_p = ext_rewards.copy()
		test_coor_t = ext_rewards.copy()
		'''

		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				cen_rewards[i, j] = self.args.gamma_cen * self.cen.output(state_n[j], i)
				penalty_rewards[i, j] = self.args.penalty
				'''
				if self.pre_state_n[j] != None:
					test_coor_rewards[i, j], test_coor_p[i, j], test_coor_t[i, j] = \
						self.test_key_points.output([self.pre_state_n[j], actions[j]], [state_n[j], None], i)
				else:
					test_coor_rewards[i, j] = 0.
					test_coor_p[i, j] = 0
					test_coor_t[i, j] = 0
				'''

		coor_rewards = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')
		coor_p = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')
		coor_t = np.zeros((self.n_agent, self.n_agent, self.num_envs), dtype='float32')

		if not self.key_points.not_run:

			C_label = []
			C_x_p = []
			C_x_t = []
			for j in range(self.num_envs):
				if self.pre_state_n[j] != None:
					label, x_p, x_t = self.key_points.make([self.pre_state_n[j], actions[j]], [state_n[j], None])
					C_label.append(label)
					C_x_p.append(x_p)
					C_x_t.append(x_t)

			num_data = len(C_label)

			if num_data > 0:

				C_label = np.transpose(np.array(C_label), (1, 0))
				C_x_p = np.transpose(np.array(C_x_p), (1, 0, 2))
				C_x_t = np.transpose(np.array(C_x_t), (1, 0, 2))
				coor_output, t_coor_p, t_coor_t = self.key_points.output(C_label, C_x_p, C_x_t)

				tt_stamp = 0
				for k in range(self.num_envs):
					if self.pre_state_n[k] != None:
						coor_rewards[:, :, k] = coor_output[:, :, tt_stamp]
						coor_p[:, :, k] = t_coor_p[:, :, tt_stamp]
						coor_t[:, :, k] = t_coor_t[:, :, tt_stamp]
						tt_stamp += 1
				'''
				for i in range(self.n_agent):
					for j in range(self.n_agent):
						if i != j:
							self.error_coor[j].append(np.mean(np.absolute(test_coor_rewards[j] - coor_rewards[j, i])))
							self.error_coor_p[j].append(np.mean(np.absolute(test_coor_p[j] - coor_p[j, i])))
							self.error_coor_t[j].append(np.mean(np.absolute(test_coor_t[j] - coor_t[j, i])))
				'''

		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j])
			self.cen.update(state_n[j])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):

			info_r = infos[i]['rew']
			self.num_kill.append(info_r['kill'])
			self.landmark.append(info_r['landmark'])
			for j, death in enumerate(info_r['death']):
				self.ext_rewards_list[j].append(ext_rewards[j, i])
				self.death[j].append(int(death))
				self.time_length[j].append(info_r['time_length'][j])

			if dones[i]:
				self.e_step += 1

				s_rate = self.args.t_save_rate
				if (self.e_step + 1) % (1000 * s_rate) == 0:

					print(self.e_step + 1)
					for i in range(self.args.n_agent):
						print('agent_%d : ' % i,
							  'ext', round(float(sum(self.ext_rewards_list[i])) / (1000.0 * s_rate), 2),
							  'death', round(float(sum(self.death[i])) / (1000.0 * s_rate), 2),
							  'time_length', round(sum(self.time_length[i]) / (1000.0 * s_rate), 2))
						'''
							  'error_coor', round(sum(self.error_coor[i]) / (1000.0 * s_rate), 2),
							  'error_coor_p', round(sum(self.error_coor_p[i]) / (1000.0 * s_rate), 2),
							  'error_coor_t', round(sum(self.error_coor_t[i]) / (1000.0 * s_rate), 2))
							  '''
					print('kill', float(sum(self.num_kill)) / (1000.0 * s_rate),
						  'landmark', float(sum(self.landmark)) / (1000.0 * s_rate))

					self.ext_rewards_list = [[] for _ in range(self.args.n_agent)]
					self.num_kill = []
					self.time_length = [[] for _ in range(self.args.n_agent)]
					self.death = [[] for _ in range(self.args.n_agent)]
					self.landmark = []
					'''
					self.error_coor = [[] for _ in range(self.args.n_agent)]
					self.error_coor_p = [[] for _ in range(self.args.n_agent)]
					self.error_coor_t = [[] for _ in range(self.args.n_agent)]
					'''
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs, (ext_rewards, dec_rewards, coor_rewards, penalty_rewards, cen_rewards), dones, infos


class SubprocVecEnv_PushBall(SubprocVecEnv_Pass):
	"""
	VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
	Recommended to use when num_envs > 1 and step() can be a bottleneck.
	"""

	def __init__(self, env_fns, args):
		SubprocVecEnv_Pass.__init__(self, env_fns, args)
		# For debugging
		self.dec = Pushball_Dec(self.args.size, self.args.n_agent, self.args)
		self.cen = Pushball_Cen(self.args.size, self.args.n_agent, self.args)
		self.key_points = Pushball_Appro_C_points(self.args.size, self.args.n_action, self.args, is_print=True)
		self.win_rewards = [[] for _ in range(4)]

	def step(self, actions):
		self.step_async(actions)
		obs, ext_rewards, dones, infos = self.step_wait()
		ext_rewards = self.ma_reshape(ext_rewards, 1).astype('float32')
		state_n = []
		for i in range(self.num_envs):
			state_n.append(copy.deepcopy(infos[i]['state']))
			infos[i]['pre_state'] = self.pre_state_n[i]

		# estimate coor
		for j in range(self.num_envs):
			if self.pre_state_n[j] != None:
				self.key_points.update([self.pre_state_n[j], actions[j]], [state_n[j], None])

		# add intrinsic rew
		dec_rewards = ext_rewards.copy()
		coor_rewards = ext_rewards.copy()
		penalty_rewards = ext_rewards.copy()
		cen_rewards = ext_rewards.copy()
		for i in range(self.args.n_agent):
			for j in range(self.num_envs):
				dec_rewards[i, j] = self.args.gamma_dec * self.dec_curiosity(state_n[j], i)
				if self.pre_state_n[j] != None:
					coor_rewards[i, j] = self.coor_curiosity([self.pre_state_n[j], actions[j]], [state_n[j], None], i)
				else:
					coor_rewards[i, j] = 0.
				cen_rewards[i, j] = self.args.gamma_cen * self.cen.output(state_n[j])
				penalty_rewards[i, j] = self.args.penalty
		# update intrinsic rew
		for j in range(self.num_envs):
			self.dec.update(state_n[j])
			self.cen.update(state_n[j])

		self.pre_state_n = state_n
		for i in range(self.num_envs):
			if dones[i]:
				self.pre_state_n[i] = None

		# debug
		for i in range(self.num_envs):

			info_r = infos[i]['rew']
			for i in range(4):
				self.win_rewards[i].append(info_r[i])

			if dones[i]:
				self.e_step += 1

				s_rate = self.args.t_save_rate
				if (self.e_step + 1) % (1000 * s_rate) == 0:
					print(self.e_step + 1,
						  'up', float(sum(self.win_rewards[0])) / (1000.0 * s_rate),
						  'left', float(sum(self.win_rewards[1])) / (1000.0 * s_rate),
						  'down', float(sum(self.win_rewards[2])) / (1000.0 * s_rate),
						  'right', float(sum(self.win_rewards[3])) / (1000.0 * s_rate))
					self.win_rewards = [[] for _ in range(4)]
					self.dec.show(self.hot_map_save_path, self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.key_points.show(self.e_step + 1)

				if (self.e_step + 1) % (100000 * s_rate) == 0:
					self.save_results(self.ext_rewards, '%d' % (self.e_step + 1), self.args)

		return obs, (ext_rewards, dec_rewards, coor_rewards, penalty_rewards, cen_rewards), dones, infos


def _flatten_obs(obs):
	assert isinstance(obs, (list, tuple))
	assert len(obs) > 0

	if isinstance(obs[0], dict):
		keys = obs[0].keys()
		return {k: np.stack([o[k] for o in obs]) for k in keys}
	else:
		return np.stack(obs)
