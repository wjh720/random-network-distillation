"""
Helpers for scripts like run_atari.py.
"""

import os

import gym
from gym.wrappers import FlattenDictWrapper
from mpi4py import MPI
from baselines import logger
from monitor import Monitor
from atari_wrappers import make_atari, wrap_deepmind
from vec_env import SubprocVecEnv
from pass_environment import Pass
from island_environment import Island
from pushball_environment import PushBall
from x_island_environment import x_Island


def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, max_episode_steps=4500):
	"""
	Create a wrapped, monitored SubprocVecEnv for Atari.
	"""
	if wrapper_kwargs is None: wrapper_kwargs = {}

	def make_env(rank):  # pylint: disable=C0111
		def _thunk():
			env = make_atari(env_id, max_episode_steps=max_episode_steps)
			env.seed(seed + rank)
			env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
			return wrap_deepmind(env, **wrapper_kwargs)

		return _thunk

	# set_global_seeds(seed)
	return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def make_pass_env(env_id, env_type, num_env, seed, args, subrank=0, wrapper_kwargs=None, start_index=0,
				  reward_scale=1.0):
	env = Pass(args, subrank)
	env.initialization(args)

	return env


def make_multi_pass_env(env_id, env_type, num_env, seed, args, wrapper_kwargs=None, start_index=0, reward_scale=1.0,
						flatten_dict_observations=True,
						gamestate=None):
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_pass_env(
			env_id=env_id,
			env_type=env_type,
			subrank=rank,
			num_env=1,
			seed=seed,
			args=args,
			reward_scale=reward_scale,
			wrapper_kwargs=wrapper_kwargs
		)

	return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)], args)


def make_pushball_env(env_id, env_type, num_env, seed, args, subrank=0, wrapper_kwargs=None, start_index=0,
					  reward_scale=1.0):
	env = PushBall(args, subrank)
	env.initialization(args)
	return env


def make_m_pushball_env(env_id, env_type, num_env, seed, args, wrapper_kwargs=None, start_index=0, reward_scale=1.0,
						flatten_dict_observations=True,
						gamestate=None):
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_pushball_env(
			env_id=env_id,
			env_type=env_type,
			subrank=rank,
			num_env=1,
			seed=seed,
			args=args,
			reward_scale=reward_scale,
			wrapper_kwargs=wrapper_kwargs
		)

	return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)], args)


def make_x_island_env(env_id, env_type, num_env, seed, args, subrank=0, wrapper_kwargs=None, start_index=0,
					reward_scale=1.0):
	#env = test_Island(args, subrank)
	env = x_Island(args, subrank)
	env.initialization(args)
	return env


def make_m_x_island_env(env_id, env_type, num_env, seed, args, wrapper_kwargs=None, start_index=0, reward_scale=1.0,
					  flatten_dict_observations=True,
					  gamestate=None):
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_x_island_env(
			env_id=env_id,
			env_type=env_type,
			subrank=rank,
			num_env=1,
			seed=seed,
			args=args,
			reward_scale=reward_scale,
			wrapper_kwargs=wrapper_kwargs
		)

	return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)], args)


def make_island_env(env_id, env_type, num_env, seed, args, subrank=0, wrapper_kwargs=None, start_index=0,
					reward_scale=1.0):
	env = Island(args, subrank)
	env.initialization(args)

	return env


def make_m_island_env(env_id, env_type, num_env, seed, args, wrapper_kwargs=None, start_index=0, reward_scale=1.0,
					  flatten_dict_observations=True,
					  gamestate=None):
	wrapper_kwargs = wrapper_kwargs or {}
	mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
	seed = seed + 10000 * mpi_rank if seed is not None else None
	logger_dir = logger.get_dir()

	def make_thunk(rank):
		return lambda: make_island_env(
			env_id=env_id,
			env_type=env_type,
			subrank=rank,
			num_env=1,
			seed=seed,
			args=args,
			reward_scale=reward_scale,
			wrapper_kwargs=wrapper_kwargs
		)

	return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)], args)


def arg_parser():
	"""
	Create an empty argparse.ArgumentParser.
	"""
	import argparse
	return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def atari_arg_parser():
	"""
	Create an argparse.ArgumentParser for run_atari.py.
	"""
	parser = arg_parser()
	parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
	parser.add_argument('--seed', help='RNG seed', type=int, default=0)
	parser.add_argument('--num-timesteps', type=int, default=int(10e6))
	return parser
