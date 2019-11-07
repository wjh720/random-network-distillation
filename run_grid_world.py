#!/usr/bin/env python3
import functools
import os

from baselines import logger
from mpi4py import MPI
import mpi_util
import tf_util
from cmd_util import make_atari_env, arg_parser
from policies.cnn_gru_policy_dynamics import CnnGruPolicy
from policies.cnn_policy_param_matched import CnnPolicy
from policies.fc_policy_param_matched import FcPolicy
from ppo_agent import PpoAgent
from utils import set_global_seeds
from vec_env import VecFrameStack


def train(*, env_id, num_env, hps, num_timesteps, seed):
	venv = VecFrameStack(
		make_atari_env(env_id, num_env, seed, wrapper_kwargs=dict(),
		               start_index=num_env * MPI.COMM_WORLD.Get_rank(),
		               max_episode_steps=hps.pop('max_episode_steps')),
		hps.pop('frame_stack'))
	# venv.score_multiple = {'Mario': 500,
	#                        'MontezumaRevengeNoFrameskip-v4': 100,
	#                        'GravitarNoFrameskip-v4': 250,
	#                        'PrivateEyeNoFrameskip-v4': 500,
	#                        'SolarisNoFrameskip-v4': None,
	#                        'VentureNoFrameskip-v4': 200,
	#                        'PitfallNoFrameskip-v4': 100,
	#                        }[env_id]
	venv.score_multiple = 1
	venv.record_obs = True if env_id == 'SolarisNoFrameskip-v4' else False
	ob_space = venv.observation_space
	ac_space = venv.action_space
	gamma = hps.pop('gamma')
	policy = {'rnn': CnnGruPolicy,
	          'cnn': CnnPolicy}[hps.pop('policy')]
	agent = PpoAgent(
		scope='ppo',
		ob_space=ob_space,
		ac_space=ac_space,
		stochpol_fn=functools.partial(
			policy,
			scope='pol',
			ob_space=ob_space,
			ac_space=ac_space,
			update_ob_stats_independently_per_gpu=hps.pop('update_ob_stats_independently_per_gpu'),
			proportion_of_exp_used_for_predictor_update=hps.pop('proportion_of_exp_used_for_predictor_update'),
			dynamics_bonus=hps.pop("dynamics_bonus")
		),
		gamma=gamma,
		gamma_ext=hps.pop('gamma_ext'),
		lam=hps.pop('lam'),
		nepochs=hps.pop('nepochs'),
		nminibatches=hps.pop('nminibatches'),
		lr=hps.pop('lr'),
		cliprange=0.1,
		nsteps=128,
		ent_coef=0.001,
		max_grad_norm=hps.pop('max_grad_norm'),
		use_news=hps.pop("use_news"),
		comm=MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None,
		update_ob_stats_every_step=hps.pop('update_ob_stats_every_step'),
		int_coeff=hps.pop('int_coeff'),
		ext_coeff=hps.pop('ext_coeff'),
	)
	agent.start_interaction([venv])
	if hps.pop('update_ob_stats_from_random_agent'):
		agent.collect_random_statistics(num_timesteps=128 * 50)
	assert len(hps) == 0, "Unused hyperparameters: %s" % list(hps.keys())

	counter = 0
	while True:
		info = agent.step()
		if info['update']:
			logger.logkvs(info['update'])
			logger.dumpkvs()
			counter += 1
		if agent.I.stats['tcount'] > num_timesteps:
			break

	agent.stop_interaction()


def add_env_params(parser):
	parser.add_argument('--env', help='environment ID', default='MontezumaRevengeNoFrameskip-v4')
	parser.add_argument('--seed', help='RNG seed', type=int, default=0)
	parser.add_argument('--max_episode_steps', type=int, default=4500)


def common_arg_parser():
	"""
	Create an argparse.ArgumentParser for run_mujoco.py.
	"""
	parser = arg_parser()
	parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
	parser.add_argument('--env_type',
	                    help='type of environment, used when the environment type cannot be automatically determined',
	                    type=str)
	parser.add_argument('--seed', help='RNG seed', type=int, default=None)
	parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
	parser.add_argument('--num_timesteps', type=float, default=1e6),
	parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
	parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
	parser.add_argument('--num_env',
	                    help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco',
	                    default=1, type=int)
	parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
	parser.add_argument('--save_path', help='Path to save trained model to',
	                    default='../../results/PPO/try_1/Random_start/', type=str)
	parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
	parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
	parser.add_argument('--play', default=False, action='store_true')
	parser.add_argument('--nsteps', default=2048, type=int)
	parser.add_argument('--size', default=30, type=int)
	parser.add_argument('--n_action', default=4, type=int)
	parser.add_argument('--n_agent', default=2, type=int)
	parser.add_argument('--episode_length', default=300, type=int)
	parser.add_argument('--doi', default=8, type=int, help='door_open_interval')
	parser.add_argument('--penalty', default=0.0, type=float)
	parser.add_argument('--gamma_dec', default=0.0, type=float)
	parser.add_argument('--gamma_cen', default=0.0, type=float)
	parser.add_argument('--fix_start', default=False, action='store_true')
	parser.add_argument('--gamma_coor_r', default=0.0, type=float)
	parser.add_argument('--gamma_coor_t', default=0.0, type=float)
	parser.add_argument('--gamma_coor_tv', default=0.0, type=float)
	parser.add_argument('--symmetry', default=False, action='store_true')
	parser.add_argument('--simple_env', default=False, action='store_true')
	parser.add_argument('--r', default=False, action='store_true')
	parser.add_argument('--t', default=False, action='store_true')
	parser.add_argument('--tv', default=False, action='store_true')
	parser.add_argument('--r_tv', default=False, action='store_true')
	parser.add_argument('--env_n_dim', default=2, type=int)
	parser.add_argument('--t_save_rate', default=1, type=int)
	parser.add_argument('--s_data_gather', default=False, action='store_true')
	parser.add_argument('--s_data_path', default='/data1/wjh/code/results/data/', type=str)
	parser.add_argument('--s_try_num', default=0, type=int)
	parser.add_argument('--s_alg_name', default='', type=str)
	parser.add_argument('--s_load_num', default='', type=str)
	parser.add_argument('--island_partial_obs', default=False, action='store_true')
	parser.add_argument('--island_agent_max_power', default=11, type=int)
	parser.add_argument('--island_wolf_max_power', default=9, type=int)
	parser.add_argument('--island_wolf_recover_time', default=5, type=int)
	parser.add_argument('--i_num_landmark', default=2, type=int)
	# parser.add_argument('--x_island_agent_max_power', default=11, type=int)
	# parser.add_argument('--x_island_wolf_max_power', default=10, type=int)
	parser.add_argument('--x_island_agent_max_power', default=51, type=int)
	parser.add_argument('--x_island_wolf_max_power', default=21, type=int)
	parser.add_argument('--x_island_wolf_recover_time', default=5, type=int)
	parser.add_argument('--x_island_harm_range', default=11, type=int)
	parser.add_argument('--x_num_landmark', default=2, type=int)
	parser.add_argument('--x_wolf_rew', default=600, type=int)
	parser.add_argument('--x_landmark_rew', default=10, type=int)
	parser.add_argument('--not_view_landmark', default=False, action='store_true')
	parser.add_argument('--appro_T', default=0.5, type=float)
	return parser


def main():
	parser = arg_parser()
	add_env_params(parser)
	parser.add_argument('--num-timesteps', type=int, default=int(1e12))
	parser.add_argument('--num_env', type=int, default=32)
	parser.add_argument('--use_news', type=int, default=0)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--gamma_ext', type=float, default=0.99)
	parser.add_argument('--lam', type=float, default=0.95)
	parser.add_argument('--update_ob_stats_every_step', type=int, default=0)
	parser.add_argument('--update_ob_stats_independently_per_gpu', type=int, default=0)
	parser.add_argument('--update_ob_stats_from_random_agent', type=int, default=1)
	parser.add_argument('--proportion_of_exp_used_for_predictor_update', type=float, default=1.)
	parser.add_argument('--tag', type=str, default='')
	parser.add_argument('--policy', type=str, default='fc', choices=['fc'])
	parser.add_argument('--int_coeff', type=float, default=1.)
	parser.add_argument('--ext_coeff', type=float, default=2.)
	parser.add_argument('--dynamics_bonus', type=int, default=0)

	args = parser.parse_args()
	logger.configure(dir=logger.get_dir(),
	                 format_strs=['stdout', 'log', 'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else [])
	if MPI.COMM_WORLD.Get_rank() == 0:
		with open(os.path.join(logger.get_dir(), 'experiment_tag.txt'), 'w') as f:
			f.write(args.tag)
		# shutil.copytree(os.path.dirname(os.path.abspath(__file__)), os.path.join(logger.get_dir(), 'code'))

	mpi_util.setup_mpi_gpus()

	seed = 10000 * args.seed + MPI.COMM_WORLD.Get_rank()
	set_global_seeds(seed)

	hps = dict(
		frame_stack=4,
		nminibatches=4,
		nepochs=4,
		lr=0.0001,
		max_grad_norm=0.0,
		use_news=args.use_news,
		gamma=args.gamma,
		gamma_ext=args.gamma_ext,
		max_episode_steps=args.max_episode_steps,
		lam=args.lam,
		update_ob_stats_every_step=args.update_ob_stats_every_step,
		update_ob_stats_independently_per_gpu=args.update_ob_stats_independently_per_gpu,
		update_ob_stats_from_random_agent=args.update_ob_stats_from_random_agent,
		proportion_of_exp_used_for_predictor_update=args.proportion_of_exp_used_for_predictor_update,
		policy=args.policy,
		int_coeff=args.int_coeff,
		ext_coeff=args.ext_coeff,
		dynamics_bonus=args.dynamics_bonus
	)

	tf_util.make_session(make_default=True)
	train(env_id=args.env, num_env=args.num_env, seed=seed,
	      num_timesteps=args.num_timesteps, hps=hps)


if __name__ == '__main__':
	main()
