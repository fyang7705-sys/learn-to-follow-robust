from pogema_toolbox.algorithm_config import AlgoBase

from follower.inference import FollowerInferenceConfig
from follower.preprocessing import PreprocessorConfig
# noinspection PyUnresolvedReferences

import json

from follower.training_config import Experiment
from follower.register_env import register_custom_components

import os
from argparse import Namespace
from collections import OrderedDict
from os.path import join

import numpy as np


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from sample_factory.utils.utils import log
from pydantic import Extra

from sample_factory.algo.learning.learner import Learner
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
# from follower.algorithm_utils import AlgoBase

from follower_robust.register_training_utils import register_custom_model_context
from pathlib import Path
from datetime import datetime




class PlannerInferenceConfig(FollowerInferenceConfig, extra=Extra.forbid):
    name: Literal['Planner'] = 'Planner'
    num_process: int = 8
    num_threads: int = 8
    path_to_weights: str = "model/follower-robust"
    preprocessing: PreprocessorConfig = PreprocessorConfig()
class PlannerInference:
    def __init__(self, config):
        self.algo_cfg: PlannerInferenceConfig = config
        device = config.device

        register_custom_model_context()
        self.path = config.path_to_weights

        
        with open(join(self.path, 'config.json'), "r") as f:
            flat_config = json.load(f)
            self.exp = Experiment(**flat_config)
            flat_config = Namespace(**flat_config)
        env_name = self.exp.environment.env
        register_custom_components(env_name)
        config = flat_config

        config.num_envs = 1
        env = make_env_func_batched(config, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
        actor_critic = create_actor_critic(config, env.observation_space, env.action_space)
        actor_critic.eval()
        env.close()

        if device != 'cpu' and not torch.cuda.is_available():
            os.environ['OMP_NUM_THREADS'] = str(1)
            os.environ['MKL_NUM_THREADS'] = str(1)
            device = torch.device('cpu')
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            log.warning('CUDA is not available, using CPU. This might be slow.')

        actor_critic.model_to_device(device)
        name_prefix = dict(latest="checkpoint", best="best")['best']
        policy_index = 0 if 'policy_index' not in flat_config else flat_config.policy_index

        checkpoint = Learner.get_checkpoints(os.path.join(self.path, f"checkpoint_p{policy_index}"),
                                              f"{name_prefix}_*")

        if self.algo_cfg.custom_path_to_weights:
            log.info(f"custom_path_to_weights:{self.algo_cfg.custom_path_to_weights}")
            checkpoint = self.algo_cfg.custom_path_to_weights
            # checkpoint = torch.load(self.algo_cfg.custom_path_to_weights, map_location="cpu")

        checkpoint_dict = Learner.load_checkpoint(checkpoint, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])
        self.net = actor_critic
        self.device = device
        self.cfg = config

        self.rnn_states = None

    
    def act(self, observations):
        self.observations = observations
        self.rnn_states = torch.zeros([len(observations), get_rnn_size(self.cfg)], dtype=torch.float32,
                                      device=self.device) if self.rnn_states is None else self.rnn_states
        # print(f"observations shape: {len(observations)}\n")
        # print(f"observations:{observations[0]}\n")
        obs = AttrDict(self.transform_dict_observations(observations))
        with torch.no_grad():
            normalized_obs = prepare_and_normalize_obs(self.net, obs)
            # log.error(f"normalized_obs{normalized_obs}")
            # print("normalized_obs shape", {k: v.shape for k, v in normalized_obs.items()})
            policy_outputs = self.net(normalized_obs, self.rnn_states)
            # if self.save_json:
            #     self.collect_data(normalized_obs, self.rnn_states, policy_outputs)
        # print(f"observations after prepare_and_normalize_obs:{obs}\n")
        self.rnn_states = policy_outputs['new_rnn_states']
        # print(f"policy_outputs:, {policy_outputs}\n")
        return policy_outputs['actions'].cpu().numpy()
    

    def reset_states(self):
        torch.manual_seed(self.algo_cfg.seed)
        self.rnn_states = None

    @staticmethod
    def transform_dict_observations(observations):
        """Transform list of dict observations into a dict of lists."""
        obs_dict = dict()
        if isinstance(observations[0], (dict, OrderedDict)):
            for key in observations[0].keys():
                if not isinstance(observations[0][key], str):
                    obs_dict[key] = [o[key] for o in observations]
        else:
            # handle flat observations also as dict
            obs_dict['obs'] = observations

        for key, x in obs_dict.items():
            obs_dict[key] = np.stack(x)

        return obs_dict
