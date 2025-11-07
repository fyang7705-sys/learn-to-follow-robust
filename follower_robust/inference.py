from pogema_toolbox.algorithm_config import AlgoBase

from follower.inference import FollowerInferenceConfig
from follower.preprocessing import PreprocessorConfig
# noinspection PyUnresolvedReferences

import json
from copy import deepcopy

from follower.training_config import Experiment
from follower.register_env import register_custom_components

from follower_robust.model import DistilledActor, make_student_model

import glob

import os
from argparse import Namespace
from collections import OrderedDict
from os.path import join

import numpy as np

from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from sample_factory.utils.utils import log
from pydantic import Extra, validator

from sample_factory.algo.learning.learner import Learner
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.context import global_model_factory

# from follower.algorithm_utils import AlgoBase

from follower.register_training_utils import register_custom_model
from pathlib import Path
from datetime import datetime

class FollowerConfigRobust(FollowerInferenceConfig, extra=Extra.forbid):
    name: Literal['FollowerRobust'] = 'FollowerRobust'
    num_process: int = 8
    num_threads: int = 8
    path_to_weights: str = "model/follower-robust/"
    preprocessing: PreprocessorConfig = PreprocessorConfig()


class FollowerInferenceRobust:
    def __init__(self, config):
        self.algo_cfg: FollowerConfigRobust = config
        device = config.device

        self.path = config.path_to_weights

        
        with open(join(self.path, 'config.json'), "r") as f:
            flat_config = json.load(f)
            self.exp = Experiment(**flat_config)
            flat_config = Namespace(**flat_config)
        config = flat_config

        config.num_envs = 1
        self.save_json = config.save_json
        if self.save_json:
            self.save_dir = config.save_dir
            Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        student = make_student_model(self.path)
    #     model_factory = global_model_factory()

    #     student = DistilledActor(
    #     model_factory=model_factory,  # uses default factory inside
    #     obs_space=env.observation_space,  # sample obs space
    #     action_space=env.action_space,  # will be inferred from data
    #     cfg=config  # default config
    # )
        # actor_critic = create_actor_critic(config, env.observation_space, env.action_space)
        student.eval()

        if device != 'cpu' and not torch.cuda.is_available():
            os.environ['OMP_NUM_THREADS'] = str(1)
            os.environ['MKL_NUM_THREADS'] = str(1)
            device = torch.device('cpu')
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            log.warning('CUDA is not available, using CPU. This might be slow.')

        student.model_to_device(device)
        if self.algo_cfg.custom_path_to_weights:
            checkpoints = self.algo_cfg.custom_path_to_weights
        else:
            paths = glob.glob(os.path.join(self.path, "checkpoint/student*.pt"))
            if len(paths) == 0:
                raise FileNotFoundError(f"No checkpoint found in {self.path}/checkpoint/")
            checkpoints = sorted(paths)[-1] 

        log.info(f'Loading weights from {checkpoints}')
        state_dict = torch.load(checkpoints, map_location=device, weights_only=True) 
        student.load_state_dict(state_dict)

        self.net = student
        self.device = device
        self.cfg = config

        self.rnn_states = None
        self.observations = None 

    def collect_data(self, normalized_obs, rnn_states, policy_outputs):
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        save_path = os.path.join(self.save_dir, f"sample_{timestamp}.pt")

        data_to_save = {
            "normalized_obs": {k: v.detach().cpu() for k, v in normalized_obs.items()},
            "rnn_states": rnn_states.detach().cpu() ,
            "policy_outputs": {
                k: v.detach().cpu()
                for k, v in policy_outputs.items()
                if k in ['actions', 'action_logits', 'log_prob_actions']
            }
        }

        # 保存为pt格式
        torch.save(data_to_save, save_path)
    
    def act(self, observations):
        self.observations = observations
        self.rnn_states = torch.zeros([len(observations), get_rnn_size(self.cfg)], dtype=torch.float32,
                                      device=self.device) if self.rnn_states is None else self.rnn_states
        # print(f"observations shape: {len(observations)}\n")
        # print(f"observations:{observations[0]}\n")
        obs = AttrDict(self.transform_dict_observations(observations))
        with torch.no_grad():
            normalized_obs = prepare_and_normalize_obs(self.net, obs)
            # print("normalized_obs shape", {k: v.shape for k, v in normalized_obs.items()})
            policy_outputs = self.net(normalized_obs, self.rnn_states)
            if self.save_json:
                self.collect_data(normalized_obs, self.rnn_states, policy_outputs)
        # print(f"observations after prepare_and_normalize_obs:{obs}\n")
        self.rnn_states = policy_outputs['new_rnn_states']
        # print(f"policy_outputs:, {policy_outputs}\n")



        return policy_outputs['actions'].cpu().numpy()

    def reset_states(self):
        torch.manual_seed(self.algo_cfg.seed)
        self.rnn_states = None

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_model_parameters(self):
        return self.count_parameters(self.net)

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

    def to_onnx(self, filename='follower.onnx'):
        self.net.eval()
        r = self.algo_cfg.training_config.preprocessing.network_input_radius
        log.info(f"Saving model with network_input_radius = {r}")
        d = 2 * r + 1
        obs_example = torch.rand(1, 2, d, d, device=self.device)
        rnn_example = torch.rand(1, 1, device=self.device)
        with torch.no_grad():
            q = self.net({'obs': obs_example}, rnn_example)
            print(q)
        input_names = ['obs', 'rnn_state']
        output_names = ['values', 'action_logits', 'log_prob_actions', 'actions', 'new_rnn_states']

        torch.onnx.export(self.net, ({'obs': obs_example}, rnn_example), filename,
                          input_names=input_names, output_names=output_names,
                          export_params=True)
