import gymnasium
import numpy as np
from planner.plan import Planner
from follower.preprocessing import wrap_preprocessors
from sample_factory.algo.learning.learner import Learner

from argparse import Namespace

import torch
import json
from os.path import join
from sample_factory.model.actor_critic import create_actor_critic
def planner_preprocessor(env, algo_config):
    env = wrap_preprocessors(env, algo_config.training_config.preprocessing)
    env = LowLevelWrapper(env, algo_config.training_config.preprocessing)
    env = PlannerWrapper(env, algo_config.training_config.preprocessing)
    return env

class LowLevelWrapper(gymnasium.ActionWrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.action_space = gymnasium.spaces.Discrete(1)
        self.paths = None
        self.prev_observations = None
        self.model_path = 'model/follower'
        with open(join(self.path, 'config.json'), "r") as f:
            flat_config = json.load(f)
            flat_config = Namespace(**flat_config)
        self.follower = create_actor_critic(flat_config, env.observation_space, env.action_space)
        checkpoints = Learner.get_checkpoints(join(self.path, "checkpoint_p0"),
                                              "checkpoint_*")
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        self.follower.load_state_dict(checkpoint_dict['model'])

    def set_path(self, paths):
        self.paths = paths

    def action(self, action):
        with torch.no_grad():
            action = self.follower(self.prev_observations)
        return action

    def step(self, action):
        self.env.set_path(self.paths)
        
        observation, reward, done, tr, info = self.env.step(self.action(action))
        self.prev_observations = observation
        return observation, reward, done, tr, info

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.prev_observations = observations
        return observations, infos


class PlannerWrapper(gymnasium.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.action_space = gymnasium.spaces.Box(
            low=0,
            high=10,
            shape=(config.network_input_radius, config.network_input_radius),
            dtype=np.float32
        )
        self.prev_observations = None
        self.planner = Planner(config)
        self.plan_window = 3

    def observation(self, observations):
        self.prev_observations = observations
        obs_radius = len(observations[0]['obstacles']) // 2
        self.planner.update(observations=observations)
        self.planner._agent.update_dist_mat(observations)
        mats = self.planner._agent.get_dist_mat()
        mats = (mats - mats[obs_radius][obs_radius]) / obs_radius
        for k, mat in enumerate(mats):
            obs = observations[k]
            for y, row in enumerate(mat):
                for x, val in enumerate(row):
                    if obs['obstacles'][y, x] != -1:
                        obs['obstacles'][y, x] = np.exp(-val)
                    else:
                        obs['obstacles'][y, x] = -1
        return observations
    def step(self, action):
        self.planner._agent.set_dynamic_cost(action, observations=self.prev_observations)
        for _ in range(self.plan_window):
            paths = self.planner.get_path()
            self.env.set_path(paths)
            observation, reward, done, tr, info = self.env.step(0)
            self.planner.update(observations=observation)
        return self.observation(observation), reward, done, tr, info

    def reset_state(self):
        self.planner.reset_states()
        self.planner._agent.add_grid_obstacles(self.get_global_obstacles(), self.get_global_agents_xy())
        
    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.prev_observations = observations
        self.reset_state()
        return self.observation(observations), infos
