import gymnasium
import numpy as np
from planner.plan import Planner
from follower.preprocessing import wrap_preprocessors
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs


from argparse import Namespace

import torch
import json
from os.path import join
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.utils import log
from sample_factory.utils.attr_dict import AttrDict
from collections import OrderedDict
import copy
from scipy.ndimage import distance_transform_edt


def planner_preprocessor(env, algo_config):
    env = planner_wrap_preprocessor(env, algo_config.preprocessing)
    return env

def planner_wrap_preprocessor(env, config):
    env = wrap_preprocessors(env, config)
    env = LowLevelWrapper(env, config)
    env = PlannerWrapper(env, config)
    return env

class LowLevelWrapper(gymnasium.ActionWrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.action_space = gymnasium.spaces.Discrete(1)
        self.paths = None
        self.prev_observations = None
        self.model_path = 'model/follower'
        with open(join(self.model_path, 'config.json'), "r") as f:
            self.model_config = json.load(f)
            self.model_config = Namespace(**self.model_config)
        self.follower = create_actor_critic(self.model_config, env.observation_space, env.action_space)
        checkpoints = Learner.get_checkpoints(join(self.model_path, "checkpoint_p0"),
                                              "checkpoint_*")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        checkpoint_dict = Learner.load_checkpoint(checkpoints, self.device)
        self.follower.load_state_dict(checkpoint_dict['model'])
        self.follower.eval()
        self.follower.model_to_device(self.device)

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

    def set_path(self, paths):
        self.paths = paths

    def act(self, observations):
        self.rnn_states = torch.zeros([len(observations), get_rnn_size(self.model_config)], dtype=torch.float32,
                                      device=self.device) if self.rnn_states is None else self.rnn_states
        # print(observations[0])
        # print("="*50)
        r = observations[0]['obs'][0].shape[0] // 2
        for k, obs in enumerate(observations):
            for idx, (gx, gy) in enumerate(self.paths[k]):
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)
                if x is not None and y is not None:
                    obs['obs'][0][x, y] = 1.0
                else:
                    break
        obs = AttrDict(self.transform_dict_observations(observations))
        with torch.no_grad():
            # print(obs['obs'][0])
            normalized_obs = prepare_and_normalize_obs(self.follower, obs)
            # print(normalized_obs)
            # print("normalized_obs shape", {k: v.shape for k, v in normalized_obs.items()})
            policy_outputs = self.follower(normalized_obs, self.rnn_states)
        # print(f"observations after prepare_and_normalize_obs:{obs}\n")
        self.rnn_states = policy_outputs['new_rnn_states']
        # print('='*50)
        # print(observations[0]['obs'][0])
        # print("xy", observations[0]['xy'], "target_xy", observations[0]['target_xy'])
        # print(policy_outputs['actions'][0])
        # print(self.paths[0])
        
        # print(f"policy_outputs:, {policy_outputs}\n")
        return policy_outputs['actions'].cpu().numpy()
    
    def action(self, action):
        action = self.act(self.prev_observations)
        return action

    def step(self, action):
        
        observation, reward, done, tr, info = self.env.step(self.action(action))
        self.prev_observations = observation
        return observation, reward, done, tr, info

    def reset(self, **kwargs):
        self.rnn_states = None
        observations, infos = self.env.reset(**kwargs)
        self.prev_observations = observations
        return observations, infos


class PlannerWrapper(gymnasium.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.latent_dim = 3
        self.action_space = gymnasium.spaces.Box(
            low=0,
            high=10,
            # shape=((2 * config.network_input_radius + 1) * (2 * config.network_input_radius + 1), ),
            shape = (self.latent_dim,),
            dtype=np.float32
        )
        self.prev_observations = None
        self.planner = Planner(config)
        self.plan_window = 1
        self.obs_radius = None
        self.cost_basis = [
        self.goal_seeking,
        self.obstacle_avoidance,
        self.agent_avoidance,
        ]
        self.arrive_instances = 0

    def observation(self, observations):
        observations = copy.deepcopy(observations)
        self.prev_observations = copy.deepcopy(observations)
        self.obs_radius = len(observations[0]['obs'][0]) // 2
        self.planner.update(observations=observations)
        mats = self.planner._agent.get_dist_mat(observations)
        # print(np.array2string(
        # observations[0]['obs'][0],
        # precision=2,
        # suppress_small=True
        # ))
        for k, mat in enumerate(mats):
            obs_map = observations[k]['obs'][0]
            observations[k]['obs'][0] = np.where(obs_map != -1, mat, -1)
            # print('obs')
            # print(obs['obs'][0][self.obs_radius, self.obs_radius])
            # print(obs['obs'][1])
        # print(observations[0]['xy'])
        # print(observations[0]['target_xy'])
        # print(np.array2string(
        # observations[0]['obs'][0],
        # precision=2,
        # suppress_small=True
        # ))
        # print('-'*50)
        return observations

    def goal_seeking(self, observations):
        self.planner.update(observations=observations)
        goal_maps = self.planner._agent.get_dist_mat(observations)
        # print("goal maps")
        # print(goal_maps[0])
        return -goal_maps

    def obstacle_avoidance(self, observations, sigma=1.5):
        B = len(observations)
        H, W = observations[0]['obs'][0].shape
        obstacle_maps = np.zeros((B, H, W), dtype=np.float32)
        for k, o in enumerate(observations):
            obs = o['obs'][0]
            free_mask = (obs != -1)
            clearance = distance_transform_edt(free_mask)
            phi_clear = np.exp(-clearance / sigma)
            phi_clear[~free_mask] = 1

            obstacle_maps[k] = phi_clear
        # print("obstacle maps")
        # print(obstacle_maps[0])
        return obstacle_maps
    
    def agent_avoidance(self, observations, sigma=0.8):
        B = len(observations)
        H, W = observations[0]['obs'][1].shape

        agent_maps = np.zeros((B, H, W), dtype=np.float32)
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        for k, o in enumerate(observations):
            obs = o['obs'][1]
            potential = np.zeros((H, W), dtype=np.float32)
            agent_positions = np.argwhere(obs > 0)
            for ay, ax in agent_positions:
                if ay == H // 2 and ax == W // 2:
                    continue
                dy = ys - ay
                dx = xs - ax
                dist2 = dy * dy + dx * dx
                potential += np.exp(-dist2 / (2 * sigma * sigma))
            agent_maps[k] = potential / max(len(agent_positions), 1)
        # print("agent maps")
        # print(agent_maps[0])
        return agent_maps
    
    def cost_map(self, observations, action):
        action = np.array(action)
        B = len(observations)
        H, W = observations[0]['obs'][1].shape
        # print('obs')
        # print(np.array2string(
        #     observations[0]['obs'][0],
        #     precision=1,
        #     suppress_small=True
        #     ))
        # print('agent')
        # print(np.array2string(
        #     observations[0]['obs'][1],
        #     precision=1,
        #     suppress_small=True
        #     ))
        cost_maps = np.zeros((B, H, W), dtype=np.float32)
        for i, phi in enumerate(self.cost_basis):
            cost_basis = phi(observations)
            # print(action[0, i])
            # print(np.array2string(
            # cost_basis[0],
            # precision=1,
            # suppress_small=True
            # ))
            cost_maps += action[:, i][:, None, None] * cost_basis   
        # print("cost maps")
        # print(cost_maps[0])
        return cost_maps  
                  
    def step(self, action):
        num_agents = len(self.prev_observations)
        # print('*'*50)
        cost_map = self.cost_map(self.prev_observations, action)
        cost_map = np.clip(cost_map, 0.0, 0)
        # print("cost_map")
        # print(np.array2string(
        #     cost_map[0],
        #     precision=1,
        #     suppress_small=True
        # ))
        # print('*'*50)
        self.planner._agent.set_dynamic_cost(cost_map, observations=self.prev_observations)
        paths_len = np.zeros((num_agents,), dtype=np.float32)
        on_goal = np.zeros((num_agents,), dtype=np.float32)
        for _ in range(self.plan_window):
            paths = self.planner.get_path()    
            self.env.set_path(paths)
            observation, reward, done, tr, info = self.env.step(0)
            # print("** xy", observation[0]['xy'], "** target_xy", observation[0]['target_xy'])
            self.planner.update(observations=observation) # update cur_pos and cur_goal
            # reward += 0.1 * reward
            if tr[0] == True:
                # print("truncated", tr[0])
                # print("avg_throughout", info[0]['metrics'])
                # print("avg_throughout_local", self.arrive_instances / 512)
                self.arrive_instances = 0
            # if 'metrics' in info:
            #     print("info", info[0])    
            for k in range(num_agents):
                paths_len[k] += len(paths[k])
                reward[k] = 0                               
                if info[k]['on_goal'] == True:
                    on_goal[k] += 1
                    self.arrive_instances += 1
            # print("done", done[0])
            # print("tr", tr[0])
            # print("info", info[0])
        for k in range(num_agents):
            prev_dist = self.planner._agent.planner[k].get_dist_to_goal(self.prev_observations[k]['xy'])
            dist = self.planner._agent.planner[k].get_dist_to_goal(observation[k]['xy']) 
            reward[k] += (prev_dist - dist) / self.plan_window
            reward[k] -= 0.01 * paths_len[k] / self.plan_window
            # reward[k] += on_goal[k] * 10
           
        # print("reward", reward[0])
        # print('*'*50)
        # print("key", info[0].keys())
        return self.observation(observation), reward, done, tr, info

    def reset_state(self):
        self.planner.reset_states()
        self.planner._agent.add_grid_obstacles(self.get_global_obstacles(), self.get_global_agents_xy())
        
    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.prev_observations = observations
        self.reset_state()
        return self.observation(observations), infos
