import numpy as np
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict
import torch
from pydantic import BaseModel

from follower.planning import ResettablePlanner, PlannerConfig
from follower_robust.encoder import CNNEncoder
from typing import List

class InferenceNetConfig(BaseModel):
    weight_path: str
    hidden_size: int
    task_embedding_size: int
    action_size: int
    reward_size: int
    term_size: int
    obs_shape: List[int]
    window_size: int
    normalize: bool
    transformer_layers: int
    transformer_heads: int


class PreprocessorConfig(PlannerConfig):
    network_input_radius: int = 5
    intrinsic_target_reward: float = 0.01
    use_latent_embedding: bool = True
    inference_windowsize: int = 5
    inference_net: InferenceNetConfig = InferenceNetConfig(
        weight_path="model/follower-robust/checkpoint/encoder/encoder_20251118_065830_762916.pt",
        hidden_size=64,
        task_embedding_size=32,
        action_size=1,
        reward_size=1,
        term_size=1,
        obs_shape=[2, 11, 11],
        window_size=5,
        normalize=False,
        transformer_layers=2,
        transformer_heads=4,
    )



def follower_preprocessor(env, algo_config):
    env = wrap_preprocessors(env, algo_config.training_config.preprocessing)
    return env


def wrap_preprocessors(env, config: PreprocessorConfig, auto_reset=False):
    
    env = FollowerWrapper(env=env, config=config)
    env = CutObservationWrapper(env, target_observation_radius=config.network_input_radius)
    env = ConcatPositionalFeatures(env)
    if config.use_latent_embedding:
        env = EncodeDataCollectionWrapper(env, config=config)
    if auto_reset:
        env = AutoResetWrapper(env)
    return env


class FollowerWrapper(ObservationWrapper):

    def __init__(self, env, config: PreprocessorConfig):
        super().__init__(env)
        self._cfg: PreprocessorConfig = config
        self.re_plan = ResettablePlanner(self._cfg)
        self.prev_goals = None
        self.intrinsic_reward = None

    @staticmethod
    def get_relative_xy(x, y, tx, ty, obs_radius):
        dx, dy = x - tx, y - ty
        if dx > obs_radius or dx < -obs_radius or dy > obs_radius or dy < -obs_radius:
            return None, None
        return obs_radius - dx, obs_radius - dy

    def observation(self, observations):
        # Update cost penalties based on the current observations, independently for each agent.
        self.re_plan.update(observations)

        # Retrieve the shortest path to the global target for each agent.
        paths = self.re_plan.get_path()

        new_goals = []  # Initialize a list to store new goals for each agent.
        intrinsic_rewards = []  # Initialize a list to store intrinsic rewards for each agent.

        # Iterate through agents and their respective paths.
        for k, path in enumerate(paths):
            obs = observations[k]

            # Check if there is no valid path available.
            if path is None:
                new_goals.append(obs['target_xy'])  # Use the target position as a new goal.
                path = []
            else:
                # Check if the agent reached their subgoal from its previous step
                subgoal_achieved = self.prev_goals and obs['xy'] == self.prev_goals[k]
                # Assign an intrinsic reward if conditions are met, otherwise set it to 0.
                intrinsic_rewards.append(self._cfg.intrinsic_target_reward if subgoal_achieved else 0.0)
                # Select a new target point.
                new_goals.append(path[1])

            # Set obstacle values to -1.0 in the observation.
            obs['obstacles'][obs['obstacles'] > 0] *= -1

            # Adding path to the observation, setting path values to +1.0.
            r = obs['obstacles'].shape[0] // 2
            for idx, (gx, gy) in enumerate(path):
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)
                if x is not None and y is not None:
                    obs['obstacles'][x, y] = 1.0
                else:
                    break
            # print(obs['obstacles'])
        # Update the previous goals and intrinsic rewards for the next step.
        self.prev_goals = new_goals
        self.intrinsic_reward = intrinsic_rewards

        return observations

    def get_intrinsic_rewards(self, reward):
        for agent_idx, r in enumerate(reward):
            reward[agent_idx] = self.intrinsic_reward[agent_idx]
        return reward

    def step(self, action):
        observation, reward, done, tr, info = self.env.step(action)
        return self.observation(observation), self.get_intrinsic_rewards(reward), done, tr, info

    def reset_state(self):
        self.re_plan.reset_states()
        self.re_plan._agent.add_grid_obstacles(self.get_global_obstacles(), self.get_global_agents_xy())

        self.prev_goals = None
        self.intrinsic_reward = None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.reset_state()
        return self.observation(observations), infos


class CutObservationWrapper(ObservationWrapper):
    def __init__(self, env, target_observation_radius):
        super().__init__(env)
        self._target_obs_radius = target_observation_radius
        self._initial_obs_radius = self.env.observation_space['obstacles'].shape[0] // 2

        for key, value in self.observation_space.items():
            d = self._initial_obs_radius * 2 + 1
            if value.shape == (d, d):
                r = self._target_obs_radius
                self.observation_space[key] = Box(0.0, 1.0, shape=(r * 2 + 1, r * 2 + 1))

    def observation(self, observations):
        tr = self._target_obs_radius
        ir = self._initial_obs_radius
        d = ir * 2 + 1

        for obs in observations:
            for key, value in obs.items():
                if hasattr(value, 'shape') and value.shape == (d, d):
                    obs[key] = value[ir - tr:ir + tr + 1, ir - tr:ir + tr + 1]

        return observations


class ConcatPositionalFeatures(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.to_concat = []

        observation_space = Dict()
        full_size = self.env.observation_space['obstacles'].shape[0]

        for key, value in self.observation_space.items():
            if value.shape == (full_size, full_size):
                self.to_concat.append(key)
            else:
                observation_space[key] = value

        obs_shape = (len(self.to_concat), full_size, full_size)
        observation_space['obs'] = Box(0.0, 1.0, shape=obs_shape)
        self.to_concat.sort(key=self.key_comparator)
        self.observation_space = observation_space

    def observation(self, observations):
        for agent_idx, obs in enumerate(observations):
            main_obs = np.concatenate([obs[key][None] for key in self.to_concat])
            for key in self.to_concat:
                del obs[key]

            for key in obs:
                obs[key] = np.array(obs[key], dtype=np.float32)
            observations[agent_idx]['obs'] = main_obs.astype(np.float32)
        return observations

    @staticmethod
    def key_comparator(x):
        if x == 'obstacles':
            return '0_' + x
        elif 'agents' in x:
            return '1_' + x
        return '2_' + x


class EncodeDataCollectionWrapper(ObservationWrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.obs_buffer = []
        self.reward_buffer = []
        self.action_buffer = []
        self.terminal_buffer = []
        self.window_size = config.inference_windowsize
        self.inference_net = CNNEncoder()
        inference_net_state_dict = torch.load(config.inference_net.weight_path, map_location = torch.device('cpu'))
        self.inference_net.load_state_dict(inference_net_state_dict)
    def step(self, action):
        observations, reward, terminated, truncated, info = self.env.step(action)
        self.obs_buffer.append(observations['obs'])
        self.reward_buffer.append(reward)
        self.action_buffer.append(action)
        self.terminal_buffer.append(0) # 训练encoder的时候默认terminal为0
        if len(self.obs_buffer) > self.window_size:
            self.obs_buffer.pop(0)
            self.reward_buffer.pop(0)
            self.action_buffer.pop(0)
            self.terminal_buffer.pop(0)
        if len(self.obs_buffer) == self.window_size:
            obs_t = torch.stack(self.obs_buffer).transpose(0, 1) # [agent, window_size, C, H, W]
            reward_t = torch.tensor(self.reward_buffer).transpose(0, 1) # [agent, window_size, 1]
            action_t = torch.tensor(self.action_buffer).transpose(0, 1) # [agent, window_size, 1]
            terminal_t = torch.tensor(self.terminal_buffer).transpose(0, 1) # [agent, window_size, 1]
            with torch.no_grad():
                z = self.inference_net(obs_t, action_t, reward_t, terminal_t)
                z = torch.mean(z, dim = 0)
                z = z.detach().cpu().numpy()
                observations['latent'] = z
        return observations, reward, terminated, truncated, info

class AutoResetWrapper(gymnasium.Wrapper):
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if all(terminated) or all(truncated):
            observations, _ = self.env.reset()
        return observations, rewards, terminated, truncated, infos
