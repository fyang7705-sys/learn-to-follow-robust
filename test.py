from pathlib import Path
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.model.actor_critic import create_actor_critic
from follower.training_config import Experiment
from follower.register_env import register_custom_components
from follower.register_training_utils import register_custom_model

from argparse import Namespace
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.algo.utils.context import global_model_factory

import json
import os
# full_algo_name = 'Follower-v1'
# directory = Path(f'renders/{full_algo_name}/')
# print(directory)
path = 'model/follower'
with open(os.path.join(path, 'config.json'), "r") as f:
    flat_config = json.load(f)
    exp = Experiment(**flat_config)
    flat_config = Namespace(**flat_config)
env_name = exp.environment.env
register_custom_model()
register_custom_components(env_name)
config = flat_config
config.num_envs = 1
env = make_env_func_batched(config, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
# print("config: ", config)
print(f"env.observation_space: {env.observation_space}\n")
print(f"env.action_space: {env.action_space}\n")
model_factory = global_model_factory()
model = create_actor_critic(config, env.observation_space, env.action_space)
# model = ActorCriticSharedWeights(model_factory = model_factory, obs_space=env.observation_space, action_space=env.action_space, cfg=config)



