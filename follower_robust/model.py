from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.context import global_model_factory

from follower.training_config import Experiment
from follower.register_env import register_custom_components
from follower.register_training_utils import register_custom_model

from argparse import Namespace

from torch import nn, Tensor
import os
import json
from typing import Dict, Optional

class DistilledActor(ActorCritic):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
        super().__init__(obs_space, action_space, cfg)

        self.encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.encoders = [self.encoder]  # a single shared encoder

        self.core = model_factory.make_model_core_func(cfg, self.encoder.get_out_size())

        self.decoder = model_factory.make_model_decoder_func(cfg, self.core.get_out_size())
        decoder_out_size: int = self.decoder.get_out_size()

        self.action_parameterization = self.get_action_parameterization(decoder_out_size)

        self.apply(self.initialize_weights)

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(normalized_obs_dict)
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(
        self, core_output, sample_actions: bool
    ) -> TensorDict:
        decoder_output = self.decoder(core_output)

        result = TensorDict()

        action_distribution_params, self.last_action_distribution = self.action_parameterization(
            decoder_output
        )

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(
        self, normalized_obs_dict, rnn_states
    ) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result
        
        
def make_student_model(cfg_path):

    register_custom_model()
    with open(os.path.join(cfg_path, 'config.json'), "r") as f:
        flat_config = json.load(f)
        exp = Experiment(**flat_config)
        flat_config = Namespace(**flat_config)
        
    config = flat_config
    env_name = exp.environment.env
    register_custom_components(env_name)
    # print(f"config: {config}\n")
    env = make_env_func_batched(config, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))

    model_factory = global_model_factory()
    student = DistilledActor(
        model_factory=model_factory,  # uses default factory inside
        obs_space=env.observation_space,  # sample obs space
        action_space=env.action_space,  # will be inferred from data
        cfg=config  # default config
    )
    return student