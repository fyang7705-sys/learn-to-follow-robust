from pogema_toolbox.create_env import create_env_base, Environment
from gymnasium import ActionWrapper
from typing import List, Optional
import numpy as np
from pogema_toolbox.create_env import create_env_base

class BugEnvironment(Environment, ):
    bug_probs: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    custom_bug_prob: Optional[float] = None

class BugActionWrapper(ActionWrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.bug_probs = config.bug_probs
        self.custom_bug_prob = config.custom_bug_prob
        self.bug_prob = 0.0
        self.episode = 0
    def bug_action(self, action_outputs, bug_prob):
        # print("action", action_outputs)
        action_outputs = np.array(action_outputs, dtype=np.int64)
        random_values = np.random.random(len(action_outputs))
        action_outputs[random_values < bug_prob] = 0
        # print("giao")
        return action_outputs.tolist()
    def action(self, action) :
        return self.bug_action(action, self.bug_prob)
    
    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        # self.bug_prob = self.bug_probs[(self.episode) % (len(self.bug_probs) * 100) //  100]
        if self.custom_bug_prob != None:
            self.bug_prob = self.custom_bug_prob
        else:
            self.bug_prob = np.random.choice(self.bug_probs)
            print("giao", self.bug_prob)
        self.episode += 1
        return observations, infos

def create_env_eval(config: BugEnvironment):
    env = create_env_base(config=config)
    env = BugActionWrapper(env, config)
    return env
    