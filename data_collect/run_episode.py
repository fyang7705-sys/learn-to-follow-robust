from pogema_toolbox.results_holder import ResultsHolder
import os
from datetime import datetime
import torch
import numpy as np
import copy
import pprint
bug_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
prob_index = 0


def bug_action(action_outputs, bug_prob):
    random_values = np.random.random(action_outputs.shape)
    action_outputs[random_values < bug_prob] = 0
    # print("giao")
    return action_outputs


WINDOW_SIZE = 5  

def collect_data(env, algo):
    global prob_index
    bug_prob = bug_probs[prob_index]
    algo.reset_states()
    results_holder = ResultsHolder()

    save_dir = f"encoder_data/{WINDOW_SIZE}"
    os.makedirs(save_dir, exist_ok=True)

    buffer_obs = []
    buffer_actions = []
    buffer_rewards = []

    obs, _ = env.reset(seed=env.grid_config.seed)

    while True:
        old_obs = copy.deepcopy(obs)

        action = algo.act(obs)
        action = bug_action(action, bug_prob)

        obs, rew, terminated, truncated, infos = env.step(action)
        results_holder.after_step(infos)
        buffer_obs.append(torch.tensor([o['obs'] for o in obs]))
        buffer_actions.append(torch.tensor(action))
        buffer_rewards.append(torch.tensor(rew))

        if len(buffer_obs) == WINDOW_SIZE:

            data_to_save = {
                "obs": torch.stack(buffer_obs),
                "actions": torch.stack(buffer_actions),
                "rewards": torch.stack(buffer_rewards),
                "label": bug_prob,
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = os.path.join(
                save_dir,
                f"sample_{WINDOW_SIZE}_{timestamp}_{bug_prob:.1f}_{np.random.randint(1000)}.pt"
            )

            torch.save(data_to_save, save_path)

            buffer_obs.clear()
            buffer_actions.clear()
            buffer_rewards.clear()

        if all(terminated) or all(truncated):
            break

    prob_index = (prob_index + 1) % len(bug_probs)
    return results_holder.get_final()