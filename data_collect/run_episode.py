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



def run_episode(env, algo):
    global prob_index
    bug_prob = bug_probs[prob_index]
    algo.reset_states()
    results_holder = ResultsHolder()
    save_dir = "encode_data"
    os.makedirs(save_dir, exist_ok=True)

    obs, _ = env.reset(seed=env.grid_config.seed)
    while True:
        old_obs = copy.deepcopy(obs)
        action = algo.act(obs)
        action = bug_action(action, bug_prob)
        obs, rew, terminated, truncated, infos = env.step(action)
        results_holder.after_step(infos)
        # print("\n===== OBS STRUCTURE =====")
        # pprint.pprint(obs)
        # print("===== OBS TYPE =====")
        # print(type(obs))
        # if isinstance(obs, list):
        #     print("List length:", len(obs))
        #     if len(obs) > 0:
        #         print("Type of obs[0]:", type(obs[0]))
        #         if isinstance(obs[0], dict):
        #             print("Keys of obs[0]:", list(obs[0].keys()))
        # print("==========================\n")
        data_to_save = {
            "old_obs": old_obs,
            "new_obs": obs,
            "actions": torch.tensor(action),
            "rewards": torch.tensor(rew),
            "label": bug_prob,
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(save_dir, f"sample_{timestamp}.pt")

        # 保存为pt格式
        torch.save(data_to_save, save_path)

        if all(terminated) or all(truncated):
            break
    prob_index = (prob_index + 1) % len(bug_probs)
    return results_holder.get_final()