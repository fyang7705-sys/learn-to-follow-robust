import os
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
class DistillationDataset(Dataset):

    def __init__(self, data_dir, obs_keys=None, output_keys=None, max_samples=None):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if max_samples:
            self.files = self.files[:max_samples]

        self.obs_keys = obs_keys 
        self.output_keys = output_keys or ["action_logits"] 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")

        obs = data["normalized_obs"]
        # print("obs shape", {k: v.shape for k, v in obs.items()})
        if self.obs_keys:
            obs = {k: obs[k] for k in self.obs_keys if k in obs}
        # print("obs key shape", {k: v.shape for k, v in obs.items()})
        rnn_states = data.get("rnn_states", None)

        outputs = data["policy_outputs"]

        return {
            "normalized_obs": obs,
            "rnn_states": rnn_states,
            "policy_outputs": outputs,
        }

class EncodeDataset(Dataset):
    def __init__(self, folder, window_size):
        file_path = folder + "/*.pt"
        self.files = glob.glob(file_path)
        print(f"✅ Found {len(self.files)} data files in {folder}")
        self.window_size = window_size


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        old_obs = torch.stack([
            torch.tensor(item["obs"], dtype=torch.float32)
            for item in data["old_obs"]
        ], dim=0)

        new_obs = torch.stack([
            torch.tensor(item["obs"], dtype=torch.float32)
            for item in data["new_obs"]
        ], dim=0)

        action = data["actions"].to(torch.float32).unsqueeze(1)

        reward = data["rewards"].to(torch.float32).unsqueeze(1)

        term = torch.zeros_like(reward)
        label = torch.tensor(data["label"], dtype=torch.float32).repeat(old_obs.shape[0])
        # print("shape of label: ", label.shape)
        # print("shape of old_obs: ", old_obs.shape)
        return {
            "state": old_obs,       
            "action": action,     
            "reward": reward,       
            "next_state": new_obs,   
            "term": term,
            "label": label,
        }

class EncodeDatasetForWindow(Dataset):
    def __init__(self, folder, window_size):
        folder = folder + f"/{window_size}"
        file_path = folder + "/*.pt"
        self.files = glob.glob(file_path)
        print(f"✅ Found {len(self.files)} data files in {folder}")
        self.window_size = window_size

    def __len__(self):   
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        obs = data["obs"].to(torch.float32).transpose(0, 1) # [num_agents, window_size, C, H, W]
        action = data["actions"].to(torch.float32).unsqueeze(-1).transpose(0, 1)  # [num_agents, window_size, 1]
        reward = data["rewards"].to(torch.float32).unsqueeze(-1).transpose(0, 1)  # [num_agents, window_size, 1]
        term = torch.zeros_like(reward)
        label = torch.tensor(data["label"], dtype=torch.float32).repeat(obs.shape[0])
        # print("obs shape:", obs.shape)
        # print("action shape:", action.shape)
        # print("reward shape:", reward.shape)
        # print("term shape:", term.shape)
        # print("label shape:", label.shape)
        return {
            "obs": obs,
            "action": action,
            "reward": reward,
            "term": term,
            "label": label,
        }

def prepare_batch_for_encoder(batch):
    obs = batch['obs']
    action = batch['action']
    reward = batch['reward']
    term = batch['term']
    label = batch['label']
    return obs, action, reward, term, label


def collate_fn(samples):

    obs_keys = set()
    for s in samples:
        obs_keys.update(s["normalized_obs"].keys())
    batched_obs = {}
    for k in obs_keys:
        tensors = []
        for s in samples:
            v = s["normalized_obs"].get(k, None)
            if v is None:
                continue
            # 确保是Tensor
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            tensors.append(v)
        if len(tensors) == 0:
            continue
        batched_obs[k] = torch.cat(tensors, dim=0)  # concat agent dimension

    batched_rnn = torch.cat([torch.tensor(s["rnn_states"]) for s in samples], dim=0)


    keys = set()
    for s in samples:
        keys.update(s["policy_outputs"].keys())
    batched_outputs = {}
    for k in keys:
        parts = []
        for s in samples:
            v = s["policy_outputs"].get(k, None)
            if v is None:
                continue
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            parts.append(v)
        if len(parts) == 0:
            batched_outputs[k] = None
        else:
            batched_outputs[k] = torch.cat(parts, dim=0)
    return {
        "normalized_obs": batched_obs,
        "rnn_states": batched_rnn,
        "policy_outputs": batched_outputs
    }


def encode_collate_fn(samples):

    batched = {}

    for k in ["state", "next_state", "action", "reward", "term"]:
        tensors = []
        for s in samples:
            v = s[k]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            tensors.append(v)
        batched[k] = torch.cat(tensors, dim=0)

    label_list = []
    for s in samples:
        label_list.append(s["label"])

    batched["label"] = torch.cat(label_list, dim=0)

    return batched

def encode_collate_fn_for_window(samples):
    batched = {}
    for k in ["obs", "action", "reward", "term", "label"]:
        tensors = []
        for s in samples:
            v = s[k]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            tensors.append(v)
        batched[k] = torch.cat(tensors, dim=0)
    return batched

if __name__ == "__main__":
    # dataset = DistillationDataset(
    # data_dir="../teacher_data/",
    # obs_keys=["xy", "target_xy", "obs"],  # 仅选部分输入（可选）
    # output_keys=["action_logits", "actions"]
    # )

    # loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn = collate_fn)
    # sample = next(iter(loader))
    # print("normalized_obs shape:", {k: v.shape for k, v in sample["normalized_obs"].items()})
    # print("✅ normalized_obs keys:", sample["normalized_obs"])
    # print("✅ rnn_states shape:", sample["rnn_states"])
    # print("✅ policy_outputs keys:", sample["policy_outputs"])
    
    # data_dir = "../encode_data/"
    # batch_files = 4
    # dataset = EncodeDataset(data_dir)
    # loader = DataLoader(dataset, batch_size=batch_files, shuffle=True, collate_fn=None)
    # sample = next(iter(loader))
    # print("state shape:", sample["state"].shape)
    # print("action shape:", sample["action"].shape)
    # print("reward shape:", sample["reward"].shape)
    # print("next_state shape:", sample["next_state"].shape)
    # print("term shape:", sample["term"].shape)
    # print("label shape:", sample["label"].shape)
    window_size = 5
    data_dir = "../encoder_data/"
    dataset = EncodeDatasetForWindow(data_dir, window_size)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    sample = next(iter(loader))
    # print("obs shape:", sample["obs"].shape)
    # print("action shape:", sample["action"].shape)
    # print("reward shape:", sample["reward"].shape)
    # print("term shape:", sample["term"].shape)
    # print("label shape:", sample["label"].shape)