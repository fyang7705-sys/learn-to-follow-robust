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



if __name__ == "__main__":
    dataset = DistillationDataset(
    data_dir="../teacher_data/",
    obs_keys=["xy", "target_xy", "obs"],  # 仅选部分输入（可选）
    output_keys=["action_logits", "actions"]
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn = collate_fn)

    sample = next(iter(loader))
    print("normalized_obs shape:", {k: v.shape for k, v in sample["normalized_obs"].items()})
    # print("✅ normalized_obs keys:", sample["normalized_obs"])
    # print("✅ rnn_states shape:", sample["rnn_states"])
    # print("✅ policy_outputs keys:", sample["policy_outputs"])