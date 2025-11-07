from follower_robust.dataloader import DistillationDataset, collate_fn
from follower_robust.model import make_student_model

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
from datetime import datetime

def train_distill(
    data_dir: str,
    config_dir: str = "model/follower-robust",
    save_model_path: str = "model/follower-robust/checkpoint/",
    epochs: int = 10,
    batch_files: int = 16,
    lr: float = 1e-3,
    temp: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Using device: {device}")
   
    os.makedirs(save_model_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    save_model_path = os.path.join(save_model_path, f"student_{timestamp}.pt")
    ds = DistillationDataset(data_dir=data_dir, obs_keys=None, output_keys=["action_logits"])
    loader = DataLoader(ds, batch_size=batch_files, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    student = make_student_model(config_dir)
    student.to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    # training loop
    for ep in range(epochs):
        student.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            obs = {k: v.to(device) for k, v in batch["normalized_obs"].items()}
            rnn = batch["rnn_states"].to(device)

            t_logits = batch["policy_outputs"]["action_logits"].to(device)  # shape [N, action_dim]
            # teacher soft targets
            with torch.no_grad():
                t_soft = F.softmax(t_logits / temp, dim=-1)

            # student forward
            s_logits = student(obs, rnn)['action_logits']  # [N, action_dim]

            # KL Loss: KL( teacher || student ) implemented as KLDivLoss(log_softmax(student), soft_teacher)
            log_p = F.log_softmax(s_logits / temp, dim=-1)
            loss_kl = kl_loss(log_p, t_soft) * (temp * temp)

            optimizer.zero_grad()
            loss_kl.backward()
            optimizer.step()

            total_loss += loss_kl.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"[Epoch {ep+1}/{epochs}] avg_loss={avg_loss:.6f}")

        # epoch end -> optionally save intermediate
        torch.save(student.state_dict(), save_model_path)

    # final save
    torch.save(student.state_dict(), save_model_path)
    print(f"Saved final student to {save_model_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="teacher_data/")
    parser.add_argument("--save", type=str, default="model/follower-robust/checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_files", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp", type=float, default=1.0)
    args = parser.parse_args()
    
    
    train_distill(
        data_dir=args.data_dir,
        save_model_path=args.save,
        epochs=args.epochs,
        batch_files=args.batch_files,
        lr=args.lr,
        temp=args.temp
    )
