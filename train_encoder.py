import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from follower_robust.dataloader import EncodeDataset, EncodeDatasetForWindow, prepare_batch_for_encoder, encode_collate_fn, encode_collate_fn_for_window
from follower_robust.encoder import CNNEncoder
from follower_robust.loss import metric_loss, metric_loss_fast
from pogema_toolbox.registry import ToolboxRegistry
from datetime import datetime
import wandb
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
def log_tsne(encoder, loader, device):
    encoder.eval()

    batch = next(iter(loader))  # 取一个 batch 做可视化
    with torch.no_grad():
        obs, action, reward, term, label = prepare_batch_for_encoder(batch)

        obs = obs.to(device)
        action = action.to(device)
        reward = reward.to(device)
        term = term.to(device)
        label = label.to(device)
        # Forward pass
        z = encoder.context_encoding(
            states=obs,
            actions=action,
            rewards=reward,
            terms=term
        )
        z = z.cpu().numpy()
        label = label.cpu().numpy()

    # t-SNE to 2D
    tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30)
    z_2d = tsne.fit_transform(z)
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c=label, cmap="viridis")
    plt.colorbar()
    plt.savefig("robust-experiments/01-random-20x20/tsne.png")
    plt.close()
    # logging to wandb
    # wandb.log({
    #     "tsne": wandb.Table(
    #         data=[[float(z_2d[i, 0]), float(z_2d[i, 1]), float(labels[i])] for i in range(len(z_2d))],
    #         columns=["x", "y", "label"]
    #     )
    # })




def train(data_dir, learning_rate, batch_size=20, epochs=20, save_path="model/follower-robust/checkpoint/encoder", window_size=5):
    
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    save_path = os.path.join(save_path, f"encoder_{timestamp}.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ToolboxRegistry.info(f"Using device: {device}")
    
    # dataset = EncodeDataset(data_dir, window_size)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=encode_collate_fn)
    dataset = EncodeDatasetForWindow(data_dir, window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn = encode_collate_fn_for_window)
    
    encoder = CNNEncoder(
        obs_shape=(2, 11, 11),
        action_size=1,
        reward_size=1,
        term_size=1,
        window_size=window_size,
        task_embedding_size=32
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    # wandb.init(
    #     project="learn-to-follow-robust",
    #     name=f"encoder_{timestamp}",
    #     config={
    #         "learning_rate": learning_rate,
    #         "epochs": epochs,
    #         "batch_size": batch_size,
    #         "data_dir": data_dir,
    #         "obs_shape": (2, 11, 11),
    #         "action_size": 1,
    #         "reward_size": 1,
    #         "term_size": 1,
    #         "task_embedding_size": 32,
    #     }
    # )
    
    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(epochs):
        encoder.train()
        total_loss = 0

        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress:
            # Prepare batch
            # print(batch['obs'].shape)
            obs, action, reward, term, label = prepare_batch_for_encoder(batch)
            # Move to device
            obs = obs.to(device)
            action = action.to(device)
            reward = reward.to(device)
            term = term.to(device)
            label = label.to(device)
            # Forward pass
            z = encoder.context_encoding(
                states=obs,
                actions=action,
                rewards=reward,
                terms=term
            )
            # print("shape of old_obs: ", old_obs.shape)
            # print("shape of z: ", z.shape)
            # print("shape of labels: ", label.shape)
            # Compute loss
            pos_z_loss, pos_cnt, neg_z_loss, neg_cnt, loss = metric_loss_fast(z, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            # wandb.log({"loss": loss.item()})
            # wandb.log({"pos_z_loss": pos_z_loss.item(), "neg_z_loss": neg_z_loss.item(), "pos_cnt": pos_cnt, "neg_cnt": neg_cnt})
            
        avg_loss = total_loss / len(loader)
        ToolboxRegistry.info(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")
        # wandb.log({"avg_loss": avg_loss, "epoch": epoch+1})
        log_tsne(encoder, loader, device)
    torch.save(encoder.state_dict(), save_path)
    ToolboxRegistry.info(f"Model saved to {save_path}")
    ToolboxRegistry.info("Training finished.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="encoder_data/")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="model/follower-robust/checkpoint/encoder")
    parser.add_argument("--window_size", type=int, default=5)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_path=args.save_path,
        window_size=args.window_size
    )
