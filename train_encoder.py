import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from follower_robust.dataloader import EncodeDataset, prepare_batch_for_encoder
from follower_robust.encoder import CNNEncoderForObs
from follower_robust.loss import metric_loss
from pogema_toolbox.registry import ToolboxRegistry
from datetime import datetime

def train(data_dir, learning_rate, batch_size=32, epochs=20, save_path="model/follower-robust/checkpoint/encoder"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ToolboxRegistry.info(f"Using device: {device}")
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    save_path = os.path.join(save_path, f"encoder_{timestamp}.pt")
    
    dataset = EncodeDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = CNNEncoderForObs(
        obs_shape=(2, 11, 11),
        action_size=1,
        reward_size=1,
        term_size=1,
        task_embedding_size=32
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(epochs):
        encoder.train()
        total_loss = 0

        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress:
            # Prepare batch
            old_obs, actions, rewards, new_obs, terms, labels = prepare_batch_for_encoder(batch)

            # Move to device
            old_obs   = old_obs.to(device)
            new_obs   = new_obs.to(device)
            actions   = actions.to(device)
            rewards   = rewards.to(device)
            terms     = terms.to(device)
            labels    = labels.to(device)

            # Forward pass
            z = encoder.context_encoding(
                obs=old_obs,
                actions=actions,
                rewards=rewards,
                next_obs=new_obs,
                terms=terms
            )

            # Compute loss
            loss = metric_loss(z, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        ToolboxRegistry.info(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")
     
    torch.save(encoder.state_dict(), save_path)
    ToolboxRegistry.info(f"Model saved to {save_path}")
    ToolboxRegistry.info("Training finished.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="encoder_data/")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="model/follower-robust/checkpoint/encoder")

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_path=args.save_path
    )
