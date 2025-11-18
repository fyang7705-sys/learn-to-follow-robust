
import torch
import torch.nn as nn
from torch.nn import functional as F

from follower_robust.model import FlattenMlp

# permutation invariant task encoder: mlps+average
# if not use average output, do not use context_encoding()
class MLPEncoder(nn.Module):
    def __init__(self,
                 # network size
                 hidden_size=64,
                 num_hidden_layers=2,
                 task_embedding_size=32,
                 # actions, states, rewards
                 action_size=2,
                 state_size=2,
                 reward_size=1,
                 term_size=1,
                 normalize=False
                 ):
        super(MLPEncoder, self).__init__()
        self.task_embedding_size=task_embedding_size
        self.action_size=action_size
        self.state_size=state_size
        self.reward_size=reward_size
        self.term_size=term_size
        self.encoder = FlattenMlp(input_size=state_size*2+action_size+reward_size+term_size,
                                    output_size=task_embedding_size,
                                    hidden_sizes=[hidden_size for i in range(num_hidden_layers)])
        self.normalize=normalize
        self.use_termination = True if term_size else False # if term_size=0, encode (s,a,r,s') only

    # input state transition sample, output task embedding
    def forward(self, obs, action, reward, next_obs, term=None):
        assert obs.shape[1] == self.state_size and action.shape[1] == self.action_size \
            and reward.shape[1] == self.reward_size and next_obs.shape[1] == self.state_size \
            and ((not self.use_termination) or (term.shape[1] == self.term_size))
        out = self.encoder(obs, action, reward, next_obs, term) if self.use_termination \
            else self.encoder(obs, action, reward, next_obs)
        if not self.normalize:
            return out
        else:
            return F.normalize(out)

    # extract task representation from context sequence
    # input size: (timesteps, task, dim)
    # output size: (task, z_dim)
    def context_encoding(self, obs, actions, rewards, next_obs, terms):
        n_timesteps, batch_size, _ = obs.shape
        #print(obs.shape, actions.shape, rewards.shape, next_obs.shape, terms.shape)
        z = self.forward(
                obs.reshape(n_timesteps*batch_size, -1),
                actions.reshape(n_timesteps*batch_size, -1),
                rewards.reshape(n_timesteps*batch_size, -1),
                next_obs.reshape(n_timesteps*batch_size, -1),
                terms.reshape(n_timesteps*batch_size, -1)
            )
        z = z.reshape(n_timesteps, batch_size, -1)
        z = z.mean(0) # average over timesteps
        #print(z.shape)
        return z

class CNNEncoder(nn.Module):
    def __init__(self,
                 hidden_size=64,
                 task_embedding_size=32,
                 action_size=1,
                 reward_size=1,
                 term_size=1,
                 obs_shape=(2, 11, 11),
                 window_size=5,
                 normalize=False,
                 transformer_layers=2,
                 transformer_heads=4):

        super().__init__()
        self.task_embedding_size = task_embedding_size
        self.use_termination = term_size > 0
        self.normalize = normalize
        self.obs_shape = obs_shape
        self.window_size = window_size

        C, H, W = obs_shape

        # ---- CNN ----
        self.conv = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()  # result dimension = 32 * H * W
        )
        self.conv_out_size = 32 * H * W

        # ---- MLP to fuse CNN + action/reward/term ----
        mlp_input_dim = self.conv_out_size + action_size + reward_size + term_size
        self.frame_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=transformer_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,  # important: output shapes are (B,T,D)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        # ---- final projection ----
        self.out_linear = nn.Linear(hidden_size, task_embedding_size)


    def forward(self, states, actions, rewards, terms):
        B, T = states.shape[0], states.shape[1]
        C, H, W = self.obs_shape

        # ----- CNN feature extraction -----
        states_flat = states.reshape(B*T, C, H, W)
        state_feats = self.conv(states_flat)  # (B*T, conv_out_size)
        state_feats = state_feats.reshape(B, T, self.conv_out_size)

        # ----- concat with action/reward/term -----
        per_frame_input = torch.cat([
            state_feats,                # (B,T,F)
            actions,                    # (B,T,1)
            rewards,                    # (B,T,1)
            terms                       # (B,T,1)
        ], dim=-1)  # shape = (B,T,F+3)

        # ----- per-frame MLP -----
        frame_feat = self.frame_mlp(per_frame_input)  # (B,T,H)

        # ----- transformer -----
        # key: transformer expects (B,T,H) if batch_first=True
        trans_out = self.transformer(frame_feat)      # (B,T,H)

        # ----- mean-pooling / last-token / cls-token ----
        z = trans_out.mean(dim=1)                     # (B,H)

        # ----- final projection ----
        z = self.out_linear(z)                        # (B, task_embedding_size)

        return z
    
    def context_encoding(self, states, actions, rewards, terms):
        z = self.forward(states, actions, rewards, terms)
        return z