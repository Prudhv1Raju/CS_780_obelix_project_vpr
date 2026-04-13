from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
OBS_DIM   = 18


class SLAMNet(nn.Module):
    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        n_actions:  int = N_ACTIONS,
        embed_dim:  int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_encoder = nn.Sequential(
            nn.Linear(obs_dim + n_actions, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        self.obs_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
        )

    def forward(self, obs, action_onehot, hx, cx):
        x             = torch.cat([obs, action_onehot], dim=-1)
        encoded       = self.input_encoder(x)
        hx, cx        = self.lstm(encoded, (hx, cx))
        next_obs_pred = self.obs_predictor(hx)
        return hx, cx, next_obs_pred

    def init_hidden(self, batch_size: int = 1):
        return (
            torch.zeros(batch_size, self.hidden_dim),
            torch.zeros(batch_size, self.hidden_dim),
        )


class ActorCritic(nn.Module):
    def __init__(self, hidden_dim: int = 128, n_actions: int = N_ACTIONS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, h):
        f      = self.shared(h)
        logits = self.actor(f)
        value  = self.critic(f).squeeze(-1)
        return logits, value


class SLAMActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        n_actions:  int = N_ACTIONS,
        embed_dim:  int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.slam       = SLAMNet(obs_dim, n_actions, embed_dim, hidden_dim)
        self.policy     = ActorCritic(hidden_dim, n_actions)
        self.n_actions  = n_actions
        self.hidden_dim = hidden_dim

    def step(self, obs, last_action, hx, cx):
        action_onehot = torch.zeros(1, self.n_actions)
        action_onehot[0, last_action] = 1.0
        hx, cx, next_obs_pred = self.slam(obs, action_onehot, hx, cx)
        logits, value         = self.policy(hx)
        return logits, value, hx, cx, next_obs_pred

    def forward_sequence(self, obs_seq, action_seq):
        T  = obs_seq.shape[0]
        hx, cx = self.slam.init_hidden(batch_size=1)
        logits_list, value_list, pred_list = [], [], []
        for t in range(T):
            action_onehot = torch.zeros(1, self.n_actions)
            action_onehot[0, action_seq[t].item()] = 1.0
            obs_t = obs_seq[t].unsqueeze(0)
            hx, cx, next_obs_pred = self.slam(obs_t, action_onehot, hx, cx)
            logits, value         = self.policy(hx)
            logits_list.append(logits)
            value_list.append(value)
            pred_list.append(next_obs_pred)
        return (
            torch.cat(logits_list, dim=0),
            torch.cat(value_list,  dim=0),
            torch.cat(pred_list,   dim=0),
        )

    def init_hidden(self):
        return self.slam.init_hidden(batch_size=1)


# ── Persistent inference state ────────────────────────────────────────────────
_model:       Optional[SLAMActorCritic] = None
_hx:          Optional[torch.Tensor]   = None
_cx:          Optional[torch.Tensor]   = None
_last_action: int                      = 0
_needs_reset: bool                     = True


def _load_once():
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. "
            "Train offline and include it in the submission zip."
        )
    m  = SLAMActorCritic()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hx, _cx, _last_action, _needs_reset

    _load_once()

    # Reset LSTM hidden state at episode start
    if _needs_reset or _hx is None:
        _hx, _cx     = _model.init_hidden()
        _last_action  = 0
        _needs_reset  = False

    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    logits, _, _hx, _cx, _ = _model.step(obs_t, _last_action, _hx, _cx)

    probs        = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
    best         = int(np.argmax(probs))
    _last_action = best

    return ACTIONS[best]


def reset():
    """Call at the start of each episode to clear LSTM memory."""
    global _needs_reset
    _needs_reset = True