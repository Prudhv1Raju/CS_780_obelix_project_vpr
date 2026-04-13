from __future__ import annotations
from typing import List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
OBS_DIM   = 18


class LSTMDQN(nn.Module):
    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        n_actions:  int = N_ACTIONS,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTMCell(128, hidden_dim)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs, hx, cx):
        enc    = self.encoder(obs)
        hx, cx = self.lstm(enc, (hx, cx))
        v      = self.value_stream(hx)
        a      = self.adv_stream(hx)
        q      = v + (a - a.mean(dim=1, keepdim=True))
        return q, hx, cx

    def init_hidden(self, batch_size: int = 1):
        return (
            torch.zeros(batch_size, self.hidden_dim),
            torch.zeros(batch_size, self.hidden_dim),
        )


_model:       Optional[LSTMDQN]      = None
_hx:          Optional[torch.Tensor] = None
_cx:          Optional[torch.Tensor] = None
_needs_reset: bool                   = True


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
    m  = LSTMDQN()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hx, _cx, _needs_reset

    _load_once()

    if _needs_reset or _hx is None:
        _hx, _cx     = _model.init_hidden()
        _needs_reset  = False

    obs_t        = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q, _hx, _cx = _model(obs_t, _hx, _cx)
    best         = int(q.squeeze(0).argmax().item())
    return ACTIONS[best]


def reset():
    global _needs_reset
    _needs_reset = True