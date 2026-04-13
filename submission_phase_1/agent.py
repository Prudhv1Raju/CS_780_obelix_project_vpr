from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


class DQN(nn.Module):
    def __init__(self, in_dim: int = 18, n_actions: int = 5):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))


_model: Optional[DQN] = None


def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    m = DQN()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    x      = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits = _model(x).squeeze(0)
    probs  = torch.softmax(logits, dim=-1).cpu().numpy()
    best   = int(np.argmax(probs))      # greedy — same as argmax of raw logits
    return ACTIONS[best]