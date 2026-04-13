from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
 

ACTIONS:    List[str] = ["L45", "L22", "FW", "R22", "R45"]
FW_IDX:     int       = ACTIONS.index("FW")
N_ACTIONS:  int       = len(ACTIONS)
OBS_DIM:    int       = 18
HIDDEN_DIM: int       = 128
 
IR_IDX    = 16   
STUCK_IDX = 17   
 
TEMPERATURE = 0.5  
 
 
class DRQN(nn.Module):
    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        n_actions:  int = N_ACTIONS,
        hidden_dim: int = HIDDEN_DIM,
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
 
 

class HeuristicState:
    def __init__(self):
        self.turn_parity     = 0
        self.turn_steps_left = 0
        self.turn_action     = None
        self.box_confirmed   = False
        self.ir_probe_active = False
        self.fw_after_turn   = 0    
        self.cooldown        = 0     
 
 
def _start_turn(hs: HeuristicState) -> str:
    """Initiate 180° escape. 4 × L45 or R45, alternating. Then 3 forced FW."""
    hs.turn_action     = "L45" if hs.turn_parity == 0 else "R45"
    hs.turn_parity     = 1 - hs.turn_parity
    hs.turn_steps_left = 3
    hs.fw_after_turn   = 3
    hs.cooldown        = 0
    return hs.turn_action
 
 
def heuristic_action(obs: np.ndarray, hs: HeuristicState) -> Optional[str]:
    """
    Returns a hard-override action string or None (let network decide).
 
    Priority:
      1. Finish any in-progress 180° turn.
      2. Forced FW after turn — clear the wall.
      3. Cooldown — ignore stuck briefly after escape sequence.
      4. Box-confirmed mode → force FW; escape if stuck again.
      5. IR=1, not stuck → probe forward (box or wall?).
      6. IR probe active + stuck → wall confirmed, escape.
      7. Stuck without IR → blind wall hit, escape.
      8. IR probe was active, IR dropped, not stuck → box attached, confirm.
    """
    ir    = int(obs[IR_IDX])
    stuck = int(obs[STUCK_IDX])
 
    if hs.turn_steps_left > 0:
        hs.turn_steps_left -= 1
        return hs.turn_action
 

    if hs.fw_after_turn > 0:
        hs.fw_after_turn -= 1
        if hs.fw_after_turn == 0:
            hs.cooldown = 4
        return "FW"
 
    if hs.cooldown > 0:
        hs.cooldown -= 1
        return None
 

    if hs.box_confirmed:
        if stuck:
            hs.box_confirmed = False
            return _start_turn(hs)
        return "FW"
 
    if ir and not stuck:
        hs.ir_probe_active = True
        return "FW"
 

    if hs.ir_probe_active and stuck:
        hs.ir_probe_active = False
        return _start_turn(hs)
 
    if stuck:
        hs.ir_probe_active = False
        return _start_turn(hs)
 

    if hs.ir_probe_active and not stuck and not ir:
        hs.ir_probe_active = False
        hs.box_confirmed   = True
        return "FW"
 
    return None
 
 
_model:    Optional[DRQN]           = None
_hx:       Optional[torch.Tensor]   = None
_cx:       Optional[torch.Tensor]   = None
_hs:       Optional[HeuristicState] = None
_prev_obs: Optional[np.ndarray]     = None
 
 
def _load_once():
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights_m_.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent_template.py. "
            "Train offline and include it in the submission zip."
        )
    m  = DRQN()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m
 
 
def _reset_episode_state():
    """Reset LSTM hidden state and heuristic state for a new episode."""
    global _hx, _cx, _hs
    _hx = torch.zeros(1, HIDDEN_DIM)
    _cx = torch.zeros(1, HIDDEN_DIM)
    _hs = HeuristicState()
 
 
def _is_new_episode(obs: np.ndarray) -> bool:
    """Detect episode reset — all zeros is the typical reset observation."""
    global _prev_obs
    if _prev_obs is None:
        return True
    if np.all(obs == 0):
        return True
    return False
 
 
@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hx, _cx, _hs, _prev_obs
 
    _load_once()
 
    if _is_new_episode(obs):
        _reset_episode_state()
 
    _prev_obs = obs.copy()
 

    override = heuristic_action(obs, _hs)
    if override is not None:

        obs_t        = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        _, _hx, _cx  = _model(obs_t, _hx, _cx)
        _hx = _hx.detach()
        _cx = _cx.detach()
        return override
 

    obs_t        = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q, _hx, _cx  = _model(obs_t, _hx, _cx)
    _hx = _hx.detach()
    
    _cx = _cx.detach()
 
    q_np     = q.squeeze(0).cpu().numpy()
    q_scaled = q_np / TEMPERATURE
    q_scaled -= q_scaled.max()     
    probs    = np.exp(q_scaled)
    probs   /= probs.sum()        
 
    action = int(rng.choice(N_ACTIONS, p=probs))
    return ACTIONS[action]

