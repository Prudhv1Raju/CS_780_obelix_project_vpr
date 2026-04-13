from __future__ import annotations
import argparse
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
FW_IDX    = ACTIONS.index("FW")   # index 2
N_ACTIONS = len(ACTIONS)
OBS_DIM   = 18


IR_IDX    = 16   # infrared sensor — box directly ahead
STUCK_IDX = 17   # stuck flag — wall or boundary hit

# reward shaping 
NON_FW_STREAK_LIMIT   = 4
NON_FW_STREAK_PENALTY = -10
FW_BONUS              = 0.0

# biased random action probabilities L45, L22, FW, R22, R45
ACTION_PROBS = np.array([0.02, 0.03, 0.9, 0.03, 0.02], dtype=np.float32)


# heuristic state 
class HeuristicState:
    """Carries all mutable heuristic state across steps within one episode."""

    def __init__(self):
        self.turn_parity     = 0     
        self.turn_steps_left = 0     
        self.turn_action     = None  
        self.box_confirmed   = False 
        self.ir_probe_active = False 
        self.fw_after_turn   = 0     
        self.cooldown        = 0     


def _start_turn(hs: HeuristicState) -> str:
    """
    Initiate a 180° escape turn. 4 × L45 or 4 × R45, alternating direction.
    After the 4 turn steps, forces 3 FW steps so robot clears the wall
    before stuck can re-trigger.
    """
    hs.turn_action     = "L45" if hs.turn_parity == 0 else "R45"
    hs.turn_parity     = 1 - hs.turn_parity
    hs.turn_steps_left = 3    
    hs.fw_after_turn   = 3    
    hs.cooldown        = 0
    return hs.turn_action


def heuristic_action(obs: np.ndarray, hs: HeuristicState):
    """
    Returns a hard-override action (str) or None if network should decide.

    Priority order:
      1. Finish any in-progress 180° turn.
      2. Forced FW after turn — clear the wall.
      3. Cooldown — ignore stuck briefly after escape sequence.
      4. Box-confirmed mode → force FW; if stuck again, exit & escape.
      5. IR=1, not stuck → probe forward (box or wall?).
      6. IR probe active + stuck → wall confirmed, escape.
      7. Stuck (blind) → escape.
      8. IR probe was active, IR dropped, not stuck → box attached, confirm.
    """
    ir    = int(obs[IR_IDX])
    stuck = int(obs[STUCK_IDX])

    
    if hs.turn_steps_left > 0:
        hs.turn_steps_left -= 1
        return hs.turn_action

    # 2. forced FW after turn — move away from wall before anything else
    if hs.fw_after_turn > 0:
        hs.fw_after_turn -= 1
        if hs.fw_after_turn == 0:
            hs.cooldown = 4   
        return "FW"

    # 3. cooldown — stuck flag may still linger, ignore it
    if hs.cooldown > 0:
        hs.cooldown -= 1
        return None   

    # 4. box-confirmed drive-to-boundary mode
    if hs.box_confirmed:
        if stuck:
            hs.box_confirmed = False
            return _start_turn(hs)
        return "FW"

    # 5. IR=1, not stuck → probe forward
    if ir and not stuck:
        hs.ir_probe_active = True
        return "FW"

    # 6. IR probe active + now stuck → wall confirmed
    if hs.ir_probe_active and stuck:
        hs.ir_probe_active = False
        return _start_turn(hs)

    # 7. stuck without IR → blind wall hit
    if stuck:
        hs.ir_probe_active = False
        return _start_turn(hs)

    # 8. IR probe was active, IR dropped, not stuck → box attached
    if hs.ir_probe_active and not stuck and not ir:
        hs.ir_probe_active = False
        hs.box_confirmed   = True
        return "FW"

    return None   


class DRQN(nn.Module):
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



@dataclass
class Episode:
    obs:     List[np.ndarray] = field(default_factory=list)
    actions: List[int]        = field(default_factory=list)
    rewards: List[float]      = field(default_factory=list)
    dones:   List[bool]       = field(default_factory=list)



class SequencePER:
    def __init__(
        self,
        cap:     int   = 5_000,
        seq_len: int   = 80,
        alpha:   float = 0.6,
    ):
        self.cap     = cap
        self.seq_len = seq_len
        self.alpha   = alpha
        self.buf:        List[Tuple] = []
        self.priorities: np.ndarray  = np.zeros(cap, dtype=np.float32)
        self.pos = 0

    def _extract_sequences(self, episode: Episode):
        total = len(episode.obs)
        seqs  = []
        start = 0
        while start + self.seq_len <= total:
            seqs.append((episode, start))
            start += self.seq_len
        if start < total:
            seqs.append((episode, start))
        return seqs

    def add_episode(self, episode: Episode):
        seqs = self._extract_sequences(episode)
        for seq in seqs:
            max_prio = self.priorities.max() if self.buf else 1.0
            if len(self.buf) < self.cap:
                self.buf.append(seq)
            else:
                self.buf[self.pos] = seq
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.cap

    def sample(self, batch: int, beta: float = 0.4):
        n     = len(self.buf)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs /= probs.sum()
        idx     = np.random.choice(n, batch, p=probs)
        items   = [self.buf[i] for i in idx]
        weights = (n * probs[idx]) ** (-beta)
        weights /= weights.max()
        return items, idx, weights.astype(np.float32)

    def update_priorities(self, idx: np.ndarray, prios: np.ndarray):
        for i, p in zip(idx, prios):
            self.priorities[i] = float(p) + 1e-6

    def __len__(self):
        return len(self.buf)



def process_sequence(item, net, tgt, gamma, seq_len, hidden_dim):
    episode, start = item
    total = len(episode.obs)
    end   = min(start + seq_len, total)


    hx     = torch.zeros(1, hidden_dim)
    cx     = torch.zeros(1, hidden_dim)
    hx_tgt = torch.zeros(1, hidden_dim)
    cx_tgt = torch.zeros(1, hidden_dim)

    preds, targets, td_errors = [], [], []

    for t in range(start, end):
        if t >= total - 1:
            break

        obs_t  = torch.tensor(episode.obs[t],     dtype=torch.float32).unsqueeze(0)
        obs_t1 = torch.tensor(episode.obs[t + 1], dtype=torch.float32).unsqueeze(0)
        a_t    = episode.actions[t]
        r_t    = episode.rewards[t]
        done_t = episode.dones[t]

        q,             hx,     cx     = net(obs_t,  hx,     cx)
        with torch.no_grad():
            q_next_online, _,      _       = net(obs_t1, hx.detach(), cx.detach())
            q_next_tgt,    hx_tgt, cx_tgt  = tgt(obs_t1, hx_tgt,      cx_tgt)

        next_a   = q_next_online.argmax(dim=1)
        next_val = q_next_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
        y        = r_t + gamma * (1.0 - float(done_t)) * next_val.item()

        pred = q[0, a_t]
        preds.append(pred.unsqueeze(0))
        targets.append(torch.tensor([y], dtype=torch.float32))
        td_errors.append(abs(pred.item() - y))

    if not preds:
        return None, None, None

    return torch.cat(preds), torch.cat(targets), float(np.mean(td_errors))



def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights.pth")
    ap.add_argument("--episodes",       type=int,   default=2000)
    ap.add_argument("--max_steps",      type=int,   default=2000)
    ap.add_argument("--difficulty",     type=int,   default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed",      type=int,   default=2)
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--arena_size",     type=int,   default=500)
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lr",             type=float, default=1e-4)
    ap.add_argument("--batch",          type=int,   default=32)
    ap.add_argument("--replay_cap",     type=int,   default=5_000)
    ap.add_argument("--warmup",         type=int,   default=200)
    ap.add_argument("--tau",            type=float, default=0.005)
    ap.add_argument("--seq_len",        type=int,   default=80)
    ap.add_argument("--beta",           type=float, default=0.4)
    ap.add_argument("--eps",            type=float, default=0.7)
    ap.add_argument("--hidden_dim",     type=int,   default=128)
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    net = DRQN(hidden_dim=args.hidden_dim)
    tgt = DRQN(hidden_dim=args.hidden_dim)
    tgt.load_state_dict(net.state_dict())
    tgt.eval()

    opt    = optim.Adam(net.parameters(), lr=args.lr)
    replay = SequencePER(
        cap=args.replay_cap,
        seq_len=args.seq_len,
    )

    start_time = time.time()

    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )

        obs    = env.reset()
        hx, cx = net.init_hidden()

        episode       = Episode()
        hs            = HeuristicState()
        ep_real_ret   = 0.0
        non_fw_streak = 0
        success       = False

        for step in range(args.max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                q, hx, cx = net(obs_t, hx, cx)

            hx = hx.detach()
            cx = cx.detach()

            
            override = heuristic_action(obs, hs)

            if override is not None:
                
                action = ACTIONS.index(override)
            elif random.random() < args.eps:
                
                action = int(np.random.choice(N_ACTIONS, p=ACTION_PROBS))
            else:
            
                action = int(q.squeeze(0).argmax().item())

            obs2, r, done = env.step(ACTIONS[action], render=False)

            ep_real_ret += float(r)

            
            if done and float(r) >= 1799.0:
                success = True

            #  reward shaping 
            shaped_r = float(r)
            if action == FW_IDX:
                shaped_r      += FW_BONUS
                non_fw_streak  = 0
            else:
                non_fw_streak += 1
                if non_fw_streak >= NON_FW_STREAK_LIMIT:
                    shaped_r      += NON_FW_STREAK_PENALTY
                    non_fw_streak  = 0

            episode.obs.append(obs.copy())
            episode.actions.append(action)
            episode.rewards.append(shaped_r)
            episode.dones.append(bool(done))

            obs = obs2
            if done:
                break

        episode.obs.append(obs.copy())
        replay.add_episode(episode)

        # training step 
        if len(replay) >= args.warmup:
            items, idx, weights = replay.sample(args.batch, args.beta)

            all_preds, all_targets, all_td = [], [], []
            valid_idx, valid_w = [], []

            for i, item in enumerate(items):
                preds_t, targets_t, td_mean = process_sequence(
                    item, net, tgt,
                    args.gamma, args.seq_len, args.hidden_dim
                )
                if preds_t is None:
                    continue
                all_preds.append(preds_t)
                all_targets.append(targets_t)
                all_td.append(td_mean)
                valid_idx.append(idx[i])
                valid_w.append(weights[i])

            if all_preds:
                preds_cat   = torch.cat(all_preds)
                targets_cat = torch.cat(all_targets)
                w_t = torch.tensor(valid_w, dtype=torch.float32).repeat_interleave(
                    torch.tensor([len(p) for p in all_preds])
                )

                loss = (w_t * nn.functional.smooth_l1_loss(
                    preds_cat, targets_cat, reduction="none"
                )).mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()

                replay.update_priorities(
                    np.array(valid_idx), np.array(all_td)
                )

                for tp, qp in zip(tgt.parameters(), net.parameters()):
                    tp.data.copy_(args.tau * qp.data + (1 - args.tau) * tp.data)

        result = "SUCCESS" if success else "failed"
        print(
            f"Ep {ep+1}/{args.episodes}  "
            f"real_ret={ep_real_ret:.1f}  "
            f"steps={step+1}  "
            f"replay={len(replay)}  "
            f"t={time.time()-start_time:.1f}s  "
            f"[{result}]"
        )

    torch.save(net.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()