import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, sys, time, math
from placer import (parse_kicad_pcb, hpwl, overlap_cost, boundary_cost,
                    total_cost, build_adjacency, spectral_placement, sa_placement)

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except Exception:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = _get_device()

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads, self.d = heads, out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.d))
        self.a_dst = nn.Parameter(torch.randn(heads, self.d))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        N = x.size(0)
        h = self.W(x).view(N, self.heads, self.d)          # (N, H, D)
        src = (h * self.a_src).sum(-1)                       # (N, H)
        dst = (h * self.a_dst).sum(-1)                       # (N, H)
        attn = src.unsqueeze(1) + dst.unsqueeze(0)           # (N, N, H)
        mask = (adj == 0).unsqueeze(-1).expand(-1, -1, self.heads)
        attn = attn.masked_fill(mask, -1e9)
        attn = F.softmax(attn, dim=1)
        out = torch.einsum('ijh,jhd->ihd', attn, h).reshape(N, -1)
        return self.norm(out + self.W(x))

class PlacementGNN(nn.Module):
    def __init__(self, node_dim=8, hidden=64, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.gat_layers = nn.ModuleList([GATLayer(hidden, hidden, heads) for _ in range(layers)])
        self.move_head = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.quality_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, node_feats, adj, positions):
        h = F.relu(self.embed(node_feats))
        for layer in self.gat_layers:
            h = F.relu(layer(h, adj))
        pos_norm = positions / positions.abs().max().clamp(min=1e-6)
        moves = self.move_head(torch.cat([h, pos_norm], dim=-1))
        quality = self.quality_head(h).squeeze(-1)
        return moves, quality

def build_node_features(board, positions):
    n = len(board.components)
    feats = np.zeros((n, 8), dtype=np.float32)
    for i, c in enumerate(board.components):
        feats[i] = [c.width / 10, c.height / 10, len(c.pads) / 20,
                    (positions[i, 0] - board.origin_x) / max(board.width, 1),
                    (positions[i, 1] - board.origin_y) / max(board.height, 1),
                    1.0 if 'F.Cu' in c.layer else 0.0,
                    sum(1 for p in c.pads if p.net != 0) / 10,
                    c.width * c.height / 100]
    return feats

def collect_sa_rollouts(board, positions, n_rollouts=10, T0=10.0, Tf=2.0, alpha=0.97):
    rollouts = []
    n = len(board.components)
    for seed in range(n_rollouts):
        rng = np.random.RandomState(seed)
        pos = positions.copy()
        cost = total_cost(board, pos)
        T = T0
        steps = max(n // 2, 20)
        r_accepted = 0
        r_total = 0
        while T > Tf:
            for _ in range(steps):
                i = rng.randint(n)
                scale = T / T0
                dx = rng.normal(0, board.width * 0.03 * scale)
                dy = rng.normal(0, board.height * 0.03 * scale)
                old_x, old_y = pos[i, 0], pos[i, 1]
                pos[i, 0] += dx
                pos[i, 1] += dy
                new_cost = total_cost(board, pos)
                delta = new_cost - cost
                accepted = delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10))
                r_total += 1
                if accepted:
                    r_accepted += 1
                    rollouts.append({'component': i, 'dx': dx, 'dy': dy, 'delta': delta,
                                     'improvement': max(0, -delta), 'temperature': T,
                                     'accepted': True, 'positions': pos.copy(), 'cost': new_cost})
                    cost = new_cost
                else:
                    pos[i, 0], pos[i, 1] = old_x, old_y
                    rollouts.append({'component': i, 'dx': dx, 'dy': dy, 'delta': delta,
                                     'improvement': 0.0, 'temperature': T,
                                     'accepted': False, 'positions': pos.copy(), 'cost': cost})
            T *= alpha
        print(f"  rollout {seed+1}/{n_rollouts}: {r_total} moves, {r_accepted} accepted ({r_accepted/max(r_total,1)*100:.0f}%), final cost={cost:.1f}")
    return rollouts

def train_gnn(board, positions, epochs=80, lr=5e-4, rollouts_n=10):
    adj = build_adjacency(board)
    adj_t = torch.tensor(adj, dtype=torch.float32).to(device)
    model = PlacementGNN(node_dim=8, hidden=64, heads=4, layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print("collecting SA rollouts...")
    t0 = time.time()
    rollouts = collect_sa_rollouts(board, positions, n_rollouts=rollouts_n)
    print(f"collected {len(rollouts)} rollouts in {time.time()-t0:.1f}s")

    accepted = [r for r in rollouts if r['accepted']]
    if not accepted:
        print("WARNING: no accepted rollouts, returning untrained model")
        return model, {'epoch': [], 'loss': [], 'move_loss': [], 'quality_loss': [], 'quality_r2': [], 'lr': []}

    improvements = np.array([r['improvement'] for r in accepted])
    imp_mean = improvements.mean()
    imp_std = max(improvements.std(), 1e-8)

    history = {'epoch': [], 'loss': [], 'move_loss': [], 'quality_loss': [], 'quality_r2': [], 'lr': []}

    for epoch in range(epochs):
        model.train()
        batch_size = min(256, len(accepted))
        batch = [accepted[i] for i in np.random.choice(len(accepted), batch_size, replace=len(accepted) < batch_size)]

        total_move, total_qual, r2_num, r2_den = 0.0, 0.0, 0.0, 0.0

        for r in batch:
            feats_t = torch.tensor(build_node_features(board, r['positions']), dtype=torch.float32).to(device)
            pos_t = torch.tensor(r['positions'], dtype=torch.float32).to(device)
            moves, quality = model(feats_t, adj_t, pos_t)

            ci = r['component']
            target_move = torch.tensor([r['dx'], r['dy']], dtype=torch.float32).to(device)
            move_loss = F.mse_loss(moves[ci], target_move)

            target_q = (r['improvement'] - imp_mean) / imp_std
            qual_loss = F.mse_loss(quality[ci], torch.tensor(target_q, dtype=torch.float32).to(device))

            loss = move_loss + 0.5 * qual_loss
            opt.zero_grad()
            loss.backward()

            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch}, aborting training")
                return model, history

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_move += move_loss.item()
            total_qual += qual_loss.item()
            r2_num += (quality[ci].item() - target_q) ** 2
            r2_den += target_q ** 2

        scheduler.step()
        avg_move, avg_qual = total_move / batch_size, total_qual / batch_size
        r2 = max(-1.0, min(1.0, 1.0 - r2_num / max(r2_den, 1e-8)))

        history['epoch'].append(epoch)
        history['loss'].append(avg_move + avg_qual)
        history['move_loss'].append(avg_move)
        history['quality_loss'].append(avg_qual)
        history['quality_r2'].append(r2)
        history['lr'].append(scheduler.get_last_lr()[0])

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg_move+avg_qual:.4f}  move={avg_move:.4f}  qual={avg_qual:.4f}  R²={r2:.3f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return model, history

def gnn_guided_move(model, board, adj_t, positions, T, T0, rng):
    feats_t = torch.tensor(build_node_features(board, positions), dtype=torch.float32).to(device)
    pos_t = torch.tensor(positions, dtype=torch.float32).to(device)
    with torch.no_grad():
        moves, quality = model(feats_t, adj_t, pos_t)
    moves, quality = moves.cpu(), quality.cpu()
    probs = F.softmax(quality / max(T / T0, 0.1), dim=0).numpy()
    i = rng.choice(len(board.components), p=probs)
    scale = T / T0
    dx = moves[i, 0].item() * scale + rng.normal(0, 0.2 * scale)
    dy = moves[i, 1].item() * scale + rng.normal(0, 0.2 * scale)
    return i, dx, dy

def run_experiment(board, pcb_name, epochs=80, rollouts_n=10):
    original = np.array([[c.x, c.y] for c in board.components])
    spec_pos = spectral_placement(board)
    adj = build_adjacency(board)
    adj_t = torch.tensor(adj, dtype=torch.float32).to(device)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {pcb_name}")
    print(f"components={len(board.components)} nets={len(board.nets)} board={board.width:.1f}x{board.height:.1f}mm")
    print(f"device={device}")
    print(f"{'='*60}")

    print("\n[1/4] training GNN on SA rollouts...")
    model, train_hist = train_gnn(board, original, epochs=epochs, rollouts_n=rollouts_n)

    print("\n[2/4] baseline SA from original positions...")
    t0 = time.time()
    sa1_pos, sa1_cost, sa1_hist = sa_placement(board, original, T0=15.0, Tf=0.3, alpha=0.995, verbose=True, label="baseline")
    sa1_time = time.time() - t0

    print("\n[3/4] baseline SA from spectral init...")
    t0 = time.time()
    sa2_pos, sa2_cost, sa2_hist = sa_placement(board, spec_pos, T0=15.0, Tf=0.3, alpha=0.995, seed=77, verbose=True, label="spectral")
    sa2_time = time.time() - t0

    T0_gnn = 15.0
    def gnn_move(board, pos, T, rng):
        return gnn_guided_move(model, board, adj_t, pos, T, T0_gnn, rng)

    print("\n[4/4] GNN-guided SA from original positions...")
    t0 = time.time()
    sa3_pos, sa3_cost, sa3_hist = sa_placement(board, original, T0=T0_gnn, Tf=0.3, alpha=0.995, move_fn=gnn_move, seed=42, verbose=True, label="GNN+SA")
    sa3_time = time.time() - t0

    results = {
        'pcb': pcb_name,
        'components': len(board.components),
        'nets': len(board.nets),
        'original_hpwl': hpwl(board, original),
        'original_overlap': overlap_cost(board, original),
        'original_cost': total_cost(board, original),
        'spectral_hpwl': hpwl(board, spec_pos),
        'sa_baseline': {'cost': sa1_cost, 'hpwl': hpwl(board, sa1_pos),
                        'overlap': overlap_cost(board, sa1_pos), 'time': sa1_time, 'history': sa1_hist},
        'sa_spectral': {'cost': sa2_cost, 'hpwl': hpwl(board, sa2_pos),
                        'overlap': overlap_cost(board, sa2_pos), 'time': sa2_time, 'history': sa2_hist},
        'sa_gnn': {'cost': sa3_cost, 'hpwl': hpwl(board, sa3_pos),
                   'overlap': overlap_cost(board, sa3_pos), 'time': sa3_time, 'history': sa3_hist},
        'training': train_hist
    }

    print(f"\n--- RESULTS ({pcb_name}) ---")
    print(f"{'Method':<20} {'Cost':>10} {'HPWL':>10} {'Overlap':>10} {'Time':>8}")
    print("-" * 60)
    for label, c, h, o, t in [
        ('Original', total_cost(board, original), hpwl(board, original), overlap_cost(board, original), None),
        ('SA (baseline)', sa1_cost, hpwl(board, sa1_pos), overlap_cost(board, sa1_pos), sa1_time),
        ('SA (spectral)', sa2_cost, hpwl(board, sa2_pos), overlap_cost(board, sa2_pos), sa2_time),
        ('SA+GNN (ours)', sa3_cost, hpwl(board, sa3_pos), overlap_cost(board, sa3_pos), sa3_time),
    ]:
        ts = f"{t:.1f}s" if t else "--"
        print(f"{label:<20} {c:>10.1f} {h:>10.1f} {o:>10.4f} {ts:>8}")

    return results

if __name__ == "__main__":
    pcb_path = sys.argv[1] if len(sys.argv) > 1 else "data/stickhub.kicad_pcb"
    pcb_name = os.path.basename(pcb_path).replace('.kicad_pcb', '')
    t_start = time.time()
    board = parse_kicad_pcb(pcb_path)
    results = run_experiment(board, pcb_name)
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{pcb_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nsaved {out_path}")
    print(f"total wall time: {time.time()-t_start:.1f}s")
