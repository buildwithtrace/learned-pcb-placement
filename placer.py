import re, math, random, json, sys, os, time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

@dataclass
class Pad:
    x: float
    y: float
    net: int

@dataclass
class Component:
    ref: str
    footprint: str
    x: float
    y: float
    rotation: float
    width: float
    height: float
    pads: List[Pad] = field(default_factory=list)
    layer: str = "F.Cu"

@dataclass
class Net:
    idx: int
    name: str
    pad_refs: List[Tuple[int, int]] = field(default_factory=list)

@dataclass
class Board:
    components: List[Component]
    nets: List[Net]
    width: float
    height: float
    origin_x: float
    origin_y: float

def tokenize(s):
    tokens, i, n = [], 0, len(s)
    while i < n:
        if s[i] in ' \t\n\r':
            i += 1
        elif s[i] == '(':
            tokens.append('(')
            i += 1
        elif s[i] == ')':
            tokens.append(')')
            i += 1
        elif s[i] == '"':
            j = i + 1
            while j < n and s[j] != '"':
                j += 1
            tokens.append(s[i+1:j])
            i = j + 1
        else:
            j = i
            while j < n and s[j] not in ' \t\n\r()':
                j += 1
            tokens.append(s[j-j+i:j] if False else s[i:j])
            i = j
    return tokens

def parse_sexp(tokens, pos=0):
    if tokens[pos] != '(':
        return tokens[pos], pos + 1
    pos += 1
    lst = []
    while tokens[pos] != ')':
        val, pos = parse_sexp(tokens, pos)
        lst.append(val)
    return lst, pos + 1

def find_nodes(tree, tag):
    results = []
    if isinstance(tree, list) and len(tree) > 0 and tree[0] == tag:
        results.append(tree)
    if isinstance(tree, list):
        for child in tree:
            results.extend(find_nodes(child, tag))
    return results

def find_first(tree, tag):
    if isinstance(tree, list) and len(tree) > 0 and tree[0] == tag:
        return tree
    if isinstance(tree, list):
        for child in tree:
            r = find_first(child, tag)
            if r is not None:
                return r
    return None

def parse_kicad_pcb(path: str) -> Board:
    with open(path, 'r', errors='replace') as f:
        raw = f.read()
    toks = tokenize(raw)
    tree, _ = parse_sexp(toks, 0)

    net_map = {}
    for node in find_nodes(tree, 'net'):
        if len(node) >= 3 and isinstance(node[1], str) and node[1].isdigit():
            net_map[int(node[1])] = Net(idx=int(node[1]), name=node[2])

    components = []
    # KiCad v6+ uses 'footprint', v5 and earlier use 'module'
    footprints = find_nodes(tree, 'footprint') + find_nodes(tree, 'module')
    for fp in footprints:
        if len(fp) < 2:
            continue
        at_node = find_first(fp, 'at')
        if at_node is None or len(at_node) < 3:
            continue
        is_top_level = False
        for child in fp:
            if isinstance(child, list) and len(child) > 0 and child[0] == 'layer':
                is_top_level = True
                break
        if not is_top_level:
            continue

        try:
            cx, cy = float(at_node[1]), float(at_node[2])
        except (ValueError, IndexError):
            continue
        rot = float(at_node[3]) if len(at_node) > 3 else 0.0

        ref_node = None
        for child in fp:
            if isinstance(child, list) and len(child) > 0 and child[0] == 'property':
                if len(child) > 2 and child[1] == 'Reference':
                    ref_node = child[2]
                    break
        if ref_node is None:
            for child in fp:
                if isinstance(child, list) and len(child) > 0 and child[0] == 'fp_text':
                    if len(child) > 2 and child[1] == 'reference':
                        ref_node = child[2]
                        break
        ref = ref_node if ref_node else f"UNK{len(components)}"

        layer_node = find_first(fp, 'layer')
        layer = layer_node[1] if layer_node and len(layer_node) > 1 else "F.Cu"

        pads = []
        xs, ys = [0.0], [0.0]
        for pad_node in find_nodes(fp, 'pad'):
            pad_at = find_first(pad_node, 'at')
            pad_net = find_first(pad_node, 'net')
            pad_size = find_first(pad_node, 'size')
            px = float(pad_at[1]) if pad_at and len(pad_at) > 1 else 0.0
            py = float(pad_at[2]) if pad_at and len(pad_at) > 2 else 0.0
            net_idx = int(pad_net[1]) if pad_net and len(pad_net) > 1 else 0
            pads.append(Pad(px, py, net_idx))
            if pad_size and len(pad_size) > 2:
                hw = float(pad_size[1]) / 2
                hh = float(pad_size[2]) / 2
                xs.extend([px - hw, px + hw])
                ys.extend([py - hh, py + hh])

        cyard = find_nodes(fp, 'fp_rect') + find_nodes(fp, 'fp_line')
        for shape in cyard:
            layer_s = find_first(shape, 'layer')
            if layer_s and len(layer_s) > 1 and 'CrtYd' in str(layer_s[1]):
                start = find_first(shape, 'start')
                end = find_first(shape, 'end')
                if start and end and len(start) > 2 and len(end) > 2:
                    xs.extend([float(start[1]), float(end[1])])
                    ys.extend([float(start[2]), float(end[2])])

        w = max(xs) - min(xs) if len(xs) > 1 else 1.0
        h = max(ys) - min(ys) if len(ys) > 1 else 1.0
        w = max(w, 0.5)
        h = max(h, 0.5)

        comp = Component(ref=ref, footprint=fp[1] if len(fp) > 1 else "",
                         x=cx, y=cy, rotation=rot, width=w, height=h,
                         pads=pads, layer=layer)
        components.append(comp)

    for ci, comp in enumerate(components):
        for pi, pad in enumerate(comp.pads):
            if pad.net in net_map:
                net_map[pad.net].pad_refs.append((ci, pi))

    all_x = [c.x for c in components]
    all_y = [c.y for c in components]
    margin = 5.0
    ox = min(all_x) - margin if all_x else 0
    oy = min(all_y) - margin if all_y else 0
    bw = (max(all_x) - min(all_x) + 2 * margin) if all_x else 100
    bh = (max(all_y) - min(all_y) + 2 * margin) if all_y else 100

    nets = [n for n in net_map.values() if len(n.pad_refs) >= 2 and n.idx != 0]
    return Board(components=components, nets=nets, width=bw, height=bh,
                 origin_x=ox, origin_y=oy)


def hpwl(board: Board, positions: np.ndarray) -> float:
    total = 0.0
    for net in board.nets:
        if len(net.pad_refs) < 2:
            continue
        coords = np.array([[positions[ci, 0] + board.components[ci].pads[pi].x,
                            positions[ci, 1] + board.components[ci].pads[pi].y]
                           for ci, pi in net.pad_refs])
        total += (coords[:, 0].max() - coords[:, 0].min()) + (coords[:, 1].max() - coords[:, 1].min())
    return total

def overlap_cost(board: Board, positions: np.ndarray) -> float:
    n = len(board.components)
    if n < 2:
        return 0.0
    ws = np.array([c.width for c in board.components])
    hs = np.array([c.height for c in board.components])
    total = 0.0
    for i in range(n):
        dx = np.abs(positions[i, 0] - positions[i+1:, 0])
        dy = np.abs(positions[i, 1] - positions[i+1:, 1])
        ox = (ws[i] + ws[i+1:]) / 2 - dx
        oy = (hs[i] + hs[i+1:]) / 2 - dy
        mask = (ox > 0) & (oy > 0)
        total += np.sum(ox[mask] * oy[mask])
    return total

def boundary_cost(board: Board, positions: np.ndarray) -> float:
    ws = np.array([c.width for c in board.components]) / 2
    hs = np.array([c.height for c in board.components]) / 2
    left = np.maximum(0, board.origin_x - (positions[:, 0] - ws))
    right = np.maximum(0, (positions[:, 0] + ws) - (board.origin_x + board.width))
    top = np.maximum(0, board.origin_y - (positions[:, 1] - hs))
    bottom = np.maximum(0, (positions[:, 1] + hs) - (board.origin_y + board.height))
    return float(np.sum(left**2 + right**2 + top**2 + bottom**2))

def total_cost(board, positions, w_wl=1.0, w_ov=10.0, w_bd=5.0):
    return (w_wl * hpwl(board, positions) +
            w_ov * overlap_cost(board, positions) +
            w_bd * boundary_cost(board, positions))


def delta_overlap(board, positions, idx, old_x, old_y, new_x, new_y):
    n = len(board.components)
    ws = np.array([c.width for c in board.components])
    hs = np.array([c.height for c in board.components])
    others = np.arange(n) != idx
    pos_o = positions[others]
    w_i, h_i = ws[idx], hs[idx]
    w_o, h_o = ws[others], hs[others]

    dx_old = np.abs(old_x - pos_o[:, 0])
    dy_old = np.abs(old_y - pos_o[:, 1])
    ox_old = np.maximum(0, (w_i + w_o) / 2 - dx_old)
    oy_old = np.maximum(0, (h_i + h_o) / 2 - dy_old)
    old_ov = np.sum(ox_old * oy_old)

    dx_new = np.abs(new_x - pos_o[:, 0])
    dy_new = np.abs(new_y - pos_o[:, 1])
    ox_new = np.maximum(0, (w_i + w_o) / 2 - dx_new)
    oy_new = np.maximum(0, (h_i + h_o) / 2 - dy_new)
    new_ov = np.sum(ox_new * oy_new)

    return new_ov - old_ov

def sa_placement(board: Board, positions: np.ndarray, T0=50.0, Tf=0.01,
                 alpha=0.995, steps_per_temp=None, move_fn=None, seed=42,
                 verbose=False, label="SA"):
    rng = np.random.RandomState(seed)
    n = len(board.components)
    if steps_per_temp is None:
        steps_per_temp = max(n, 50)
    pos = positions.copy()
    cost = total_cost(board, pos)
    best_pos, best_cost = pos.copy(), cost
    T = T0
    history = [(0, cost, best_cost, T)]
    step = 0
    recompute_interval = steps_per_temp * 10
    # total iterations for progress tracking
    total_iters = int(math.log(Tf / T0) / math.log(alpha)) + 1
    iter_count = 0
    while T > Tf:
        accepted = 0
        for _ in range(steps_per_temp):
            step += 1
            if move_fn is not None:
                i, dx, dy = move_fn(board, pos, T, rng)
            else:
                i = rng.randint(n)
                scale = T / T0
                dx = rng.normal(0, board.width * 0.03 * scale)
                dy = rng.normal(0, board.height * 0.03 * scale)

            old_x, old_y = pos[i, 0], pos[i, 1]
            new_x, new_y = old_x + dx, old_y + dy
            d_ov = delta_overlap(board, pos, i, old_x, old_y, new_x, new_y)
            pos[i, 0], pos[i, 1] = new_x, new_y
            new_cost = total_cost(board, pos)
            delta = new_cost - cost

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                cost = new_cost
                accepted += 1
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos.copy()
            else:
                pos[i, 0], pos[i, 1] = old_x, old_y

        history.append((step, cost, best_cost, T))
        iter_count += 1
        # verbose progress every 50 temperature steps
        if verbose and (iter_count % 50 == 0 or T * alpha <= Tf):
            pct = iter_count / max(total_iters, 1) * 100
            acc_rate = accepted / steps_per_temp * 100
            print(f"  [{label}] T={T:.3f} cost={cost:.1f} best={best_cost:.1f} accept={acc_rate:.0f}% ({pct:.0f}%)")
        T *= alpha

    return best_pos, best_cost, history


def build_adjacency(board: Board) -> np.ndarray:
    n = len(board.components)
    adj = np.zeros((n, n), dtype=np.float32)
    for net in board.nets:
        comps = list(set(ci for ci, _ in net.pad_refs))
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                adj[comps[i], comps[j]] += 1.0
                adj[comps[j], comps[i]] += 1.0
    return adj

def build_laplacian(adj: np.ndarray) -> np.ndarray:
    D = np.diag(adj.sum(axis=1))
    return D - adj

def spectral_placement(board: Board) -> np.ndarray:
    adj = build_adjacency(board)
    L = build_laplacian(adj)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_x = eigenvectors[:, 1]
    fiedler_y = eigenvectors[:, 2] if eigenvectors.shape[1] > 2 else eigenvectors[:, 1]
    x_min, x_max = fiedler_x.min(), fiedler_x.max()
    y_min, y_max = fiedler_y.min(), fiedler_y.max()
    rx = (board.width * 0.8) / max(x_max - x_min, 1e-6)
    ry = (board.height * 0.8) / max(y_max - y_min, 1e-6)
    positions = np.zeros((len(board.components), 2))
    cx = board.origin_x + board.width / 2
    cy = board.origin_y + board.height / 2
    for i in range(len(board.components)):
        positions[i, 0] = cx + (fiedler_x[i] - (x_max + x_min) / 2) * rx
        positions[i, 1] = cy + (fiedler_y[i] - (y_max + y_min) / 2) * ry
    return positions


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <path.kicad_pcb>")
        sys.exit(1)

    board = parse_kicad_pcb(sys.argv[1])
    print(f"parsed: {len(board.components)} components, {len(board.nets)} nets")
    print(f"board: {board.width:.1f} x {board.height:.1f} mm, origin ({board.origin_x:.1f}, {board.origin_y:.1f})")

    original = np.array([[c.x, c.y] for c in board.components])
    print(f"\noriginal HPWL: {hpwl(board, original):.2f}")
    print(f"original overlap: {overlap_cost(board, original):.4f}")
    print(f"original total cost: {total_cost(board, original):.2f}")

    print("\n--- spectral placement (graph Laplacian) ---")
    spec_pos = spectral_placement(board)
    print(f"spectral HPWL: {hpwl(board, spec_pos):.2f}")
    print(f"spectral overlap: {overlap_cost(board, spec_pos):.4f}")

    print("\n--- SA from original positions ---")
    t0 = time.time()
    sa_pos, sa_cost, sa_hist = sa_placement(board, original, T0=20.0, Tf=0.1, alpha=0.993)
    print(f"SA time: {time.time()-t0:.1f}s")
    print(f"SA final cost: {sa_cost:.2f}")
    print(f"SA HPWL: {hpwl(board, sa_pos):.2f}")
    print(f"SA overlap: {overlap_cost(board, sa_pos):.4f}")

    print("\n--- SA from spectral init ---")
    t0 = time.time()
    sa2_pos, sa2_cost, sa2_hist = sa_placement(board, spec_pos, T0=20.0, Tf=0.1, alpha=0.993, seed=123)
    print(f"SA+spectral time: {time.time()-t0:.1f}s")
    print(f"SA+spectral cost: {sa2_cost:.2f}")
    print(f"SA+spectral HPWL: {hpwl(board, sa2_pos):.2f}")
    print(f"SA+spectral overlap: {overlap_cost(board, sa2_pos):.4f}")

    results = {
        "components": len(board.components),
        "nets": len(board.nets),
        "original_hpwl": hpwl(board, original),
        "spectral_hpwl": hpwl(board, spec_pos),
        "sa_original_cost": sa_cost,
        "sa_spectral_cost": sa2_cost,
        "sa_original_hpwl": hpwl(board, sa_pos),
        "sa_spectral_hpwl": hpwl(board, sa2_pos),
        "sa_history": [(s, c, b, t) for s, c, b, t in sa_hist],
        "sa_spectral_history": [(s, c, b, t) for s, c, b, t in sa2_hist],
    }
    os.makedirs("results", exist_ok=True)
    with open("results/baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nresults saved to results/baseline.json")
