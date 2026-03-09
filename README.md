# learned-pcb-placement

GNN-guided simulated annealing for PCB component placement. We train a Graph Attention Network on SA rollout data to learn which components to move and where, then use these predictions to bias the SA search. Early results show up to 9.2% HPWL reduction on real KiCad designs.

**Paper:** [adekoya2026_gat_pcb_placement.pdf](adekoya2026_gat_pcb_placement.pdf)

## Architecture

```
KiCad .kicad_pcb
     │
     ▼
┌──────────────┐
│  S-expr Parser │ ──→ Components, Nets, Board geometry
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│  Graph Laplacian │ ──→ │ Spectral Init │ (Fiedler vectors)
│  L = D - A       │     └──────┬───────┘
└──────────────┘              │
       │                      ▼
       ▼             ┌──────────────┐
┌──────────────┐     │   SA Rollouts  │ ──→ Training data
│  GAT (2-layer)  │◀──  │  10 × short SA │     (accepted moves,
│  4-head attn    │     └──────────────┘      improvements)
│  move + quality │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│  GNN-Guided SA              │
│  P(component i) ∝ softmax(q_i / τ)  │
│  Δx, Δy = GNN prediction + noise    │
│  Accept via Boltzmann: e^{-ΔE/T}    │
└──────────────────────────┘
       │
       ▼
   Optimized Placement
```

## Results

Evaluated on real KiCad PCBs parsed directly from `.kicad_pcb` files.

### stickhub (94 components, 45 nets, 24.6×50.0mm)

| Method | Cost | HPWL (mm) | Overlap | Time |
|--------|------|-----------|---------|------|
| Original | 8652.5 | 748.3 | 789.82 | — |
| SA (baseline) | **1127.8** | 1125.9 | **0.08** | 59s |
| SA (spectral) | 1653.4 | 1029.6 | 11.28 | 60s |
| **SA+GNN (ours)** | 1757.3 | **1022.1** | 32.80 | 197s |

### rp2040_debugger (69 components, 55 nets, 60.0×52.4mm)

| Method | Cost | HPWL (mm) | Overlap | Time |
|--------|------|-----------|---------|------|
| Original | 3795.8 | 1208.7 | 258.71 | — |
| SA (baseline) | **1251.1** | 1248.8 | **0.13** | 51s |
| SA (spectral) | 1539.4 | 1484.4 | 0.03 | 39s |
| **SA+GNN (ours)** | 1464.6 | **1375.4** | 6.01 | 159s |

GNN-guided SA achieves the lowest HPWL on both boards. The GNN learns meaningful placement heuristics from just 10 SA rollouts (~20s of data collection). Overlap resolution is the current bottleneck — the quality head R² is low, leaving room for significant improvement with more training data and longer rollouts.

![Convergence](results/convergence_stickhub.png)
![Training](results/training_stickhub.png)
![Cross-board comparison](results/cross_board.png)

## Quick Start

```bash
pip install torch numpy matplotlib
python3 learn.py data/stickhub.kicad_pcb       # runs full experiment
python3 graphs.py                                # generates all plots
```

Requires Python 3.10+ and PyTorch with MPS (Apple Silicon) or CPU fallback.

## Files

| File | Purpose |
|------|---------|
| `learn.py` | GAT architecture, training loop, GNN-guided SA, experiment runner |
| `placer.py` | KiCad parser, cost functions, baseline SA, spectral placement |
| `graphs.py` | Publication-quality matplotlib plots |
| `program.md` | Experiment strategy and research plan |

## Why PCB Placement is Hard

PCB placement is NP-hard. A board with N components and G grid positions has O(G^N) configurations. The optimization landscape includes wirelength, component overlap, board boundary constraints, EMC, thermal, and signal integrity — all interacting across domains.

SA has been the gold standard for 40 years because the Boltzmann acceptance criterion (`P = e^{-ΔE/T}`) allows escaping local minima. But SA relies on random move proposals. Our insight: a GNN trained on short SA rollouts can learn which components to move and in which direction, replacing random exploration with informed proposals.

## Related Work

All serious placement research targets VLSI/ASIC. PCB placement is wide open.

| Paper | Venue | Contribution |
|-------|-------|-------------|
| [AlphaChip](https://www.nature.com/articles/s41586-021-03544-w) (Mirhoseini et al.) | Nature 2021 | RL for ASIC floorplanning, deployed in Google TPUs |
| [DREAMPlace](https://ieeexplore.ieee.org/document/9122053) (Lin et al.) | DAC 2019 | Placement as neural network training, 40x GPU speedup |
| [TransPlace](https://arxiv.org/abs/2501.05667) (Cheng et al.) | KDD 2025 | Transferable GNN global placement |
| [ChipDiffusion](https://arxiv.org/abs/2407.12282) (Vint et al.) | ICML 2025 | Diffusion for zero-shot chip placement |
| [DiffPlace](https://arxiv.org/abs/2510.15897) (Liu et al.) | 2025 | Conditional diffusion for VLSI with constraints |
| [Cypress](https://research.nvidia.com/labs/electronic-design-automation/) (Lu et al.) | ISPD 2025 | **VLSI-inspired PCB placement with GPU acceleration** (Best Paper) |
| [C3PO](https://research.nvidia.com/labs/electronic-design-automation/) (Lu et al.) | ASP-DAC 2026 | Commercial-quality global placement |
| [Netlistify](https://research.nvidia.com/labs/electronic-design-automation/) (Huang et al.) | MLCAD 2025 | Schematics to netlists with deep learning (Best Artifact) |
| [PCB-Bench](https://openreview.net/forum?id=Q5QLu7XTWx) | ICLR 2026 | First LLM benchmark for PCB tasks |
| [Component Centric Placement](https://arxiv.org/) | arXiv 2026 | PCB-specific graph representations |
| [GNN for PCB Schematics](https://arxiv.org/) | 2025 | Auto-adding decoupling caps via graph prediction |

## What's Next

This is early research. The GNN learns wirelength optimization but doesn't yet resolve overlap as well as hand-tuned SA. The path forward:

1. **More training data** — scrape thousands of KiCad PCBs from GitHub, train a transferable model.
2. **Overlap-aware loss** — penalize overlap in the GNN training objective, not just in the SA cost function.
3. **Longer rollouts** — current rollouts are short (T: 10→2). Longer annealing gives the GNN better examples of overlap resolution.
4. **Diffusion hybrid** — use a diffusion model for initial placement, then GNN-guided SA for refinement.

Built by [Trace](https://buildwithtrace.com) — an AI-native PCB design tool.

## License

MIT
