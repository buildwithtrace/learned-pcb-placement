# program.md — experiment strategy

inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)

## hypothesis

a graph attention network trained on short SA rollouts can learn placement heuristics that improve wirelength optimization on real PCBs, even with minimal training data (10 rollouts per board).

## experiment plan

### phase 1: baselines (done)
- parse real KiCad PCBs (stickhub: 94 components, rp2040: 69 components)
- implement vectorized cost functions (HPWL, overlap, boundary)
- run baseline SA with random moves
- run SA from spectral initialization (graph Laplacian Fiedler vectors)
- record convergence histories for comparison

### phase 2: GNN training + evaluation (done)
- build GAT with 2 layers, 4 heads, 64 hidden dim
- collect 10 SA rollouts per board as training data
- train with MSE loss on (move prediction, quality prediction)
- cosine annealing LR, Adam, 80 epochs
- evaluate GNN-guided SA vs baselines on HPWL, overlap, total cost

### phase 3: analysis (done)
- evaluated on 12 real-world KiCad PCBs (5–94 components)
- GNN achieves lower HPWL on 4/12 boards (stickhub -9.2%, snapvcc -21.6%, pluto_watch -3.4%, tomu -1.4%)
- snapvcc: GNN wins on total cost too (118.2 vs 136.5)
- overlap remains primary bottleneck — quality head R² ≈ 0.001–0.079 across all boards
- diagnosis: **underfitting**, not overfitting. only 10 rollouts per board gives ~10K-25K datapoints. quality signal is too sparse for the network to learn which moves improve vs degrade
- acceptance rates 80-99% (GNN) vs 50-93% (baseline) — network proposes structurally valid moves
- GNN works best on boards with dense netlist connectivity (stickhub, snapvcc)

### phase 4: scale up on Colab (in progress)
- scale training: 50 rollouts × 200 epochs on GPU/TPU (vs 10 × 80 local)
- 5x more training data should address underfitting on quality prediction head
- longer SA rollouts (T: 15→0.1 instead of 10→2) for better overlap resolution examples
- cross-board pre-trained model: train on all 12 boards jointly, evaluate transfer
- add overlap-aware penalty directly to GNN training loss
- try TPU for boards with >50 components where attention matrix gets large
- save results to Google Drive for persistence across sessions

### phase 5: next steps (future work)
- scrape 500+ KiCad PCBs from GitHub for transferable model
- diffusion hybrid: diffusion for initial placement + GNN-guided SA refinement
- benchmark against Cypress GPU-accelerated approach (ISPD 2025 Best Paper)
- integrate into Trace's autoplacer pipeline

## key decisions
- chose GAT over vanilla GCN because attention captures variable net importance
- chose SA augmentation over replacement — SA's convergence guarantees are too valuable to discard
- chose per-board training over pre-training — validates the approach before investing in data collection
- chose MPS (Apple Silicon GPU) for fast local iteration
- kept architecture minimal (2 layers, 64 hidden) to avoid overfitting on small rollout data

## metrics
- HPWL: primary (lower is better)
- overlap: must be zero for production (currently non-zero)
- total cost: weighted combination (HPWL + 10*overlap + 5*boundary)
- quality R²: how well the GNN predicts move improvements
- wall time: practical runtime constraint
