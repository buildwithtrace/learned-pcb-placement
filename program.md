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

### phase 3: analysis
- key finding: GNN achieves lowest HPWL on both boards
- stickhub: 1022.1mm HPWL vs 1125.9mm baseline (-9.2%)
- rp2040: 1375.4mm HPWL vs 1484.4mm spectral (-7.3%)
- trade-off: GNN has higher overlap (quality head R² is low, ~0.006-0.079)
- acceptance rates much higher with GNN (86-99% vs 64-93%)

### phase 4: next steps (future work)
- train transferable model on 1000+ scraped KiCad designs
- add overlap penalty to GNN training loss
- explore diffusion model for initial placement + GNN-guided SA refinement
- benchmark against Cypress GPU-accelerated approach
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
