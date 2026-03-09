import json, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'baseline': '#2196F3',
    'spectral': '#FF9800',
    'gnn': '#4CAF50',
    'training': '#9C27B0',
    'r2': '#E91E63',
}

def load_results(path):
    with open(path) as f:
        return json.load(f)

def plot_convergence(results, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    pcb = results['pcb']

    for key, label, color in [
        ('sa_baseline', 'SA (random moves)', COLORS['baseline']),
        ('sa_spectral', 'SA (spectral init)', COLORS['spectral']),
        ('sa_gnn', 'SA + GNN (ours)', COLORS['gnn']),
    ]:
        hist = results[key]['history']
        steps = [h[0] for h in hist]
        costs = [h[1] for h in hist]
        bests = [h[2] for h in hist]
        ax1.plot(steps, costs, color=color, alpha=0.3, linewidth=0.8)
        ax1.plot(steps, bests, color=color, label=label, linewidth=1.5)

    ax1.set_xlabel('SA Steps')
    ax1.set_ylabel('Total Cost')
    ax1.set_title('Placement Cost Convergence')
    ax1.legend()
    ax1.set_yscale('log')

    for key, label, color in [
        ('sa_baseline', 'SA (random)', COLORS['baseline']),
        ('sa_spectral', 'SA (spectral)', COLORS['spectral']),
        ('sa_gnn', 'SA+GNN (ours)', COLORS['gnn']),
    ]:
        hist = results[key]['history']
        temps = [h[3] for h in hist]
        bests = [h[2] for h in hist]
        ax2.plot(temps[::-1], bests, color=color, label=label, linewidth=1.5)

    ax2.set_xlabel('Temperature (log)')
    ax2.set_ylabel('Best Cost')
    ax2.set_title('Cost vs Temperature')
    ax2.set_xscale('log')
    ax2.invert_xaxis()
    ax2.legend()

    fig.suptitle(f"{pcb} — {results['components']} components, {results['nets']} nets",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'convergence_{pcb}.png'))
    plt.close()
    print(f"  convergence_{pcb}.png")

def plot_training(results, outdir):
    tr = results['training']
    pcb = results['pcb']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0,0].plot(tr['epoch'], tr['loss'], color=COLORS['training'], linewidth=1.5)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Total Loss')
    axes[0,0].set_title('Training Loss')

    axes[0,1].plot(tr['epoch'], tr['quality_r2'], color=COLORS['r2'], linewidth=1.5)
    axes[0,1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('R²')
    axes[0,1].set_title('Move Quality Prediction R²')
    axes[0,1].set_ylim(-0.5, 1.0)

    axes[1,0].plot(tr['epoch'], tr['move_loss'], color=COLORS['baseline'], linewidth=1.5, label='Move')
    axes[1,0].plot(tr['epoch'], tr['quality_loss'], color=COLORS['gnn'], linewidth=1.5, label='Quality')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].set_title('Loss Components')
    axes[1,0].legend()

    axes[1,1].plot(tr['epoch'], tr['lr'], color='#607D8B', linewidth=1.5)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Learning Rate')
    axes[1,1].set_title('LR Schedule (Cosine Annealing)')

    fig.suptitle(f"GNN Training — {pcb}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'training_{pcb}.png'))
    plt.close()
    print(f"  training_{pcb}.png")

def plot_comparison_bar(results, outdir):
    pcb = results['pcb']
    methods = ['Original', 'SA\n(baseline)', 'SA\n(spectral)', 'SA+GNN\n(ours)']
    hpwls = [
        results['original_hpwl'],
        results['sa_baseline']['hpwl'],
        results['sa_spectral']['hpwl'],
        results['sa_gnn']['hpwl'],
    ]
    costs = [
        results['original_cost'],
        results['sa_baseline']['cost'],
        results['sa_spectral']['cost'],
        results['sa_gnn']['cost'],
    ]
    colors = ['#757575', COLORS['baseline'], COLORS['spectral'], COLORS['gnn']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    bars1 = ax1.bar(methods, hpwls, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('HPWL (mm)')
    ax1.set_title('Half-Perimeter Wirelength')
    for bar, val in zip(bars1, hpwls):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(hpwls)*0.02,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(methods, costs, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Total Cost')
    ax2.set_title('Weighted Cost (WL + Overlap + Boundary)')
    for bar, val in zip(bars2, costs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.02,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle(f"{pcb} — Method Comparison", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'comparison_{pcb}.png'))
    plt.close()
    print(f"  comparison_{pcb}.png")

def plot_cross_board(results_list, outdir):
    """Combined bar chart across multiple boards."""
    n_boards = len(results_list)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bar_w = 0.2
    x = np.arange(n_boards)
    labels = [r['pcb'] for r in results_list]

    for offset, key, label, color in [
        (-1.5, 'sa_baseline', 'SA (baseline)', COLORS['baseline']),
        (-0.5, 'sa_spectral', 'SA (spectral)', COLORS['spectral']),
        (0.5, 'sa_gnn', 'SA+GNN (ours)', COLORS['gnn']),
    ]:
        hpwls = [r[key]['hpwl'] for r in results_list]
        costs = [r[key]['cost'] for r in results_list]
        ax1.bar(x + offset * bar_w, hpwls, bar_w, label=label, color=color, edgecolor='white')
        ax2.bar(x + offset * bar_w, costs, bar_w, label=label, color=color, edgecolor='white')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('HPWL (mm)')
    ax1.set_title('HPWL Comparison')
    ax1.legend()

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Total Cost')
    ax2.set_title('Total Cost Comparison')
    ax2.legend()

    fig.suptitle('Cross-Board Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'cross_board.png'))
    plt.close()
    print(f"  cross_board.png")

def generate_all(results_path, outdir='results'):
    results = load_results(results_path)
    os.makedirs(outdir, exist_ok=True)
    plot_convergence(results, outdir)
    plot_training(results, outdir)
    plot_comparison_bar(results, outdir)

if __name__ == "__main__":
    outdir = 'results'
    os.makedirs(outdir, exist_ok=True)

    result_files = [f for f in os.listdir('results') if f.endswith('.json')]
    if not result_files:
        print("no result JSONs found in results/")
        sys.exit(1)

    all_results = []
    for rf in sorted(result_files):
        path = os.path.join('results', rf)
        r = load_results(path)
        if 'sa_gnn' not in r:
            print(f"skipping {rf} (no GNN experiment data)")
            continue
        print(f"generating plots for {rf}...")
        all_results.append(r)
        plot_convergence(r, outdir)
        plot_training(r, outdir)
        plot_comparison_bar(r, outdir)

    if len(all_results) >= 2:
        print("generating cross-board comparison...")
        # skip baseline.json if it doesn't have the right shape
        experiment_results = [r for r in all_results if 'sa_gnn' in r]
        if len(experiment_results) >= 2:
            plot_cross_board(experiment_results, outdir)

    print(f"\nall plots saved to {outdir}/")
