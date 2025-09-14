import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from collections import defaultdict

def load_matrix(path: Path):
    if path.exists():
        return torch.load(path)
    else:
        print(f"Warning: File not found at {path}")
        return None

def analyze_chunk_layer(chunk_weights_dir: Path, layer_id: int):
    print(f"\n--- Analyzing Layer {layer_id} in {chunk_weights_dir.name} ---")

    # 加载该层相关的矩阵
    W_orig = load_matrix(chunk_weights_dir / f"layer_{layer_id:02d}_W_orig.pt")
    delta_alpha = load_matrix(chunk_weights_dir / f"layer_{layer_id:02d}_delta_alphaedit.pt")
    W_alpha_applied = load_matrix(chunk_weights_dir / f"layer_{layer_id:02d}_W_alphaedit_applied.pt")
    
    K1 = load_matrix(chunk_weights_dir / f"layer_{layer_id:02d}_K1.pt") # u x d
    R_chunk_targets = load_matrix(chunk_weights_dir / f"layer_{layer_id:02d}_R_chunk_targets.pt") # d x u
    
    # 全局（非 chunk 特定）的 P 和 K0K0T
    global_weights_dir = chunk_weights_dir.parent.parent 
    P_layer = load_matrix(global_weights_dir / "weights" / f"layer_{layer_id:02d}_P.pt")
    K0K0T_layer = load_matrix(global_weights_dir / "weights" / f"layer_{layer_id:02d}_K0K0T.pt")

    # Initialize metrics dictionary
    metrics = {}

    # 1. 计算 Delta_AlphaEdit 的各种范数
    if delta_alpha is not None:
        norm_fro_alpha = torch.linalg.norm(delta_alpha, 'fro').item()
        norm_l1_alpha = torch.linalg.matrix_norm(delta_alpha, ord=1).item()
        norm_l2_alpha = torch.linalg.matrix_norm(delta_alpha, ord=2).item()
        
        metrics['frobenius_norm'] = norm_fro_alpha
        metrics['l1_norm'] = norm_l1_alpha
        metrics['l2_norm'] = norm_l2_alpha
        
        print(f"Frobenius Norm ||Delta_AlphaEdit||_F: {norm_fro_alpha:.4f}")
        print(f"L1 Norm ||Delta_AlphaEdit||_1: {norm_l1_alpha:.4f}")
        print(f"L2 Norm ||Delta_AlphaEdit||_2: {norm_l2_alpha:.4f}")
    else:
        metrics['frobenius_norm'] = float('nan')
        metrics['l1_norm'] = float('nan')
        metrics['l2_norm'] = float('nan')

    # 2. 对 preserved knowledge 的影响
    if delta_alpha is not None and K0K0T_layer is not None:
        if delta_alpha.shape[1] == K0K0T_layer.shape[0]:
            impact_preserved_alpha = delta_alpha.float() @ K0K0T_layer
            norm_impact_preserved_alpha = torch.linalg.norm(impact_preserved_alpha, 'fro').item()
            metrics['preserved_impact'] = norm_impact_preserved_alpha
            print(f"||Delta_AlphaEdit @ K0K0T||_F: {norm_impact_preserved_alpha:.4e}")
        else:
            print(f"Dimension mismatch for Delta_AlphaEdit @ K0K0T: {delta_alpha.shape} vs {K0K0T_layer.shape}")
            metrics['preserved_impact'] = float('nan')
    else:
        print("Delta_alpha or K0K0T_layer not found, skipping preserved knowledge impact calculation.")
        metrics['preserved_impact'] = float('nan')

    # 3. 对 Updated Knowledge 的影响
    if delta_alpha is not None and K1 is not None and R_chunk_targets is not None:
        if delta_alpha.shape[1] == K1.shape[1]:
            term_alpha = (delta_alpha.float() @ K1.T) - R_chunk_targets
            norm_sq_updated_alpha = (torch.linalg.norm(term_alpha, 'fro')**2).item()
            metrics['updated_impact'] = norm_sq_updated_alpha
            print(f"||Delta_AlphaEdit K1 - R||_F^2: {norm_sq_updated_alpha:.4e}")
        else:
            print(f"Dimension mismatch for Delta_AlphaEdit K1 - R: Delta {delta_alpha.shape}, K1.T {K1.T.shape}, R {R_chunk_targets.shape}")
            metrics['updated_impact'] = float('nan')
    else:
        print("K1 or R_chunk_targets not found, skipping updated knowledge impact calculation.")
        metrics['updated_impact'] = float('nan')

    return metrics

def plot_metrics_for_layer(layer_data, layer_id, save_dir):
    """Plot metrics across chunks for a specific layer"""
    chunks = sorted(layer_data.keys())
    
    # Define metrics to plot
    metrics_info = {
        'frobenius_norm': 'Frobenius Norm ||Delta_AlphaEdit||_F',
        'l1_norm': 'L1 Norm ||Delta_AlphaEdit||_1', 
        'l2_norm': 'L2 Norm ||Delta_AlphaEdit||_2',
        'preserved_impact': '||Delta_AlphaEdit @ K0K0T||_F',
        'updated_impact': '||Delta_AlphaEdit K1 - R||_F^2'
    }
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Metrics Evolution Across Chunks - Layer {layer_id}', fontsize=16)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    for idx, (metric_key, metric_title) in enumerate(metrics_info.items()):
        values = [layer_data[chunk][metric_key] for chunk in chunks]
        
        # Filter out NaN values for plotting
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
        valid_chunks = [chunks[i] for i in valid_indices]
        valid_values = [values[i] for i in valid_indices]
        
        axes[idx].plot(valid_chunks, valid_values, 'o-', linewidth=2, markersize=6)
        axes[idx].set_title(metric_title, fontsize=12)
        axes[idx].set_xlabel('Chunk')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(True, alpha=0.3)
        
        # Use log scale for very small values (preserved_impact typically)
        if metric_key == 'preserved_impact' and valid_values:
            if min(valid_values) > 0:
                axes[idx].set_yscale('log')
        
        # Rotate x-axis labels if many chunks
        if len(valid_chunks) > 10:
            axes[idx].tick_params(axis='x', rotation=45)
    
    # Remove the last subplot if we have odd number of metrics
    if len(metrics_info) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    # Save the plot
    save_path = save_dir / f"layer_{layer_id:02d}_metrics_evolution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics plot for layer {layer_id} to {save_path}")
    plt.close()

def save_metrics_data(all_metrics, save_dir):
    """Save all metrics data to JSON file"""
    # Convert to regular dict for JSON serialization
    json_data = {}
    for layer_id, layer_data in all_metrics.items():
        json_data[f"layer_{layer_id:02d}"] = dict(layer_data)
    
    json_path = save_dir / "metrics_data.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved all metrics data to {json_path}")

def create_summary_plots(all_metrics, save_dir):
    """Create summary plots showing trends across all layers"""
    if not all_metrics:
        return
        
    layers = sorted(all_metrics.keys())
    
    # Get all chunk names from first layer (assuming all layers have same chunks)
    first_layer = layers[0]
    chunks = sorted(all_metrics[first_layer].keys())
    
    metrics_info = {
        'frobenius_norm': 'Frobenius Norm ||Delta_AlphaEdit||_F',
        'l1_norm': 'L1 Norm ||Delta_AlphaEdit||_1', 
        'l2_norm': 'L2 Norm ||Delta_AlphaEdit||_2',
        'preserved_impact': '||Delta_AlphaEdit @ K0K0T||_F',
        'updated_impact': '||Delta_AlphaEdit K1 - R||_F^2'
    }
    
    for metric_key, metric_title in metrics_info.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for layer_id in layers:
            if layer_id in all_metrics:
                values = [all_metrics[layer_id][chunk][metric_key] for chunk in chunks]
                # Filter out NaN values
                valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
                valid_chunks = [chunks[i] for i in valid_indices]
                valid_values = [values[i] for i in valid_indices]
                
                if valid_values:  # Only plot if we have valid data
                    ax.plot(valid_chunks, valid_values, 'o-', 
                           label=f'Layer {layer_id}', linewidth=2, markersize=4)
        
        ax.set_title(f'{metric_title} - All Layers', fontsize=14)
        ax.set_xlabel('Chunk')
        ax.set_ylabel('Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Use log scale for preserved_impact
        if metric_key == 'preserved_impact':
            ax.set_yscale('log')
        
        # Rotate x-axis labels if many chunks
        if len(chunks) > 10:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        save_path = save_dir / f"summary_{metric_key}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot for {metric_key} to {save_path}")
        plt.close()

def main_analysis(run_dir_str: str):
    run_dir = Path(run_dir_str)
    if not run_dir.exists() or not (run_dir / "weights").exists():
        print(f"Run directory or weights subdirectory not found: {run_dir}")
        return

    # Create save directory for plots
    plots_dir = run_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)

    # 确定有多少 chunk 和 layer
    num_layers = 0
    for f in (run_dir / "weights").glob("layer_*_P.pt"):
        try:
            num_layers = max(num_layers, int(f.name.split('_')[1]) + 1)
        except:
            pass
    
    chunk_dirs = sorted([d for d in (run_dir / "weights").iterdir() 
                        if d.is_dir() and d.name.startswith("chunk_")])

    print(f"Found {len(chunk_dirs)} chunks and {num_layers} layers configured (approx).")

    # Store all metrics: {layer_id: {chunk_name: {metric: value}}}
    all_metrics = defaultdict(dict)

    for chunk_dir in chunk_dirs:
        chunk_name = chunk_dir.name  # e.g., "chunk_000", "chunk_001", etc.
        
        for layer_idx in range(num_layers):
            # 检查该层是否有文件
            if not (chunk_dir / f"layer_{layer_idx:02d}_W_orig.pt").exists():
                continue
                
            metrics = analyze_chunk_layer(chunk_dir, layer_idx)
            all_metrics[layer_idx][chunk_name] = metrics

    # Create plots for each layer
    for layer_id, layer_data in all_metrics.items():
        if layer_data:  # Only plot if we have data
            plot_metrics_for_layer(layer_data, layer_id, plots_dir)

    # Create summary plots across all layers
    create_summary_plots(all_metrics, plots_dir)
    
    # Save all metrics data to JSON
    save_metrics_data(all_metrics, plots_dir)
    
    print(f"\nAnalysis complete! All plots and data saved to: {plots_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze AlphaEdit and MEMIT deltas with plotting.")
    parser.add_argument("--run_dir", type=str, required=True, 
                       help="Path to the run directory (e.g., results/AlphaEdit/run_000)")
    args = parser.parse_args()
    
    main_analysis(args.run_dir)