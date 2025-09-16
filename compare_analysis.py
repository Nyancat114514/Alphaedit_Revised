import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
import seaborn as sns
import sys
import logging
from datetime import datetime


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip() != '':
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

def load_matrix(path: Path):
    """Safely loads a torch tensor from a file."""
    if path.exists():
        return torch.load(path)
    else:
        print(f"Warning: File not found at {path}")
        return None

def compare_chunk_layer(alphaedit_chunk_dir: Path, memit_chunk_dir: Path, layer_id: int, output_base_dir: Path):
    """
    Compares the weight deltas from AlphaEdit and MEMIT for a specific layer in a specific chunk.
    """
    print(f"\n--- Comparing Layer {layer_id} in {alphaedit_chunk_dir.name} vs {memit_chunk_dir.name} ---")

    # Load AlphaEdit and MEMIT deltas
    delta_alpha = load_matrix(alphaedit_chunk_dir / f"layer_{layer_id:02d}_delta_alphaedit.pt")
    delta_memit = load_matrix(memit_chunk_dir / f"layer_{layer_id:02d}_delta_memit.pt")

    if delta_alpha is None or delta_memit is None:
        print("Skipping comparison due to missing delta matrices.")
        return

    # --- 1. Norm Comparison ---
    print("\n--- 1. Norms of Delta Matrices ---")
    norm_fro_alpha = torch.linalg.norm(delta_alpha, 'fro').item()
    norm_fro_memit = torch.linalg.norm(delta_memit, 'fro').item()
    print(f"Frobenius Norm ||Delta_AlphaEdit||_F: {norm_fro_alpha:.4f}")
    print(f"Frobenius Norm ||Delta_MEMIT||_F:   {norm_fro_memit:.4f}")

    norm_l1_alpha = torch.linalg.matrix_norm(delta_alpha, ord=1).item()
    norm_l1_memit = torch.linalg.matrix_norm(delta_memit, ord=1).item()
    print(f"L1 Norm ||Delta_AlphaEdit||_1: {norm_l1_alpha:.4f}")
    print(f"L1 Norm ||Delta_MEMIT||_1:   {norm_l1_memit:.4f}")
    
    norm_l2_alpha = torch.linalg.matrix_norm(delta_alpha, ord=2).item()
    norm_l2_memit = torch.linalg.matrix_norm(delta_memit, ord=2).item()
    print(f"L2 Norm ||Delta_AlphaEdit||_2: {norm_l2_alpha:.4f}")
    print(f"L2 Norm ||Delta_MEMIT||_2:   {norm_l2_memit:.4f}")


    # --- 2. Difference Analysis ---
    print("\n--- 2. Difference Analysis (D = Delta_MEMIT - Delta_AlphaEdit) ---")
    D = delta_memit - delta_alpha
    norm_fro_D = torch.linalg.norm(D, 'fro').item()
    print(f"Frobenius Norm ||D||_F: {norm_fro_D:.4f}")

    # Visualize D as a heatmap
    fig_save_dir = output_base_dir / "comparison_visuals" / f"chunk_{alphaedit_chunk_dir.name.split('_')[-1]}"
    fig_save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sample_D = D[:100, :100] if D.shape[0] > 100 and D.shape[1] > 100 else D
    plt.imshow(sample_D.abs().cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Magnitude of D elements')
    plt.title(f'Heatmap of |D| (Layer {layer_id}, Chunk {alphaedit_chunk_dir.name})')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    fig_save_path = fig_save_dir / f"layer_{layer_id:02d}_D_heatmap.png"
    plt.savefig(fig_save_path)
    print(f"Saved heatmap of D to {fig_save_path}")
    plt.close()

    # Histogram of D's elements
    plt.figure(figsize=(8, 6))
    plt.hist(D.cpu().numpy().flatten(), bins=100, log=True)
    plt.title(f'Element Distribution of D (Layer {layer_id}, Chunk {alphaedit_chunk_dir.name})')
    plt.xlabel('Value')
    plt.ylabel('Frequency (log scale)')
    fig_save_path_hist = fig_save_dir / f"layer_{layer_id:02d}_D_hist.png"
    plt.savefig(fig_save_path_hist)
    print(f"Saved histogram of D to {fig_save_path_hist}")
    plt.close()

    # --- 3. Impact on Preserved Knowledge ---
    print("\n--- 3. Impact on Preserved Knowledge ---")
    global_weights_dir = alphaedit_chunk_dir.parent.parent
    K0K0T_layer = load_matrix(global_weights_dir / "weights" / f"layer_{layer_id:02d}_K0K0T.pt")

    if K0K0T_layer is not None:
        if delta_alpha.shape[1] == K0K0T_layer.shape[0]:
            impact_preserved_alpha = delta_alpha.float() @ K0K0T_layer
            norm_impact_preserved_alpha = torch.linalg.norm(impact_preserved_alpha, 'fro').item()
            print(f"||Delta_AlphaEdit @ K0K0T||_F: {norm_impact_preserved_alpha:.4e}")
        else:
            print(f"Dimension mismatch for Delta_AlphaEdit @ K0K0T: {delta_alpha.shape} vs {K0K0T_layer.shape}")
        
        if delta_memit.shape[1] == K0K0T_layer.shape[0]:
            impact_preserved_memit = delta_memit.float() @ K0K0T_layer
            norm_impact_preserved_memit = torch.linalg.norm(impact_preserved_memit, 'fro').item()
            print(f"||Delta_MEMIT @ K0K0T||_F:   {norm_impact_preserved_memit:.4e}")
        else:
            print(f"Dimension mismatch for Delta_MEMIT @ K0K0T: {delta_memit.shape} vs {K0K0T_layer.shape}")
    else:
        print("K0K0T_layer not found, skipping preserved knowledge impact calculation.")


    # --- 4. Impact on Updated Knowledge ---
    print("\n--- 4. Impact on Updated Knowledge ---")
    # K1 and R should be the same for both methods in the same chunk
    K1 = load_matrix(alphaedit_chunk_dir / f"layer_{layer_id:02d}_K1.pt")
    R_chunk_targets = load_matrix(alphaedit_chunk_dir / f"layer_{layer_id:02d}_R_chunk_targets.pt")

    if K1 is not None and R_chunk_targets is not None:
        if delta_alpha.shape[1] == K1.shape[1]: # d_in
            term_alpha = (delta_alpha.float() @ K1.T) - R_chunk_targets
            norm_sq_updated_alpha = (torch.linalg.norm(term_alpha, 'fro')**2).item()
            print(f"||Delta_AlphaEdit K1 - R||_F^2: {norm_sq_updated_alpha:.4e}")
        else:
            print(f"Dimension mismatch for AlphaEdit update term: Delta {delta_alpha.shape}, K1.T {K1.T.shape}, R {R_chunk_targets.shape}")

        if delta_memit.shape[1] == K1.shape[1]:
            term_memit = (delta_memit.float() @ K1.T) - R_chunk_targets
            norm_sq_updated_memit = (torch.linalg.norm(term_memit, 'fro')**2).item()
            print(f"||Delta_MEMIT K1 - R||_F^2:   {norm_sq_updated_memit:.4e}")
        else:
             print(f"Dimension mismatch for MEMIT update term: Delta {delta_memit.shape}, K1.T {K1.T.shape}, R {R_chunk_targets.shape}")
    else:
        print("K1 or R_chunk_targets not found, skipping updated knowledge impact calculation.")

    return norm_fro_alpha, norm_fro_memit


def main():

    parser = argparse.ArgumentParser(description="Compare AlphaEdit and MEMIT deltas.")
    parser.add_argument("--alphaedit_run_dir", type=str, required=True, help="...")
    parser.add_argument("--memit_run_dir", type=str, required=True, help="...")
    args = parser.parse_args()

    alphaedit_dir = Path(args.alphaedit_run_dir)
    memit_dir = Path(args.memit_run_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    memit_run_name = memit_dir.name

    output_base_dir = alphaedit_dir / f"comparison_with_memit_{memit_run_name}" / f"{timestamp}"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_base_dir / "comparison.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w'
    )

    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
        
    alphaedit_weights_dir = alphaedit_dir / "weights"
    memit_weights_dir = memit_dir / "weights"

    if not alphaedit_weights_dir.exists():
        print(f"AlphaEdit weights directory not found: {alphaedit_weights_dir}")
        return
    if not memit_weights_dir.exists():
        print(f"MEMIT weights directory not found: {memit_weights_dir}")
        return

    # Find common chunks and layers
    alphaedit_chunks = sorted([d for d in alphaedit_weights_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
    memit_chunks = sorted([d for d in memit_weights_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")])

    # Assume chunk names/numbers correspond
    num_chunks = min(len(alphaedit_chunks), len(memit_chunks))
    print(f"Found {num_chunks} common chunks to compare.")
    
    # Determine number of layers from one of the runs
    num_layers = 0
    if alphaedit_chunks:
        for f in (alphaedit_dir / "weights").glob("layer_*_P.pt"):
            try:
                num_layers = max(num_layers, int(f.name.split('_')[1]) + 1)
            except (ValueError, IndexError):
                pass
    print(f"Found {num_layers} layers configured (approx).")


    # Structure: {layer_idx: {'alpha': [norm1, norm2, ...], 'memit': [norm1, norm2, ...]}}
    all_norms_data = {}

    for i in range(num_chunks):
        alpha_chunk_dir = alphaedit_chunks[i]
        memit_chunk_dir = memit_chunks[i]
        
        for layer_idx in range(num_layers):
            # Check if files exist for this layer in both directories before proceeding
            alpha_delta_path = alpha_chunk_dir / f"layer_{layer_idx:02d}_delta_alphaedit.pt"
            memit_delta_path = memit_chunk_dir / f"layer_{layer_idx:02d}_delta_memit.pt"
            
            if alpha_delta_path.exists() and memit_delta_path.exists():
                norm_alpha, norm_memit = compare_chunk_layer(alpha_chunk_dir, memit_chunk_dir, layer_idx, output_base_dir)

                if norm_alpha is not None and norm_memit is not None:
                    if layer_idx not in all_norms_data:
                        all_norms_data[layer_idx] = {'alpha': [], 'memit': []}
                    all_norms_data[layer_idx]['alpha'].append(norm_alpha)
                    all_norms_data[layer_idx]['memit'].append(norm_memit)

            else:
                # Optionally print a message if a layer is skipped
                if not alpha_delta_path.exists():
                    print(f"Skipping layer {layer_idx} in {alpha_chunk_dir.name}: delta file not found.")
                if not memit_delta_path.exists():
                    print(f"Skipping layer {layer_idx} in {memit_chunk_dir.name}: delta file not found.")

    print("\n--- Generating Frobenius Norm Trend Plots ---\n")
    plot_save_dir = output_base_dir / "norm_trend_plots"
    plot_save_dir.mkdir(exist_ok=True)

    for layer_idx, data in sorted(all_norms_data.items()):
        if not data['alpha']: # Skip if no data was collected for this layer
            continue

        num_points = len(data['alpha'])
        chunk_indices = range(num_points)

        plt.figure(figsize=(12, 7))
        plt.plot(chunk_indices, data['alpha'], marker='o', linestyle='-', label='AlphaEdit $||Δ||_F$')
        plt.plot(chunk_indices, data['memit'], marker='x', linestyle='--', label='MEMIT $||Δ||_F$')
        
        plt.title(f'Frobenius Norm Trend for Layer {layer_idx}')
        plt.xlabel('Chunk Index')
        plt.ylabel('Frobenius Norm')
        plt.legend()
        plt.grid(True)
        
        # Make sure x-axis ticks are integers if there are not too many chunks
        if num_points <= 20:
            plt.xticks(chunk_indices)
        
        fig_save_path = plot_save_dir / f"layer_{layer_idx:02d}_norm_trend.png"
        plt.savefig(fig_save_path)
        plt.close() # Close the figure to free up memory
        print(f"Saved norm trend plot for layer {layer_idx} to {fig_save_path}")

if __name__ == "__main__":
    main()