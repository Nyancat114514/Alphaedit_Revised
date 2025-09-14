import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
import seaborn as sns

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
    # delta_memit = load_matrix(chunk_weights_dir / f"layer_{layer_id:02d}_delta_memit.pt")
    W_alpha_applied = load_matrix(chunk_weights_dir / f"layer_{layer_id:02d}_W_alphaedit_applied.pt")
    
    K1 = load_matrix(chunk_weights_dir / f"layer_{layer_id:02d}_K1.pt") # u x d
    R_chunk_targets = load_matrix(chunk_weights_dir / f"layer_{layer_id:02d}_R_chunk_targets.pt") # d x u
    
    # 全局（非 chunk 特定）的 P 和 K0K0T
    global_weights_dir = chunk_weights_dir.parent.parent 
    P_layer = load_matrix(global_weights_dir / "weights" / f"layer_{layer_id:02d}_P.pt")
    K0K0T_layer = load_matrix(global_weights_dir / "weights" / f"layer_{layer_id:02d}_K0K0T.pt")

    # if delta_alpha is None or delta_memit is None:
    #     print("Skipping norm calculations due to missing delta matrices.")
    #     return

    # 1. 计算 Delta_MEMIT 和 Delta_AlphaEdit 的 Frobenius norm (或其他 matrix norms)
    norm_fro_alpha = torch.linalg.norm(delta_alpha, 'fro').item() if delta_alpha is not None else float('nan')
    # norm_fro_memit = torch.linalg.norm(delta_memit, 'fro').item() if delta_memit is not None else float('nan')
    print(f"Frobenius Norm ||Delta_AlphaEdit||_F: {norm_fro_alpha:.4f}")
    # print(f"Frobenius Norm ||Delta_MEMIT||_F:   {norm_fro_memit:.4f}")
    
    # 还可以计算 L1, L2 范数
    norm_l1_alpha = torch.linalg.matrix_norm(delta_alpha, ord=1).item() if delta_alpha is not None else float('nan')
    # norm_l1_memit = torch.linalg.matrix_norm(delta_memit, ord=1).item() if delta_memit is not None else float('nan')
    norm_l2_alpha = torch.linalg.matrix_norm(delta_alpha, ord=2).item() if delta_alpha is not None else float('nan')
    # norm_l2_memit = torch.linalg.matrix_norm(delta_memit, ord=2).item() if delta_memit is not None else float('nan')
    print(f"L1 Norm ||Delta_AlphaEdit||_1: {norm_l1_alpha:.4f}")
    # print(f"L1 Norm ||Delta_MEMIT||_1:   {norm_l1_memit:.4f}")
    print(f"L2 Norm ||Delta_AlphaEdit||_2: {norm_l2_alpha:.4f}")
    # print(f"L2 Norm ||Delta_MEMIT||_2:   {norm_l2_memit:.4f}")

    '''
    # 2. 计算 D = Delta_MEMIT - Delta_AlphaEdit
    D = delta_memit - delta_alpha
    norm_fro_D = torch.linalg.norm(D, 'fro').item()
    print(f"Frobenius Norm ||D = Delta_MEMIT - Delta_AlphaEdit||_F: {norm_fro_D:.4f}")

    # 可视化 D (Heatmap)
    plt.figure(figsize=(8, 6))
    # 如果D太大，可能需要采样或取绝对值
    sample_D = D[:100, :100] if D.shape[0] > 100 and D.shape[1] > 100 else D
    plt.imshow(sample_D.abs().cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Magnitude of D elements')
    plt.title(f'Heatmap of |D| (Layer {layer_id}, Chunk {chunk_weights_dir.name}) (Sampled)')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    # plt.show() # 或者保存图像
    fig_save_path = chunk_weights_dir / f"layer_{layer_id:02d}_D_heatmap.png"
    plt.savefig(fig_save_path)
    print(f"Saved heatmap of D to {fig_save_path}")
    plt.close()

    # D 的元素分布直方图
    plt.figure(figsize=(8, 6))
    plt.hist(D.cpu().numpy().flatten(), bins=100, log=True)
    plt.title(f'Element Distribution of D (Layer {layer_id}, Chunk {chunk_weights_dir.name})')
    plt.xlabel('Value')
    plt.ylabel('Frequency (log scale)')
    fig_save_path_hist = chunk_weights_dir / f"layer_{layer_id:02d}_D_hist.png"
    plt.savefig(fig_save_path_hist)
    print(f"Saved histogram of D to {fig_save_path_hist}")
    plt.close()
    '''

    # 3. 对 preserved knowledge 的影响
    # 计算 ||Delta * K0K0T||_F
    if K0K0T_layer is not None:
        # K0K0T_layer 是 d_model x d_model
        # delta_alpha 是 d_out x d_in (取决于模型，GPT2中 c_proj 是 d_model x 4*d_model, fc_out 是 4*d_model x d_model)
        # 假设 delta_alpha 是 (d1 x d0), K0K0T 是 (d0 x d0)
        # 需要确保维度匹配。论文中 K0 是 d0 x N_preserved。 Delta P K0 = 0.
        # Delta P K0 K0^T = 0. Delta_AlphaEdit 就是 Delta P.
        # 所以我们要计算 Delta_AlphaEdit @ K0K0T_layer
        
        # 检查 delta_alpha 和 K0K0T_layer 的维度是否能相乘
        # GPT-2 MLP c_proj: weight is [n_embd, 4 * n_embd]. Delta has same shape.
        # K0 (keys) are from the input of this c_proj layer, so they are [4 * n_embd, num_preserved_samples]
        # K0K0T is [4 * n_embd, 4 * n_embd]
        # So, product is Delta @ K0K0T
        if delta_alpha.shape[1] == K0K0T_layer.shape[0]:
            impact_preserved_alpha = delta_alpha.float() @ K0K0T_layer
            norm_impact_preserved_alpha = torch.linalg.norm(impact_preserved_alpha, 'fro').item()
            print(f"||Delta_AlphaEdit @ K0K0T||_F: {norm_impact_preserved_alpha:.4e}")
        else:
            print(f"Dimension mismatch for Delta_AlphaEdit @ K0K0T: {delta_alpha.shape} vs {K0K0T_layer.shape}")
        '''
        if delta_memit.shape[1] == K0K0T_layer.shape[0]:
            impact_preserved_memit = delta_memit @ K0K0T_layer
            norm_impact_preserved_memit = torch.linalg.norm(impact_preserved_memit, 'fro').item()
            print(f"||Delta_MEMIT @ K0K0T||_F:   {norm_impact_preserved_memit:.4e}")
        else:
            print(f"Dimension mismatch for Delta_MEMIT @ K0K0T: {delta_memit.shape} vs {K0K0T_layer.shape}")
        '''
    else:
        print("K0K0T_layer not found, skipping preserved knowledge impact calculation.")

    # 4. 对 Updated Knowledge 的影响
    # 计算 ||Delta K1 - R||_F^2
    # K1: u x d (num_updates x feature_dim_keys)
    # R_chunk_targets: d_out x u (feature_dim_values x num_updates)
    # Delta: d_out x d_in (value_dim x key_dim)
    # (Delta @ K1.T) should be (d_out x u) to match R_chunk_targets
    if K1 is not None and R_chunk_targets is not None:
        # 确保 K1 和 R_chunk_targets 的批次大小 u 一致
        # K1 (u x d_in), R (d_out x u)
        # Delta (d_out x d_in)
        # Delta @ K1.T  -> (d_out x d_in) @ (d_in x u) = (d_out x u)
        if delta_alpha.shape[1] == K1.shape[1]: # d_in
            term_alpha = (delta_alpha.float() @ K1.T) - R_chunk_targets # (d_out x u)
            norm_sq_updated_alpha = (torch.linalg.norm(term_alpha, 'fro')**2).item()
            print(f"||Delta_AlphaEdit K1 - R||_F^2: {norm_sq_updated_alpha:.4e}")
        else:
            print(f"Dimension mismatch for Delta_AlphaEdit K1 - R: Delta {delta_alpha.shape}, K1.T {K1.T.shape}, R {R_chunk_targets.shape}")
        '''
        if delta_memit.shape[1] == K1.shape[1]:
            term_memit = (delta_memit @ K1.T) - R_chunk_targets
            norm_sq_updated_memit = (torch.linalg.norm(term_memit, 'fro')**2).item()
            print(f"||Delta_MEMIT K1 - R||_F^2:   {norm_sq_updated_memit:.4e}")
        else:
             print(f"Dimension mismatch for Delta_MEMIT K1 - R: Delta {delta_memit.shape}, K1.T {K1.T.shape}, R {R_chunk_targets.shape}")
        '''

    else:
        print("K1 or R_chunk_targets not found, skipping updated knowledge impact calculation.")


def main_analysis(run_dir_str: str):
    run_dir = Path(run_dir_str)
    if not run_dir.exists() or not (run_dir / "weights").exists():
        print(f"Run directory or weights subdirectory not found: {run_dir}")
        return

    # 确定有多少 chunk 和 layer
    # 假设 layer 信息从 P 或 K0K0T 文件名获取
    # 假设 chunk 信息从 chunk_xxx 目录名获取
    num_layers = 0
    for f in (run_dir / "weights").glob("layer_*_P.pt"):
        try:
            num_layers = max(num_layers, int(f.name.split('_')[1]) + 1)
        except:
            pass
    
    chunk_dirs = sorted([d for d in (run_dir / "weights").iterdir() if d.is_dir() and d.name.startswith("chunk_")])

    print(f"Found {len(chunk_dirs)} chunks and {num_layers} layers configured (approx).")

    for chunk_dir in chunk_dirs:
        for layer_idx in range(num_layers): # TODO: 获取确切的层列表而不是假设0-num_layers
            # 检查该层是否有文件，避免对不存在的层进行分析 (例如，如果hparams.layers不是连续的)
            if not (chunk_dir / f"layer_{layer_idx:02d}_W_orig.pt").exists():
                continue
            analyze_chunk_layer(chunk_dir, layer_idx)

def visualize_weight_distribution_evolution(base_dir: Path, output_dir: Path):
    """
    可视化模型权重分布随 chunk 增加的演变过程。

    Args:
        base_dir (Path): 包含所有 chunk 目录的根目录。
                         例如: Path("/path/to/your/weights")
        output_dir (Path): 保存可视化结果图片的输出目录。
    """
    if not base_dir.is_dir():
        print(f"Error: Base directory '{base_dir}' not found.")
        return

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 自动识别 chunk 目录和层数
    # 通过目录名中的数字来排序，确保 chunk_0, chunk_1, ... chunk_10 的正确顺序
    chunk_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")],
        key=lambda x: int(re.search(r'\d+', x.name).group())
    )
    
    if not chunk_dirs:
        print(f"Error: No directories starting with 'chunk_' found in '{base_dir}'.")
        return

    # 通过检查第一个 chunk 目录中的文件来确定总层数
    layer_files = list(chunk_dirs[0].glob("layer_*_W_orig.pt"))
    if not layer_files:
        print(f"Error: No layer weight files found in '{chunk_dirs[0]}'.")
        return
        
    num_layers = len(layer_files)
    print(f"Found {len(chunk_dirs)} chunks and {num_layers} layers.")

    # 2. 为每一层生成一张演变图
    for layer_file in layer_files:
        print(f"Processing {layer_file}...")
        
        plt.figure(figsize=(12, 7))
        
        # 使用 colormap 来表示时间的推移 (chunk 的增加)
        colors = plt.cm.viridis(np.linspace(0, 1, len(chunk_dirs)))

        # 3. 遍历所有 chunk，加载权重并绘制分布
        for i, chunk_dir in enumerate(chunk_dirs):
            weight_file = chunk_dir / f"layer_{layer_idx:02d}_W_orig.pt"
            
            if weight_file.exists():
                # 加载权重张量
                weights = load_matrix(weight_file)

                # 将张量展平为一维向量并转换为 numpy 数组，方便绘图
                # 使用 .detach() 来避免梯度计算
                weights_flat = weights.detach().numpy().flatten()

                if weights_flat.size > 1000000:  # Sample if more than 1M weights
                    rng = np.random.default_rng()
                    weights_flat = rng.choice(weights_flat, size=1000000, replace=False)
                
                # 绘制 KDE 图
                sns.kdeplot(
                    weights_flat,
                    color=colors[i],
                    label=f"{chunk_dir.name}",
                    fill=True,       # 添加填充效果
                    alpha=0.2        # 设置透明度
                )

        # 4. 美化并保存图像
        plt.title(f'Layer {layer_idx}: Weight Distribution Evolution', fontsize=16)
        plt.xlabel('Weight Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(title='Chunk')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 保存图像
        output_filename = output_dir / f"layer_{layer_idx:02d}_weights_evolution.png"
        plt.savefig(output_filename, dpi=150)
        plt.close() # 关闭当前图形，防止在下一个循环中继续绘制

    print(f"\nAll visualizations have been saved to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze AlphaEdit and MEMIT deltas.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory (e.g., results/AlphaEdit/run_000)")
    args = parser.parse_args()
    
    # main_analysis(args.run_dir)

    visualization_output_dir = Path(args.run_dir) / "alphaedit_weight_visualizations"
    weight_dir = Path(args.run_dir) / "weights"

    visualize_weight_distribution_evolution(weight_dir, visualization_output_dir)