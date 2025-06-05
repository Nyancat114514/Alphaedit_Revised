import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams
# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    cache_c = None,
    P = None,
    run_dir: Optional[Path] = None,
    chunk_idx: Optional[int] = None
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)


    current_chunk_weights_dir = None
    if run_dir is not None and chunk_idx is not None:
        current_chunk_weights_dir = Path(run_dir) / "weights" / f"chunk_{chunk_idx:03d}"
        current_chunk_weights_dir.mkdir(parents=True, exist_ok=True)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"

        # 1. 保存当前层的 W_orig (在应用本层本 chunk 的 delta 之前)
        W_orig_layer = weights[weight_name].detach().cpu()
        if current_chunk_weights_dir:
            torch.save(W_orig_layer, current_chunk_weights_dir / f"layer_{layer:02d}_W_orig.pt")


        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = torch.linalg.solve(
                P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T
        )
        # Adjust update matrix shape
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        if current_chunk_weights_dir:
            torch.save(layer_ks.T.detach().cpu(), current_chunk_weights_dir / f"layer_{layer:02d}_K1.pt") # (u x d_model)
            torch.save(targets.detach().cpu(), current_chunk_weights_dir / f"layer_{layer:02d}_R_chunk_targets.pt") # (d_model x u)
            torch.save(resid.detach().cpu(), current_chunk_weights_dir / f"layer_{layer:02d}_R_partial_resid.pt") # (d_model x u)




        # 为了统一，我们从 `weights` 中取出 W_orig, 计算 delta, 然后再加回去
        # AlphaEdit 的 delta (就是原代码的 upd_matrix)
        # 注意：原代码直接用 upd_matrix 更新了 weights[weight_name]，我们需要先保存它
        
        # 计算 Delta_AlphaEdit (即原代码中的 upd_matrix)
        # 这个 resid 是已经除以层数的 R_partial
        delta_alphaedit_term_A = P[i,:,:].cuda().double() @ (layer_ks.double() @ layer_ks.double().T + cache_c[i,:,:].cuda().double()) + \
                                 hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.double, device="cuda")
        delta_alphaedit_term_B = P[i,:,:].cuda().double() @ layer_ks.double() @ resid.double().T
        
        # 使用 torch.linalg.lstsq 来处理可能的奇异矩阵，或者确保矩阵可逆
        try:
            # delta_alphaedit_unshaped = torch.linalg.solve(delta_alphaedit_term_A, delta_alphaedit_term_B)
            # 更稳健的解法，如果A不是方阵或奇异
            delta_alphaedit_unshaped = torch.linalg.lstsq(delta_alphaedit_term_A, delta_alphaedit_term_B).solution

        except torch.linalg.LinAlgError as e:
            print(f"Layer {layer} AlphaEdit: Singular matrix encountered. Error: {e}")
            # 可以尝试使用伪逆，或者跳过这个delta的计算和保存
            try:
                A_pinv = torch.linalg.pinv(delta_alphaedit_term_A)
                delta_alphaedit_unshaped = A_pinv @ delta_alphaedit_term_B
                print("Used pseudo-inverse for AlphaEdit delta.")
            except torch.linalg.LinAlgError as e_pinv:
                print(f"Layer {layer} AlphaEdit: Pseudo-inverse also failed. Error: {e_pinv}. Skipping delta calculation.")
                delta_alphaedit_unshaped = torch.zeros_like(weights[weight_name].T, dtype=torch.double, device="cuda") #转置因为下面要匹配

        delta_alphaedit_layer = upd_matrix_match_shape(delta_alphaedit_unshaped, weights[weight_name].shape)

        if current_chunk_weights_dir:
            torch.save(delta_alphaedit_layer.detach().cpu(), current_chunk_weights_dir / f"layer_{layer:02d}_delta_alphaedit.pt")



        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix

        if current_chunk_weights_dir:
            torch.save(weights[weight_name].detach().cpu(), current_chunk_weights_dir / f"layer_{layer:02d}_W_alphaedit_applied.pt")


        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    print(f"Deltas successfully computed and saved for {list(weights.keys())}")
    return model, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
