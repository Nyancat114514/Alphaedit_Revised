import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import CounterFactDataset
from experiments.py.eval_utils_counterfact import evaluate_counterfact
from util.generate import generate_fast
from util.hparams import HParams

# python test_single_case.py --case_id 123

def main():
    """
    对指定的 case_id 在原始（未经编辑）的模型上进行测试，并输出结果和评估指标。
    """
    parser = argparse.ArgumentParser(description="Test a single case on the original model.")
    parser.add_argument(
        "--case_id", type=int, required=True, help="The case_id to test."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="The name or path of the model to use.",
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="hparams/AlphaEdit/Llama3-8B.json",
        help="Path to the hyperparameters file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where the dataset is stored.",
    )
    args = parser.parse_args()

    # 加载模型和分词器
    print(f"Loading model: {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).cuda()
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token
    print("Model loaded.")

    # 加载数据集
    print(f"Loading dataset from {args.data_dir}...")
    ds = CounterFactDataset(data_dir=args.data_dir)
    record = next((r for r in ds if r["case_id"] == args.case_id), None)

    if record is None:
        print(f"Error: Case ID {args.case_id} not found in the dataset.")
        return
    print(f"Found Case ID: {args.case_id}")

    # 加载评估所需的超参数
    hparams = HParams.from_json(args.hparams_fname)

    # 准备输入
    rewrite = record["requested_rewrite"]
    prompt = rewrite["prompt"].format(rewrite["subject"])
    target_new = rewrite["target_new"]["str"]
    target_true = rewrite["target_true"]["str"]

    # 使用原始模型生成文本
    print("Generating text from the original model...")
    generated_text = generate_fast(
        model, tok, [prompt], n_gen_per_prompt=1, max_out_len=100
    )[0]
    print("Text generation complete.")

    # 在原始模型上计算评估指标
    print("Calculating metrics...")
    metrics = evaluate_counterfact(model, tok, [rewrite], hparams)
    print("Metrics calculation complete.")

    # 格式化并打印结果
    print("\n" + "="*50)
    print(" " * 15 + "ORIGINAL MODEL TEST RESULT")
    print("="*50)
    print(f"Case ID: {record['case_id']}")
    print("-" * 50)
    print(f"Input Prompt:\n  {prompt}")
    print(f"\nTarget (New):\n  {target_new}")
    print(f"Target (True):\n  {target_true}")
    print(f"\nGenerated Output:\n  {generated_text}")
    print("-" * 50)

    # 根据评估指标判断重写是否成功
    if metrics.get("post", {}).get("rewrite_success"):
        print("\n--- Successful Rewrite (according to metrics) ---")
    else:
        print("\n--- Unsuccessful Rewrite (according to metrics) ---")

    # 打印详细的评估指标
    print("\nMetrics:")
    # 使用 'post' 前缀的指标，因为它们代表了评估时的模型状态（在此脚本中即为原始模型）
    post_metrics = metrics.get("post", {})
    print(json.dumps(post_metrics, indent=4))
    print("="*50)


if __name__ == "__main__":
    main()