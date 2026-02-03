import json
import os
import argparse
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
import time
import sae_bench.custom_saes.base_sae as base_sae
import sae_bench.custom_saes.batch_topk_sae as batch_topk_sae
import sae_bench.custom_saes.gated_sae as gated_sae
import sae_bench.custom_saes.jumprelu_sae as jumprelu_sae
import sae_bench.custom_saes.relu_sae as relu_sae
import sae_bench.custom_saes.topk_sae as topk_sae
import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.autointerp.main as autointerp
import sae_bench.evals.core.main as core
import sae_bench.evals.ravel.main as ravel
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.evals.unlearning.main as unlearning
import sae_bench.sae_bench_utils.general_utils as general_utils
import gc

MODEL_CONFIGS = {
    "google/gemma-2-9b": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "layers": [22],
        "d_model": 3584,
        "context_length": 1024
    },
    "google/gemma-2-2b": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "layers": [12],
        "d_model": 2304,
        "context_length": 1024
    },
    "Qwen/Qwen3-8B": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "layers": [],
        "d_model": 4096,
        "context_length": 1024
    },
    "Qwen/Qwen3-1.7B": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "layers": [],
        "d_model": 4096,
        "context_length": 1024
    },
}

output_folders = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    "unlearning": "eval_results/unlearning",
    "ravel": "eval_results/ravel",
}


TRAINER_LOADERS = {
    "MatryoshkaBatchTopKTrainer": batch_topk_sae.load_dictionary_learning_matryoshka_batch_topk_sae,
    "CoMatryoshkaBatchTopKTrainer": batch_topk_sae.load_dictionary_learning_matryoshka_batch_topk_sae,
    "BatchTopKTrainer": batch_topk_sae.load_dictionary_learning_batch_topk_sae,
    "CoBatchTopKTrainer": batch_topk_sae.load_dictionary_learning_batch_topk_sae,
    "TopKTrainer": topk_sae.load_dictionary_learning_topk_sae,
    "CoTopKTrainer": topk_sae.load_dictionary_learning_topk_sae,
    "StandardTrainerAprilUpdate": relu_sae.load_dictionary_learning_relu_sae,
    "CoStandardTrainerAprilUpdate": relu_sae.load_dictionary_learning_relu_sae,
    "StandardTrainer": relu_sae.load_dictionary_learning_relu_sae,
    "PAnnealTrainer": relu_sae.load_dictionary_learning_relu_sae,
    "JumpReluTrainer": jumprelu_sae.load_dictionary_learning_jump_relu_sae,
    "CoJumpReluTrainer": jumprelu_sae.load_dictionary_learning_jump_relu_sae,
    "GatedSAETrainer": gated_sae.load_dictionary_learning_gated_sae,
    "CoGatedSAETrainer": gated_sae.load_dictionary_learning_gated_sae,
    "OrtTrainer": batch_topk_sae.load_dictionary_learning_batch_topk_sae,
    "CoOrtTrainer": batch_topk_sae.load_dictionary_learning_batch_topk_sae,
}

def load_dictionary_learning_sae(
    repo_id: str,
    location: str,
    model_name,
    device: str,
    dtype: torch.dtype,
    layer: int | None = None,
    download_location: str = "downloaded_saes",
) -> base_sae.BaseSAE:
    if repo_id == 'local':
        config_filename = location.replace("ae.pt", "config.json")
        if 'checkpoints' in location:
            config_filename = location.split("checkpoints")[0] + "config.json"

        if 'ae.pt' in location:
             config_filename = location.split('ae.pt')[0] + "config.json"
        elif 'checkpoints' in location:
             config_filename = location.split('checkpoints')[0] + "config.json"
        else:
             config_filename = os.path.join(os.path.dirname(location), "config.json")
             if not os.path.exists(config_filename):
                 config_filename = os.path.join(os.path.dirname(os.path.dirname(location)), "config.json")

        with open(config_filename) as f:
            config = json.load(f)
        trainer_class = config["trainer"]["trainer_class"]

        sae = TRAINER_LOADERS[trainer_class](
            repo_id=repo_id,
            filename=location,
            layer=layer,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
    else:
        download_location = os.path.join(download_location, repo_id.replace("/", "_"))

        config_file = f"{download_location}/{location}/config.json"

        with open(config_file) as f:
            config = json.load(f)

        trainer_class = config["trainer"]["trainer_class"]

        location = f"{location}/ae.pt"

        sae = TRAINER_LOADERS[trainer_class](
            repo_id=repo_id,
            filename=location,
            layer=layer,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
    return sae


def verify_saes_load(
    repo_id: str,
    sae_locations: list[str],
    model_name: str,
    device: str,
    dtype: torch.dtype,
):
    """Verify that all SAEs load correctly. Useful to check this before a big evaluation run."""
    print(f"\n--- Verifying SAEs for model: {model_name} ---")
    for sae_location in sae_locations:
        print(f"Verifying {sae_location}...")
        sae, config = load_dictionary_learning_sae(
            repo_id=repo_id,
            location=sae_location,
            layer=None,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
        print(f"Successfully loaded {sae_location}.")
        del sae


def run_evals(
    repo_id: str,
    model_name: str,
    sae_locations: list[str],
    llm_batch_size: int,
    llm_dtype: str,
    device: str,
    eval_types: list[str],
    random_seed: int,
    api_key: str | None = None,
    base_url: str | None = None,
    openai_model: str | None = None,
    openai_max_workers: int = 10,
    force_rerun: bool = False,
    reverse_iterate=False,
    dataset_path: str = None,
    sae_dtype: str = "float32",
):
    """Run selected evaluations for the given model and SAEs."""

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    from datasets import load_dataset, load_from_disk

    if dataset_path:
        if dataset_path.endswith(".arrow") and os.path.exists(dataset_path):
            from datasets import Dataset
            pretrain_ds = Dataset.from_file(dataset_path)
        elif os.path.exists(dataset_path):
            if "openwebtext" in dataset_path and os.path.isdir(dataset_path):
                import glob
                arrow_files = sorted(glob.glob(os.path.join(dataset_path, "*.arrow")))
                if arrow_files:
                    from datasets import Dataset
                    print(f"\n\nOpenWebText detected. Loading last arrow file: {arrow_files[-1]}\n\n")
                    pretrain_ds = Dataset.from_file(arrow_files[-1])
                else:
                    pretrain_ds = load_from_disk(dataset_path)
            else:
                pretrain_ds = load_from_disk(dataset_path)
        else:
            pretrain_ds = load_dataset(dataset_path, split='train', streaming=False)
    else:
        pretrain_ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split='train', streaming=False)
    eval_runners = {
        "absorption": (
            lambda selected_saes, is_final: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    context_size=MODEL_CONFIGS[model_name]['context_length'],
                    k_sparse_probe_num_epochs=50,
                ),
                selected_saes,
                device,
                "eval_results/absorption",
                force_rerun,
            )
        ),
        "autointerp": (
            lambda selected_saes, is_final: autointerp.run_eval(
                autointerp.AutoInterpEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=16,
                    llm_dtype=llm_dtype,
                    n_latents=128,
                    context_size=MODEL_CONFIGS[model_name]['context_length'],
                    total_tokens=2_000_000,
                    max_tokens_in_explanation=16384,
                    dataset_name=dataset_path if dataset_path else "HuggingFaceFW/fineweb",
                    subset_name='' if dataset_path else "sample-10BT",
                ),
                selected_saes,
                device,
                api_key,
                "eval_results/autointerp",
                force_rerun,
                base_url=base_url,
                openai_model=openai_model,
                n_api_workers=openai_max_workers,
            )
        ),
        "core": (
            lambda selected_saes, is_final: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=600,
                n_eval_sparsity_variance_batches=1000,
                eval_batch_size_prompts=8,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset=pretrain_ds,
                context_size=MODEL_CONFIGS[model_name]['context_length'],
                output_folder="eval_results/core",
                verbose=True,
                dtype=llm_dtype,
                device=device,
            )
        ),
        "ravel": (
            lambda selected_saes, is_final: ravel.run_eval(
                ravel.RAVELEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    context_length=MODEL_CONFIGS[model_name]['context_length'],
                ),
                selected_saes,
                device,
                "eval_results/ravel",
                force_rerun,
            )
        ),
    }

    for eval_type in eval_types:
        if eval_type not in eval_runners:
            raise ValueError(f"Unsupported eval type: {eval_type}")

    verify_saes_load(
        repo_id,
        sae_locations,
        model_name,
        device,
        general_utils.str_to_dtype(sae_dtype),
    )

    for eval_type in tqdm(eval_types, desc=f"Evaluations for {model_name}"):
        if eval_type == "autointerp" and api_key is None:
            print("Skipping autointerp evaluation due to missing API key")
            continue
        if eval_type == "unlearning":
            if not os.path.exists(
                "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
            ):
                print(
                    "Skipping unlearning evaluation due to missing bio-forget-corpus.jsonl"
                )
                continue

        print(f"\n\n\nRunning {eval_type} evaluation for {model_name}\n\n\n")

        if reverse_iterate:
            sae_iterator = enumerate(sae_locations[::-1])
        else:
            sae_iterator = enumerate(sae_locations)

        for i, sae_location in sae_iterator:
            is_final = False
            if i == len(sae_locations) - 1:
                is_final = True

            print(f"\n--- Loading SAE from: {sae_location} ---")
            sae, config = load_dictionary_learning_sae(
                repo_id=repo_id,
                location=sae_location,
                layer=None,
                model_name=model_name,
                device=device,
                dtype=general_utils.str_to_dtype(sae_dtype),
            )

            unique_sae_id = sae_location.replace("/", "_")

            try:
                unique_sae_id_suffix = unique_sae_id.split('dictionary_learning_demo_')[1]
                unique_sae_id = f"{unique_sae_id_suffix}___{sae.cfg.architecture}"
            except (IndexError, AttributeError):
                 unique_sae_id = unique_sae_id.replace("_dictionary_learning_demo_", "")
                 unique_sae_id = f"{unique_sae_id}___{sae.cfg.architecture}"


            selected_saes = [(unique_sae_id, sae)]
            print("\n\nunique_sae_id:", unique_sae_id)
            os.makedirs(output_folders[eval_type], exist_ok=True)
            eval_runners[eval_type](selected_saes, is_final)

            del sae
            torch.cuda.empty_cache()
            gc.collect()


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run SAE evaluations from command line.")

    parser.add_argument("--sae_dirs", type=str, nargs='+', required=True,
                        help="List of SAE directory patterns. "
                             "Example: '/path/to/saes/trainer_{}/ae.pt'")

    parser.add_argument("--device", type=str, required=True,
                        help="Device to run on (e.g., 'cuda:0', 'cpu').")

    parser.add_argument("--eval_types", type=str, nargs='+',
                        default=["absorption", "core", "autointerp"],
                        help="List of evaluation types to run.")

    parser.add_argument("--dir_len", type=int, default=15,
                        help="Number of trainers (length of loop).")

    parser.add_argument("--api_key_file", type=str, default="openai_api_key.txt",
                        help="Path to file containing OpenAI API key.")

    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key.")

    parser.add_argument("--openai_base_url", type=str, default=None,
                        help="OpenAI Base URL.")

    parser.add_argument("--openai_model", type=str, default=None,
                        help="OpenAI Model.")

    parser.add_argument("--openai_max_workers", type=int, default=10,
                        help="OpenAI Max Workers.")

    parser.add_argument("--target_gb", type=int, default=70,
                        help="Target GPU memory (in GB) to reserve with padding.")

    parser.add_argument("--reverse_iterate", action='store_true',
                        help="Iterate over SAEs in reverse order.")

    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for evaluations.")

    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset.")

    parser.add_argument("--sae_dtype", type=str, default=None,
                        help="Data type for SAE (e.g., 'float32', 'float32'). If None, defaults to 'float32'.")

    return parser.parse_args()

def main(args):
    """Main execution function"""

    RANDOM_SEED = args.random_seed
    device = args.device

    api_key = args.openai_api_key
    if api_key is None:
        try:
            with open(args.api_key_file) as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            pass

    if api_key is None:
        print(f"Warning: API key not provided and file not found at {args.api_key_file}. Skipping autointerp.")

    base_url = args.openai_base_url
    openai_model = args.openai_model
    openai_max_workers = args.openai_max_workers

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

    eval_types = args.eval_types
    target_bytes = int(args.target_gb * (1024**3))
    reverse_iterate = args.reverse_iterate



    torch.cuda.empty_cache()
    needed_bytes = target_bytes - torch.cuda.memory_reserved()

    num_elements = needed_bytes

    pad_tensor = torch.empty(
        num_elements,
        dtype=torch.uint8,
        device=args.device
    )

    print(f"Successfully allocated {needed_bytes / (1024**3):.2f} GB padding tensor.")

    del pad_tensor
    gc.collect()

    print(f"Padding tensor deleted. Current Reserved Memory: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")

    print("device: ", device)

    torch.cuda.empty_cache()

    for sae_dir_pattern in args.sae_dirs:
        if  "Llama-3.2-3B" in sae_dir_pattern:
            model_name = "meta-llama/Llama-3.2-3B"
        elif "Qwen2.5-1.5B" in sae_dir_pattern:
            model_name = "Qwen/Qwen2.5-1.5B"
        elif "Qwen2.5-3B" in sae_dir_pattern:
            model_name = "Qwen/Qwen2.5-3B"
        elif "Qwen3-8B" in sae_dir_pattern:
            model_name = "Qwen/Qwen3-8B"
        elif "Qwen3-1.7B" in sae_dir_pattern:
            model_name = "Qwen/Qwen3-1.7B"
        elif "gemma-2-2b" in sae_dir_pattern:
            model_name = "google/gemma-2-2b"
        elif "gemma-2-9b" in sae_dir_pattern:
            model_name = "google/gemma-2-9b"

        print(f"\n\n\n{'='*50}")
        print(f"STARTING PROCESSING FOR MODEL: {model_name}")
        print(f"Using pattern: {sae_dir_pattern}")
        print(f"{'='*50}\n")

        if model_name not in MODEL_CONFIGS:
            print(f"Warning: Model '{model_name}' not in MODEL_CONFIGS. Skipping pattern.")
            continue

        llm_batch_size = MODEL_CONFIGS[model_name]["batch_size"]
        str_dtype = MODEL_CONFIGS[model_name]["dtype"]

        sae_locations_for_this_model = []
        print(f"Directory length: {args.dir_len}")

        for trainer_id in range(args.dir_len):
            sae_location = sae_dir_pattern.format(trainer_id)

            if 'checkpoints' in sae_location:
                config_path = sae_location.split('checkpoints')[0] + "config.json"
            elif 'ae.pt' in sae_location:
                config_path = sae_location.split('ae.pt')[0] + "config.json"
            else:
                print(f"Warning: Could not determine config path for {sae_location}. Skipping.")
                continue

            if not os.path.exists(config_path):
                continue

            if not os.path.exists(sae_location):
                continue


            sae_locations_for_this_model.append(sae_location)

        print(f"\n--- Total SAEs found for {model_name}: {len(sae_locations_for_this_model)} ---")
        if not sae_locations_for_this_model:
            print("Error: No SAEs found for this pattern. Exiting.")
            continue

        run_evals(
            repo_id="local",
            model_name=model_name,
            sae_locations=sae_locations_for_this_model,
            llm_batch_size=llm_batch_size,
            llm_dtype=str_dtype,
            device=device,
            eval_types=eval_types,
            api_key=api_key,
            base_url=base_url,
            openai_model=openai_model,
            openai_max_workers=openai_max_workers,
            random_seed=RANDOM_SEED,
            reverse_iterate=reverse_iterate,
            dataset_path=args.dataset_path,
            sae_dtype=args.sae_dtype,
        )

        print(f"\n{'='*50}")
        print(f"COMPLETED PROCESSING FOR MODEL: {model_name}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    args = get_args()
    main(args)
