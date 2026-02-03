import os
import ast

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import itertools
import random
import json
import torch.multiprocessing as mp
import time
import huggingface_hub
from datasets import config
from transformers import AutoTokenizer

import demo_config

from dictionary_learning.dictionary_learning.utils import (
    hf_dataset_to_generator,
    hf_mixed_dataset_to_generator,
    hf_sequence_packing_dataset_to_generator,
)
from dictionary_learning.dictionary_learning.pytorch_buffer import ActivationBuffer
from dictionary_learning.dictionary_learning.evaluation import evaluate
from dictionary_learning.dictionary_learning.training import trainSAE
import dictionary_learning.dictionary_learning.utils as utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="where to store sweep")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument("--save_checkpoints", action="store_true", help="save checkpoints")
    parser.add_argument("--layers", type=int, nargs="+", required=True, help="layers to train SAE on")
    parser.add_argument("--model_name", type=str, required=True, help="which language model to use")
    parser.add_argument("--architectures", type=str, nargs="+", choices=[e.value for e in demo_config.TrainerType], required=True, help="which SAE architectures to train")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to train on")
    parser.add_argument("--hf_repo_id", type=str, help="Hugging Face repo ID to push results to")
    parser.add_argument("--mixed_dataset", action="store_true", help="use mixed dataset")
    parser.add_argument("--dataset_path", type=str, default=None, help="path to dataset")
    parser.add_argument("--wandb_project", type=str, default="sae-sweep", help="wandb project name")
    parser.add_argument("--target_gb", type=float, default=40, help="target GPU memory in GB")
    parser.add_argument("--target_l0s", type=str, default=None, help="target l0s")
    parser.add_argument("--target_l1s", type=str, default=None, help="target l1s")
    parser.add_argument("--lambda_swc2r", type=str, default=None, help="lambda swc2r")
    parser.add_argument("--swc2r_alpha", type=float, default=1.0, help="swc2r alpha")
    parser.add_argument("--swc2r_tau", type=float, default=0.95, help="swc2r tau")
    parser.add_argument("--swc2r_type", type=str, default="topTauPerFeatSquare", help="swc2r type")
    parser.add_argument("--aux_loss_start_step", type=int, default=0, help="aux loss start step")
    parser.add_argument("--aux_loss_interval", type=int, default=1, help="aux loss interval")
    parser.add_argument("--dtype", type=str, default="float32", help="dtype")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--decay_start_fraction", type=float, default=0.8, help="decay start fraction")
    parser.add_argument("--sae_batch_size", type=int, default=None, help="sae batch size")
    parser.add_argument("--num_tokens", type=int, default=500000000, help="number of tokens")

    args = parser.parse_args()
    return args




def run_sae_training(
    model_name: str,
    layer: int,
    save_dir: str,
    device: str,
    architectures: list,
    num_tokens: int,
    random_seeds: list[int],
    dictionary_widths: list[int],
    learning_rates: list[float],
    dry_run: bool = False,
    use_wandb: bool = False,
    save_checkpoints: bool = False,
    buffer_tokens: int = 250_000,
    mixed_dataset: bool = False,
    dataset_path: str = None,
    wandb_project: str = "sae-sweep",
    target_l0s: list[int] = None,
    target_l1s: list[float] = None,
    lambda_swc2r: list[float] = None,
    swc2r_alpha: float = 1.0,
    swc2r_tau: float = 0.95,
    swc2r_type: str = "topTauPerFeatSquare",
    aux_loss_start_step: int = 0,
    aux_loss_interval: int = 1,
    dtype_str: str = "float32",
    decay_start_fraction: float = 0.8,
    sae_batch_size_override: int = None,
):
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    if lambda_swc2r: demo_config.SPARSITY_PENALTIES.lambda_swc2r = lambda_swc2r
    if args.swc2r_tau: demo_config.SPARSITY_PENALTIES.swc2r_tau = [args.swc2r_tau]
    if args.swc2r_type: demo_config.SPARSITY_PENALTIES.swc2r_type = [args.swc2r_type]

    context_length = demo_config.LLM_CONFIG[model_name].context_length

    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    sae_batch_size = sae_batch_size_override if sae_batch_size_override else demo_config.LLM_CONFIG[model_name].sae_batch_size

    if dtype_str == "float32":
        dtype = t.float32
    elif dtype_str == "float16":
        dtype = t.float16
    elif dtype_str == "bfloat16":
        dtype = t.bfloat16
    else:
        dtype = demo_config.LLM_CONFIG[model_name].dtype

    num_buffer_inputs = buffer_tokens // context_length
    print(f"buffer_size: {num_buffer_inputs}, buffer_size_in_tokens: {buffer_tokens}")

    log_steps = 100

    steps = int(num_tokens / sae_batch_size)

    if save_checkpoints:
        desired_checkpoints = t.logspace(-3, 0, 7).tolist()
        desired_checkpoints = [0.0] + desired_checkpoints[:-1]
        desired_checkpoints.sort()
        print(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"save_steps: {save_steps}")
    else:
        save_steps = None

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEMO] Starting model loading...")
    model_load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": device}, torch_dtype=dtype
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEMO] Model loading completed in {time.time() - model_load_start:.2f}s")

    model = utils.truncate_model(model, layer)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    submodule = utils.get_submodule(model, layer)
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    if "Qwen" in model_name and demo_config.remove_bos:
        print(
            "\n\nWARNING: Qwen models do not have a bos token, we will remove the first non-pad token"
        )

    if mixed_dataset:

        qwen_system_prompt_to_remove = None

        generator = hf_mixed_dataset_to_generator(
            tokenizer,
            system_prompt_to_remove=qwen_system_prompt_to_remove,
            sequence_pack_pretrain=True,
            system_prompt_removal_freq=0.0,
            min_chars=context_length * 4,
        )
    else:
        dataset_name = dataset_path if dataset_path else "monology/pile-uncopyrighted"
        generator = hf_sequence_packing_dataset_to_generator(
            tokenizer,
            min_chars=context_length * 4,
            pretrain_dataset=dataset_name,
            streaming=False,
        )

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEMO] Creating activation buffer...")
    buffer_creation_start = time.time()
    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=num_buffer_inputs,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=device,
        add_special_tokens=False,
        remove_bos=demo_config.remove_bos,
        max_activation_norm_multiple=demo_config.max_activation_norm_multiple,
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEMO] Activation buffer creation completed in {time.time() - buffer_creation_start:.2f}s")

    trainer_configs = demo_config.get_trainer_configs(
        architectures,
        learning_rates,
        random_seeds,
        activation_dim,
        dictionary_widths,
        model_name,
        device,
        layer,
        submodule_name,
        steps,
        decay_start_fraction=decay_start_fraction,
        dtype=dtype,
        aux_loss_start_step=aux_loss_start_step,
        aux_loss_interval=aux_loss_interval,
        target_l0s=target_l0s,
        target_l1s=target_l1s,
        swc2r_alpha=swc2r_alpha,
        swc2r_tau=swc2r_tau,
        swc2r_type=swc2r_type,
    )

    print(f"len trainer configs: {len(trainer_configs)}")
    assert len(trainer_configs) > 0
    save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEMO] Starting SAE training...")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEMO] Training configuration: steps={steps}, save_dir={save_dir}")
        training_start = time.time()
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            use_wandb=use_wandb,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            wandb_project=wandb_project,
            normalize_activations=True,
            verbose=False,
            autocast_dtype=t.bfloat16,
            backup_steps=1000000,
        )
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEMO] SAE training completed in {time.time() - training_start:.2f}s")


if __name__ == "__main__":
    """python demo.py --save_dir run2 --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures standard jump_relu batch_top_k top_k gated --use_wandb
    python demo.py --save_dir run3 --model_name google/gemma-2-2b --layers 12 --architectures standard top_k --use_wandb
    python demo.py --save_dir jumprelu --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures jump_relu --use_wandb"""
    args = get_args()

    hf_repo_id = args.hf_repo_id

    if hf_repo_id:
        assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    mp.set_start_method("spawn", force=True)

    config.STREAMING_READ_MAX_RETRIES = 100
    config.STREAMING_READ_RETRY_INTERVAL = 20

    start_time = time.time()

    target_l0s = ast.literal_eval(args.target_l0s) if args.target_l0s else None
    target_l1s = ast.literal_eval(args.target_l1s) if args.target_l1s else None
    lambda_swc2r = ast.literal_eval(args.lambda_swc2r) if args.lambda_swc2r else None

    lambda_suffix = ""
    if lambda_swc2r and lambda_swc2r != [0.0]:
        lambda_suffix += f"_swc2r{lambda_swc2r[0]}"
        lambda_suffix += f"_alpha{args.swc2r_alpha}"
        lambda_suffix += f"_tau{args.swc2r_tau}"
        lambda_suffix += f"_{args.swc2r_type}"

    dtype_suffix = f"_{args.dtype}"

    save_dir = (
        f"{args.save_dir}_{args.model_name}_{'_'.join(args.architectures)}{lambda_suffix}{dtype_suffix}".replace(
            "/", "_"
        )
    )

    for layer in args.layers:
        run_sae_training(
            model_name=args.model_name,
            layer=layer,
            save_dir=save_dir,
            device=args.device,
            architectures=args.architectures,
            num_tokens=args.num_tokens,
            random_seeds=demo_config.random_seeds,
            dictionary_widths=demo_config.dictionary_widths,
            learning_rates=[args.learning_rate],
            dry_run=args.dry_run,
            use_wandb=args.use_wandb,
            save_checkpoints=args.save_checkpoints,
            mixed_dataset=args.mixed_dataset,
            dataset_path=args.dataset_path,
            wandb_project=args.wandb_project,
            target_l0s=target_l0s,
            target_l1s=target_l1s,
            lambda_swc2r=lambda_swc2r,
            swc2r_alpha=args.swc2r_alpha,
            swc2r_tau=args.swc2r_tau,
            swc2r_type=args.swc2r_type,
            aux_loss_start_step=args.aux_loss_start_step,
            aux_loss_interval=args.aux_loss_interval,
            dtype_str=args.dtype,
            decay_start_fraction=args.decay_start_fraction,
            sae_batch_size_override=args.sae_batch_size,
        )

    ae_paths = utils.get_nested_folders(save_dir)

    print(f"Total time: {time.time() - start_time}")
