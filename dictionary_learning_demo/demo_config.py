from dataclasses import dataclass, asdict, field
from typing import Optional, Type, Any
from enum import Enum
import torch as t
import itertools

from dictionary_learning.dictionary_learning.trainers.standard import (
    StandardTrainer,
    StandardTrainerAprilUpdate,
)
from dictionary_learning.dictionary_learning.trainers.top_k import (
    TopKTrainer,
    AutoEncoderTopK,
)
from dictionary_learning.dictionary_learning.trainers.batch_top_k import (
    BatchTopKTrainer,
    BatchTopKSAE,
)
from dictionary_learning.dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKTrainer,
    MatryoshkaBatchTopKSAE,
)
from dictionary_learning.dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)
from dictionary_learning.dictionary_learning.trainers.ort import OrtTrainer


class TrainerType(Enum):
    STANDARD = "standard"
    STANDARD_NEW = "standard_new"
    TOP_K = "top_k"
    BATCH_TOP_K = "batch_top_k"
    GATED = "gated"
    P_ANNEAL = "p_anneal"
    JUMP_RELU = "jump_relu"
    ORT = "ort"
    Matryoshka_BATCH_TOP_K = "matryoshka_batch_top_k"


@dataclass
class LLMConfig:
    llm_batch_size: int
    context_length: int
    sae_batch_size: int
    dtype: t.dtype


@dataclass
class SparsityPenalties:
    standard: list[float]
    standard_new: list[float]
    p_anneal: list[float]
    gated: list[float]


    lambda_swc2r: list[float]
    swc2r_tau: list[float]
    swc2r_type: list[str]


num_tokens = 500_000_000

print(f"NOTE: Training on {num_tokens} tokens")

eval_num_inputs = 200
random_seeds = [0]
dictionary_widths = [65536]

WARMUP_STEPS = 1000
SPARSITY_WARMUP_STEPS = 5000
DECAY_START_FRACTION = 0.8
K_ANNEAL_END_FRACTION = 0.01
remove_bos = True
max_activation_norm_multiple = 10

learning_rates = [2e-4]


wandb_project = "qwen-8b-sweep"

LLM_CONFIG = {
    "EleutherAI/pythia-70m-deduped": LLMConfig(
        llm_batch_size=64, context_length=1024, sae_batch_size=2048, dtype=t.float32
    ),
    "EleutherAI/pythia-160m-deduped": LLMConfig(
        llm_batch_size=32, context_length=1024, sae_batch_size=2048, dtype=t.float32
    ),
    "google/gemma-2-2b": LLMConfig(
        llm_batch_size=16, context_length=1024, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "google/gemma-2-9b": LLMConfig(
        llm_batch_size=16, context_length=1024, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "Qwen/Qwen2.5-Coder-32B-Instruct": LLMConfig(
        llm_batch_size=4, context_length=2048, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "Qwen/Qwen3-8B": LLMConfig(
        llm_batch_size=16, context_length=2048, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "Qwen/Qwen3-14B": LLMConfig(
        llm_batch_size=8, context_length=2048, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "Qwen/Qwen3-32B": LLMConfig(
        llm_batch_size=2, context_length=2048, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "Qwen/Qwen3-1.7B": LLMConfig(
        llm_batch_size=8, context_length=1024, sae_batch_size=8192, dtype=t.bfloat16
    ),
}

SPARSITY_PENALTIES = SparsityPenalties(
    standard=[0.0],
    standard_new=[0.0],
    p_anneal=[0.006, 0.008, 0.01, 0.015, 0.02, 0.025],
    gated=[0.012, 0.018, 0.024, 0.04, 0.06, 0.08],
    lambda_swc2r=[0.0],
    swc2r_tau=[0.95],
    swc2r_type=['topTauPerFeatSquare'],
)


TARGET_L0s = [20, 40, 80, 160, 320]


@dataclass
class BaseTrainerConfig:
    activation_dim: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    trainer: Type[Any]
    dict_class: Type[Any]
    wandb_name: str
    warmup_steps: int
    steps: int
    decay_start: Optional[int]
    buffer_tokens: int
    batch_tokens: int
    dtype: t.dtype
    aux_loss_start_step: int
    aux_loss_interval: int


@dataclass
class StandardTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]
    resample_steps: Optional[int] = None
    c2r_penalty: float = 0.0
    swc2r_penalty: float = 0.0
    swc2r_alpha: float = 1.0
    swc2r_tau: float = 0.95
    swc2r_type: str = "topTauPerFeatSquare"



@dataclass
class StandardNewTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]
    c2r_penalty: float = 0.0
    swc2r_penalty: float = 0.0
    swc2r_alpha: float = 1.0
    swc2r_tau: float = 0.95
    swc2r_type: str = "topTauPerFeatSquare"



@dataclass
class PAnnealTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    initial_sparsity_penalty: float
    sparsity_warmup_steps: Optional[int]
    sparsity_function: str = "Lp^p"
    p_start: float = 1.0
    p_end: float = 0.2
    anneal_start: int = 10000
    anneal_end: Optional[int] = None
    sparsity_queue_length: int = 10
    n_sparsity_updates: int = 10
    swc2r_penalty: float = 0.0


@dataclass
class TopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    k_anneal_steps: Optional[int] = None
    c2r_penalty: float = 0.0
    swc2r_penalty: float = 0.0
    swc2r_alpha: float = 1.0
    swc2r_tau: float = 0.95
    swc2r_type: str = "topTauPerFeatSquare"



@dataclass
class MatryoshkaBatchTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    group_fractions: list[float] = field(
        default_factory=lambda: [
            (0.03125), (0.0625), (0.125), (0.25), (0.53125)
        ]
    )
    group_weights: Optional[list[float]] = None
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    k_anneal_steps: Optional[int] = None
    c2r_penalty: float = 0.0
    swc2r_penalty: float = 0.0
    swc2r_alpha: float = 1.0
    swc2r_tau: float = 0.95
    swc2r_type: str = "topTauPerFeatSquare"



@dataclass
class GatedTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]
    c2r_penalty: float = 0.0
    swc2r_penalty: float = 0.0
    swc2r_alpha: float = 1.0
    swc2r_tau: float = 0.95
    swc2r_type: str = "topTauPerFeatSquare"



@dataclass
class JumpReluTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    target_l0: int
    sparsity_warmup_steps: Optional[int]
    sparsity_penalty: float = 1.0
    bandwidth: float = 0.001
    c2r_penalty: float = 0.0
    swc2r_penalty: float = 0.0
    swc2r_alpha: float = 1.0
    swc2r_tau: float = 0.95
    swc2r_type: str = "topTauPerFeatSquare"



@dataclass
class OrtTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    orthogonality_penalty: float
    chunk_size: int
    k_anneal_steps: Optional[int] = None
    c2r_penalty: float = 0.0
    swc2r_penalty: float = 0.0
    swc2r_alpha: float = 1.0
    swc2r_tau: float = 0.95
    swc2r_type: str = "topTauPerFeatSquare"



def get_trainer_configs(
    architectures: list[str],
    learning_rates: list[float],
    seeds: list[int],
    activation_dim: int,
    dict_sizes: list[int],
    model_name: str,
    device: str,
    layer: str,
    submodule_name: str,
    steps: int,
    warmup_steps: int = WARMUP_STEPS,
    sparsity_warmup_steps: int = SPARSITY_WARMUP_STEPS,
    decay_start_fraction=DECAY_START_FRACTION,
    anneal_end_fraction=K_ANNEAL_END_FRACTION,
    buffer_tokens: int = 256000,
    batch_tokens: int = 2048,
    dtype: t.dtype = t.float32,
    aux_loss_start_step: int = 0,
    aux_loss_interval: int = 1,
    target_l0s: list[int] = None,
    target_l1s: list[float] = None,
    swc2r_alpha: float = 1.0,
    swc2r_tau: float = 0.95,
    swc2r_type: str = "topTauPerFeatSquare",
) -> list[dict]:
    if target_l0s is None:
        target_l0s = TARGET_L0s

    if target_l1s is not None:
        SPARSITY_PENALTIES.standard_new = target_l1s

    decay_start = int(steps * decay_start_fraction) if decay_start_fraction is not None else None
    anneal_end = int(steps * anneal_end_fraction)

    trainer_configs = []

    base_config = {
        "activation_dim": activation_dim,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": submodule_name,
        "buffer_tokens": buffer_tokens,
        "batch_tokens": batch_tokens,
        "dtype": dtype,
        "aux_loss_start_step": aux_loss_start_step,
        "aux_loss_interval": aux_loss_interval,
    }
    if TrainerType.P_ANNEAL.value in architectures:
        for seed, dict_size, learning_rate, sparsity_penalty, swc2r_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.p_anneal, SPARSITY_PENALTIES.lambda_swc2r
        ):
            config = PAnnealTrainerConfig(
                **base_config,
                trainer=PAnnealTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                initial_sparsity_penalty=sparsity_penalty,
                swc2r_penalty=swc2r_penalty,
                wandb_name=f"PAnnealTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty, swc2r_penalty, swc2r_tau, swc2r_type in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard, SPARSITY_PENALTIES.lambda_swc2r, SPARSITY_PENALTIES.swc2r_tau, SPARSITY_PENALTIES.swc2r_type
        ):
            if swc2r_penalty == 0:
                wandb_name = f"StandardTrainer-{model_name}-{submodule_name}"
            else:
                wandb_name = f"CoStandardTrainer-{model_name}-{submodule_name}"

            config = StandardTrainerConfig(
                **base_config,
                trainer=StandardTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                swc2r_penalty=swc2r_penalty,
                swc2r_alpha=swc2r_alpha,
                swc2r_tau=swc2r_tau,
                swc2r_type=swc2r_type,
                wandb_name=wandb_name,
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD_NEW.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty, swc2r_penalty, swc2r_tau, swc2r_type,  in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard_new, SPARSITY_PENALTIES.lambda_swc2r, SPARSITY_PENALTIES.swc2r_tau, SPARSITY_PENALTIES.swc2r_type
        ):
            if swc2r_penalty == 0 :
                wandb_name = f"StandardTrainerNew-{model_name}-{submodule_name}"
            else:
                wandb_name = f"CoStandardTrainerNew-{model_name}-{submodule_name}"

            config = StandardNewTrainerConfig(
                **base_config,
                trainer=StandardTrainerAprilUpdate,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                swc2r_penalty=swc2r_penalty,
                swc2r_alpha=swc2r_alpha,
                swc2r_tau=swc2r_tau,
                swc2r_type=swc2r_type,
                wandb_name=wandb_name,
            )
            trainer_configs.append(asdict(config))

    if TrainerType.GATED.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty, swc2r_penalty, swc2r_tau, swc2r_type in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.gated, SPARSITY_PENALTIES.lambda_swc2r, SPARSITY_PENALTIES.swc2r_tau, SPARSITY_PENALTIES.swc2r_type
        ):
            if swc2r_penalty == 0:
                wandb_name = f"GatedTrainer-{model_name}-{submodule_name}"
            else:
                wandb_name = f"CoGatedTrainer-{model_name}-{submodule_name}"

            config = GatedTrainerConfig(
                **base_config,
                trainer=GatedSAETrainer,
                dict_class=GatedAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                swc2r_penalty=swc2r_penalty,
                swc2r_alpha=swc2r_alpha,
                swc2r_tau=swc2r_tau,
                swc2r_type=swc2r_type,
                wandb_name=wandb_name,
            )
            trainer_configs.append(asdict(config))

    if TrainerType.TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k, swc2r_penalty, swc2r_tau, swc2r_type in itertools.product(
            seeds, dict_sizes, learning_rates, target_l0s, SPARSITY_PENALTIES.lambda_swc2r, SPARSITY_PENALTIES.swc2r_tau, SPARSITY_PENALTIES.swc2r_type
        ):
            if swc2r_penalty == 0:
                wandb_name = f"TopKTrainer-{model_name}-{submodule_name}"
            else:
                wandb_name = f"CoTopKTrainer-{model_name}-{submodule_name}"

            config = TopKTrainerConfig(
                **base_config,
                trainer=TopKTrainer,
                dict_class=AutoEncoderTopK,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                swc2r_penalty=swc2r_penalty,
                swc2r_alpha=swc2r_alpha,
                swc2r_tau=swc2r_tau,
                swc2r_type=swc2r_type,
                k_anneal_steps=anneal_end,
                wandb_name=wandb_name,
            )
            trainer_configs.append(asdict(config))

    if TrainerType.BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k, swc2r_penalty, swc2r_tau, swc2r_type in itertools.product(
            seeds, dict_sizes, learning_rates, target_l0s, SPARSITY_PENALTIES.lambda_swc2r, SPARSITY_PENALTIES.swc2r_tau, SPARSITY_PENALTIES.swc2r_type
        ):
            if swc2r_penalty == 0:
                wandb_name = f"BatchTopKTrainer-{model_name}-{submodule_name}"
            else:
                wandb_name = f"CoBatchTopKTrainer-{model_name}-{submodule_name}"

            config = TopKTrainerConfig(
                **base_config,
                trainer=BatchTopKTrainer,
                dict_class=BatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                swc2r_penalty=swc2r_penalty,
                swc2r_alpha=swc2r_alpha,
                swc2r_tau=swc2r_tau,
                swc2r_type=swc2r_type,
                k_anneal_steps=anneal_end,
                wandb_name=wandb_name,
            )
            trainer_configs.append(asdict(config))

    if TrainerType.Matryoshka_BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k, swc2r_penalty, swc2r_tau, swc2r_type in itertools.product(
            seeds, dict_sizes, learning_rates, target_l0s, SPARSITY_PENALTIES.lambda_swc2r, SPARSITY_PENALTIES.swc2r_tau, SPARSITY_PENALTIES.swc2r_type
        ):
            if swc2r_penalty == 0:
                wandb_name = f"MatryoshkaBatchTopKTrainer-{model_name}-{submodule_name}"
            else:
                wandb_name = f"CoMatryoshkaBatchTopKTrainer-{model_name}-{submodule_name}"

            config = MatryoshkaBatchTopKTrainerConfig(
                **base_config,
                trainer=MatryoshkaBatchTopKTrainer,
                dict_class=MatryoshkaBatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                swc2r_penalty=swc2r_penalty,
                swc2r_alpha=swc2r_alpha,
                swc2r_tau=swc2r_tau,
                swc2r_type=swc2r_type,
                k_anneal_steps=anneal_end,
                wandb_name=wandb_name,
            )
            trainer_configs.append(asdict(config))

    if TrainerType.JUMP_RELU.value in architectures:
        for seed, dict_size, learning_rate, target_l0, swc2r_penalty, swc2r_tau, swc2r_type in itertools.product(
            seeds, dict_sizes, learning_rates, target_l0s, SPARSITY_PENALTIES.lambda_swc2r, SPARSITY_PENALTIES.swc2r_tau, SPARSITY_PENALTIES.swc2r_type
        ):
            if swc2r_penalty == 0:
                wandb_name = f"JumpReluTrainer-{model_name}-{submodule_name}"
            else:
                wandb_name = f"CoJumpReluTrainer-{model_name}-{submodule_name}"

            config = JumpReluTrainerConfig(
                **base_config,
                trainer=JumpReluTrainer,
                dict_class=JumpReluAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                target_l0=target_l0,
                swc2r_penalty=swc2r_penalty,
                swc2r_alpha=swc2r_alpha,
                swc2r_tau=swc2r_tau,
                swc2r_type=swc2r_type,
                wandb_name=wandb_name,
            )
            trainer_configs.append(asdict(config))

    if TrainerType.ORT.value in architectures:
        for seed, dict_size, learning_rate, k, swc2r_penalty, swc2r_tau, swc2r_type in itertools.product(
            seeds, dict_sizes, learning_rates, target_l0s, SPARSITY_PENALTIES.lambda_swc2r, SPARSITY_PENALTIES.swc2r_tau, SPARSITY_PENALTIES.swc2r_type
        ):
            if swc2r_penalty == 0:
                wandb_name = f"OrtTrainer-{model_name}-{submodule_name}"
            else:
                wandb_name = f"CoOrtTrainer-{model_name}-{submodule_name}"

            config = OrtTrainerConfig(
                **base_config,
                trainer=OrtTrainer,
                dict_class=BatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                orthogonality_penalty=0.25,
                chunk_size=8192,
                k_anneal_steps=anneal_end,
                swc2r_penalty=swc2r_penalty,
                swc2r_alpha=swc2r_alpha,
                swc2r_tau=swc2r_tau,
                swc2r_type=swc2r_type,
                wandb_name=wandb_name,
            )
            trainer_configs.append(asdict(config))
    return trainer_configs
