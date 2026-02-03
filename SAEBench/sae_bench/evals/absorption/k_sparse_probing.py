from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch import nn
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from sae_bench.evals.absorption.common import (
    PROBES_DIR,
    RESULTS_DIR,
    get_or_make_dir,
    load_df_or_run,
    load_dfs_or_run,
    load_or_train_probe,
    load_probe_data_split_or_train,
)
from sae_bench.evals.absorption.probing import LinearProbe, train_multi_probe
from sae_bench.evals.absorption.util import batchify
from sae_bench.evals.absorption.vocab import LETTERS

EPS = 1e-6
SPARSE_PROBING_EXPERIMENT_NAME = "k_sparse_probing"


class KSparseProbe(nn.Module):
    weight: torch.Tensor
    bias: torch.Tensor
    feature_ids: torch.Tensor

    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor, feature_ids: torch.Tensor
    ):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.feature_ids = feature_ids

    @property
    def k(self) -> int:
        return self.weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filtered_acts = (
            x[:, self.feature_ids] if len(x.shape) == 2 else x[self.feature_ids]
        )
        return filtered_acts @ self.weight + self.bias


def train_sparse_multi_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    l1_decay: float = 0.01,
    num_probes: int | None = None,
    batch_size: int = 4096,
    num_epochs: int = 50,
    lr: float = 0.01,
    end_lr: float = 1e-5,
    l2_decay: float = 1e-6,
    show_progress: bool = True,
    verbose: bool = False,
    map_acts: Callable[[torch.Tensor], torch.Tensor] | None = None,
    probe_dim: int | None = None,
) -> LinearProbe:
    """
    Train a multi-probe with L1 regularization on the weights.
    """
    return train_multi_probe(
        x_train,
        y_train,
        num_probes=num_probes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        end_lr=end_lr,
        weight_decay=l2_decay,
        show_progress=show_progress,
        verbose=verbose,
        device=device,
        extra_loss_fn=lambda probe, _x, _y: l1_decay
        * probe.weights.abs().sum(dim=-1).mean(),
        map_acts=map_acts,
        probe_dim=probe_dim,
    )


def _get_sae_acts(
    sae: SAE,
    input_activations: torch.Tensor,
    batch_size: int = 4096,
    sparse_feat_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_acts = []
    for batch in batchify(input_activations, batch_size):
        batch_on_device = batch.to(device=sae.device, dtype=sae.dtype)
        acts = sae.encode(batch_on_device)
        if sparse_feat_ids is not None:
            acts = acts[:, sparse_feat_ids]
        batch_acts.append(acts.cpu())
    return torch.cat(batch_acts)


def train_k_sparse_probes(
    sae: SAE,
    train_labels: list[tuple[str, int]],
    train_activations: torch.Tensor,
    ks: Sequence[int],
    l1_decay: float = 0.01,
    batch_size: int = 4096,
    num_epochs: int = 50,
) -> dict[int, dict[int, KSparseProbe]]:
    """
    Train k-sparse probes for each k in ks.
    Returns a dict of dicts, where the outer dict is indexed by k and the inner dict is the label.
    """
    results: dict[int, dict[int, KSparseProbe]] = defaultdict(dict)
    with torch.no_grad():
        labels = {label for _, label in train_labels}
        sparse_train_y = torch.nn.functional.one_hot(
            torch.tensor([idx for _, idx in train_labels])
        )
        train_activations = train_activations.to(sae.device, dtype=sae.dtype)
    l1_probe = (
        train_sparse_multi_probe(
            train_activations,
            sparse_train_y,
            l1_decay=l1_decay,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=sae.device,
            map_acts=lambda acts: sae.encode(acts.to(sae.device, dtype=sae.dtype)),
            probe_dim=sae.cfg.d_sae,
        )
        .float()
        .cpu()
    )
    with torch.no_grad():
        train_k_y = np.array([idx for _, idx in train_labels])
        with tqdm(total=len(ks) * len(labels), desc="training k-probes") as pbar:
            for k in ks:
                for label in labels:
                    sparse_feat_ids = l1_probe.weights[label].topk(k).indices
                    train_k_x = (
                        _get_sae_acts(
                            sae,
                            train_activations,
                            sparse_feat_ids=sparse_feat_ids,
                            batch_size=batch_size,
                        )
                        .float()
                        .numpy()
                    )
                    sk_probe = LogisticRegression(
                        max_iter=500, class_weight="balanced"
                    ).fit(train_k_x, (train_k_y == label).astype(np.int64))
                    probe = KSparseProbe(
                        weight=torch.tensor(sk_probe.coef_[0]).float(),
                        bias=torch.tensor(sk_probe.intercept_[0]).float(),
                        feature_ids=sparse_feat_ids,
                    )
                    results[k][label] = probe
                    pbar.update(1)
    return results


@torch.inference_mode()
def sae_k_sparse_metadata(
    sae: SAE,
    probe: LinearProbe,
    k_sparse_probes: dict[int, dict[int, KSparseProbe]],
    sae_name: str,
    layer: int,
) -> pd.DataFrame:
    norm_probe_weights = probe.weights / torch.norm(probe.weights, dim=-1, keepdim=True)
    norm_W_enc = sae.W_enc / torch.norm(sae.W_enc, dim=0, keepdim=True)
    norm_W_dec = sae.W_dec / torch.norm(sae.W_dec, dim=-1, keepdim=True)
    probe_dec_cos = (
        (
            norm_probe_weights.to(dtype=norm_W_dec.dtype, device=norm_W_dec.device)
            @ norm_W_dec.T
        )
        .cpu()
        .float()
    )
    probe_enc_cos = (
        (
            norm_probe_weights.to(dtype=norm_W_enc.dtype, device=norm_W_enc.device)
            @ norm_W_enc
        )
        .cpu()
        .float()
    )

    metadata: dict[str, float | str | float | np.ndarray] = {
        "layer": layer,
        "sae_name": sae_name,
    }
    rows = []
    for letter_i, letter in enumerate(LETTERS):
        for k, k_probes in k_sparse_probes.items():
            row = {**metadata}
            k_probe = k_probes[letter_i]
            row["letter"] = letter
            row["k"] = k
            row["feats"] = k_probe.feature_ids.cpu().numpy()
            row["cos_probe_sae_enc"] = probe_enc_cos[
                letter_i, k_probe.feature_ids.cpu()
            ].numpy()
            row["cos_probe_sae_dec"] = probe_dec_cos[
                letter_i, k_probe.feature_ids.cpu()
            ].numpy()
            row["weights"] = k_probe.weight.float().cpu().numpy()
            row["bias"] = k_probe.bias.item()
            rows.append(row)
    return pd.DataFrame(rows)


@torch.inference_mode()
def eval_probe_and_sae_k_sparse_raw_scores(
    sae: SAE,
    probe: LinearProbe,
    k_sparse_probes: dict[int, dict[int, KSparseProbe]],
    eval_labels: list[tuple[str, int]],
    eval_activations: torch.Tensor,
    batch_size: int = 128,
) -> pd.DataFrame:
    probe = probe.to(sae.device)

    for k, k_probes_dict in k_sparse_probes.items():
        for letter_idx, k_probe in k_probes_dict.items():
            k_probe.weight = k_probe.weight.to(sae.device)
            k_probe.bias = k_probe.bias.to(sae.device)
            k_probe.feature_ids = k_probe.feature_ids.to(sae.device)

    columns = defaultdict(list)
    num_samples = len(eval_labels)

    for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating batches"):
        batch_indices = slice(i, min(i + batch_size, num_samples))
        batch_acts = eval_activations[batch_indices].to(sae.device)
        batch_labels = eval_labels[batch_indices]

        batch_probe_scores = probe(batch_acts)

        batch_sae_acts = sae.encode(batch_acts)

        k_sparse_scores = {}
        k_sparse_sums = {}
        k_sparse_acts_list = {}

        for k, k_probes_dict in k_sparse_probes.items():
            k_sparse_scores[k] = {}
            k_sparse_sums[k] = {}
            k_sparse_acts_list[k] = {}

            for letter_i, letter in enumerate(LETTERS):
                k_probe = k_probes_dict[letter_i]
                selected_acts = batch_sae_acts[:, k_probe.feature_ids].float()

                scores = selected_acts @ k_probe.weight + k_probe.bias
                sums = selected_acts.sum(dim=1)

                k_sparse_scores[k][letter] = scores
                k_sparse_sums[k][letter] = sums
                k_sparse_acts_list[k][letter] = selected_acts

        batch_probe_scores_cpu = batch_probe_scores.float().detach().cpu().numpy()

        for k in k_sparse_scores:
            for letter in k_sparse_scores[k]:
                k_sparse_scores[k][letter] = k_sparse_scores[k][letter].detach().cpu().numpy()
                k_sparse_sums[k][letter] = k_sparse_sums[k][letter].detach().cpu().numpy()
                k_sparse_acts_list[k][letter] = k_sparse_acts_list[k][letter].detach().cpu().numpy()

        columns["token"].extend([t for t, _ in batch_labels])
        columns["answer_letter"].extend([LETTERS[idx] for _, idx in batch_labels])

        for letter_i, letter in enumerate(LETTERS):
            columns[f"score_probe_{letter}"].extend(batch_probe_scores_cpu[:, letter_i])
            for k in k_sparse_probes:
                columns[f"score_sparse_sae_{letter}_k_{k}"].extend(k_sparse_scores[k][letter])
                columns[f"sum_sparse_sae_{letter}_k_{k}"].extend(k_sparse_sums[k][letter])
                columns[f"sparse_sae_{letter}_k_{k}_acts"].extend(list(k_sparse_acts_list[k][letter]))

    return pd.DataFrame(columns)


def load_and_run_eval_probe_and_sae_k_sparse_raw_scores(
    sae: SAE,
    model: HookedTransformer,
    layer: int,
    sae_name: str,
    max_k_value: int,
    prompt_template: str,
    prompt_token_pos: int,
    probes_dir: Path | str,
    device: str,
    verbose: bool = True,
    k_sparse_probe_l1_decay: float = 0.01,
    k_sparse_probe_batch_size: int = 4096,
    k_sparse_probe_num_epochs: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if verbose:
        print("Loading probe and training data", flush=True)
    probe = load_or_train_probe(
        model=model,
        layer=layer,
        probes_dir=probes_dir,
        base_template=prompt_template,
        pos_idx=prompt_token_pos,
        device=device,
    )

    train_activations, train_data = load_probe_data_split_or_train(
        model,
        base_template=prompt_template,
        pos_idx=prompt_token_pos,
        probes_dir=probes_dir,
        layer=layer,
        split="train",
        device=device,
    )
    if verbose:
        print("Training k-sparse probes", flush=True)
    k_sparse_probes = train_k_sparse_probes(
        sae,
        train_data,
        train_activations,
        ks=list(range(1, max_k_value + 1)),
        l1_decay=k_sparse_probe_l1_decay,
        batch_size=k_sparse_probe_batch_size,
        num_epochs=k_sparse_probe_num_epochs,
    )
    with torch.no_grad():
        if verbose:
            print("Loading validation data", flush=True)
        eval_activations, eval_data = load_probe_data_split_or_train(
            model,
            base_template=prompt_template,
            pos_idx=prompt_token_pos,
            probes_dir=probes_dir,
            layer=layer,
            split="test",
            device="cpu",
        )
        if verbose:
            print("Evaluating raw k-sparse probing scores", flush=True)
        df = eval_probe_and_sae_k_sparse_raw_scores(
            sae,
            probe,
            k_sparse_probes=k_sparse_probes,
            eval_labels=eval_data,
            eval_activations=eval_activations,
            batch_size=k_sparse_probe_batch_size,
        )
        if verbose:
            print("Building metadata", flush=True)
        metadata = sae_k_sparse_metadata(
            sae,
            probe,
            k_sparse_probes,
            sae_name=sae_name,
            layer=layer,
        )
    return df, metadata


def build_metrics_df(results_df, metadata_df, max_k_value: int):
    aucs = []
    for letter in LETTERS:
        y = (results_df["answer_letter"] == letter).values
        pred_probe = results_df[f"score_probe_{letter}"].values
        auc_probe = metrics.roc_auc_score(y, pred_probe)
        f1_probe = metrics.f1_score(y, pred_probe > 0.0)
        recall_probe = metrics.recall_score(y, pred_probe > 0.0)
        precision_probe = metrics.precision_score(y, pred_probe > 0.0)
        auc_info = {
            "auc_probe": auc_probe,
            "f1_probe": f1_probe,
            "recall_probe": recall_probe,
            "precision_probe": precision_probe,
            "letter": letter,
            "layer": metadata_df["layer"].iloc[0],
            "sae_name": metadata_df["sae_name"].iloc[0],
        }

        for k in range(1, max_k_value + 1):
            pred_sae = results_df[f"score_sparse_sae_{letter}_k_{k}"].values
            auc_sae = metrics.roc_auc_score(y, pred_sae)
            f1 = metrics.f1_score(y, pred_sae > 0.0)
            recall = metrics.recall_score(y, pred_sae > 0.0)
            precision = metrics.precision_score(y, pred_sae > 0.0)
            auc_info[f"auc_sparse_sae_{k}"] = auc_sae
            sum_sae_pred = results_df[f"sum_sparse_sae_{letter}_k_{k}"].values
            auc_sum_sae = metrics.roc_auc_score(y, sum_sae_pred)
            f1_sum_sae = metrics.f1_score(y, sum_sae_pred > EPS)
            recall_sum_sae = metrics.recall_score(y, sum_sae_pred > EPS)
            precision_sum_sae = metrics.precision_score(y, sum_sae_pred > EPS)

            auc_info[f"f1_sparse_sae_{k}"] = f1
            auc_info[f"recall_sparse_sae_{k}"] = recall
            auc_info[f"precision_sparse_sae_{k}"] = precision
            auc_info[f"auc_sum_sparse_sae_{k}"] = auc_sum_sae
            auc_info[f"f1_sum_sparse_sae_{k}"] = f1_sum_sae
            auc_info[f"recall_sum_sparse_sae_{k}"] = recall_sum_sae
            auc_info[f"precision_sum_sparse_sae_{k}"] = precision_sum_sae

            meta_row = metadata_df[
                (metadata_df["letter"] == letter) & (metadata_df["k"] == k)
            ]
            auc_info[f"sparse_sae_k_{k}_feats"] = meta_row["feats"].iloc[0]
            auc_info[f"cos_probe_sae_enc_k_{k}"] = meta_row["cos_probe_sae_enc"].iloc[0]
            auc_info[f"cos_probe_sae_dec_k_{k}"] = meta_row["cos_probe_sae_dec"].iloc[0]
            auc_info[f"sparse_sae_k_{k}_weights"] = meta_row["weights"].iloc[0]
            auc_info[f"sparse_sae_k_{k}_bias"] = meta_row["bias"].iloc[0]
            auc_info["layer"] = meta_row["layer"].iloc[0]
            auc_info["sae_name"] = meta_row["sae_name"].iloc[0]
        aucs.append(auc_info)
    return pd.DataFrame(aucs)


def add_feature_splits_to_metrics_df(
    df: pd.DataFrame,
    max_k_value: int,
    f1_jump_threshold: float = 0.03,
) -> None:
    """
    If a k-sparse probe has a F1 score that increases by `f1_jump_threshold` or more from the previous k-1, consider this to be feature splitting.
    """
    split_feats_by_letter = {}
    for letter in LETTERS:
        prev_best = -100
        df_letter = df[df["letter"] == letter]
        for k in range(1, max_k_value + 1):
            k_score = df_letter[f"f1_sparse_sae_{k}"].iloc[0]
            k_feats = df_letter[f"sparse_sae_k_{k}_feats"].iloc[0].tolist()
            if k_score > prev_best + f1_jump_threshold:
                prev_best = k_score
                split_feats_by_letter[letter] = k_feats
            else:
                break
    df["split_feats"] = df["letter"].apply(
        lambda letter: split_feats_by_letter.get(letter, [])
    )
    df["num_split_features"] = df["split_feats"].apply(len) - 1


def get_sparse_probing_raw_results_filename(sae_name: str, layer: int) -> str:
    return f"layer_{layer}_{sae_name}_raw_results.parquet"


def get_sparse_probing_metadata_filename(sae_name: str, layer: int) -> str:
    return f"layer_{layer}_{sae_name}_metadata.parquet"


def get_sparse_probing_metrics_filename(sae_name: str, layer: int) -> str:
    return f"layer_{layer}_{sae_name}_metrics.parquet"


def run_k_sparse_probing_experiment(
    model: HookedTransformer,
    sae: SAE,
    layer: int,
    sae_name: str,
    max_k_value: int,
    prompt_template: str,
    prompt_token_pos: int,
    device: str,
    experiment_dir: Path | str = RESULTS_DIR / SPARSE_PROBING_EXPERIMENT_NAME,
    probes_dir: Path | str = PROBES_DIR,
    force: bool = False,
    f1_jump_threshold: float = 0.03,
    k_sparse_probe_l1_decay: float = 0.01,
    k_sparse_probe_batch_size: int = 4096,
    k_sparse_probe_num_epochs: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    task_output_dir = get_or_make_dir(experiment_dir) / sae_name
    raw_results_path = task_output_dir / get_sparse_probing_raw_results_filename(
        sae_name, layer
    )
    metadata_results_path = task_output_dir / get_sparse_probing_metadata_filename(
        sae_name, layer
    )
    metrics_results_path = task_output_dir / get_sparse_probing_metrics_filename(
        sae_name, layer
    )

    def get_raw_results_df():
        return load_dfs_or_run(
            lambda: load_and_run_eval_probe_and_sae_k_sparse_raw_scores(
                sae,
                model,
                probes_dir=probes_dir,
                verbose=verbose,
                sae_name=sae_name,
                layer=layer,
                max_k_value=max_k_value,
                prompt_template=prompt_template,
                prompt_token_pos=prompt_token_pos,
                k_sparse_probe_l1_decay=k_sparse_probe_l1_decay,
                k_sparse_probe_batch_size=k_sparse_probe_batch_size,
                k_sparse_probe_num_epochs=k_sparse_probe_num_epochs,
                device=device,
            ),
            (raw_results_path, metadata_results_path),
            force=force,
        )

    metrics_df = load_df_or_run(
        lambda: build_metrics_df(*get_raw_results_df(), max_k_value=max_k_value),
        metrics_results_path,
        force=force,
    )
    add_feature_splits_to_metrics_df(
        metrics_df, max_k_value=max_k_value, f1_jump_threshold=f1_jump_threshold
    )
    return metrics_df
