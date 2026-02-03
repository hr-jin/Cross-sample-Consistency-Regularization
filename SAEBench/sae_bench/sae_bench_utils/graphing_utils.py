

import json
import os
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy import stats





plt.rcParams.update({"font.size": 20})


def get_best_results(
    results_dict: dict[str, dict[str, float]], results_path: str, ks: list[int]
) -> dict[str, dict[str, float]]:
    best_results = {}
    for sae, data in results_dict.items():
        best_results[sae] = 0
        for k in ks:
            custom_metric, _ = get_custom_metric_key_and_name(results_path, k)
            if custom_metric in data:
                best_results[sae] = max(best_results[sae], data[custom_metric])
            else:
                print(f"Custom metric {custom_metric} not found for {sae}")

    for sae in best_results.keys():
        results_dict[sae]["best_custom_metric"] = best_results[sae]

    return results_dict


def get_single_figure(
    selected_saes: list[tuple[str, str]],
    results_path: str,
    core_results_path: str,
    image_base_name: str,
    k: int | None = None,
    trainer_markers: dict[str, str] | None = None,
    title: str | None = None,
    title_prefix: str = "",
    plot_type: bool = True,
    baseline_sae: tuple[str, str] | None = None,
    baseline_label: str | None = None,
):
    eval_results = get_eval_results(selected_saes, results_path)
    core_results = get_core_results(selected_saes, core_results_path)

    for sae in eval_results:
        eval_results[sae].update(core_results[sae])

    custom_metric, custom_metric_name = get_custom_metric_key_and_name(results_path, k)

    if baseline_sae:
        baseline_results = get_eval_results([baseline_sae], results_path)
        baseline_id = f"{baseline_sae[0]}_{baseline_sae[1]}"
        baseline_results[baseline_id].update(
            get_core_results([baseline_sae], core_results_path)[baseline_id]
        )
        baseline_value = baseline_results[baseline_id][custom_metric]
        assert baseline_label, "Please provide a label for the baseline"
    else:
        baseline_value = None
        assert baseline_label is None, "Please do not provide a label for the baseline"

    if not title:
        title = f"{title_prefix}L0 vs {custom_metric_name}"

    if plot_type:
        fig = plot_2var_graph(
            eval_results,
            custom_metric,
            y_label=custom_metric_name,
            title=title,
            output_filename=f"{image_base_name}_2var_sae_type.png",
            trainer_markers=trainer_markers,
            return_fig=True,
            baseline_value=baseline_value,
            baseline_label=baseline_label,
        )
    else:
        fig = plot_2var_graph_dict_size(
            eval_results,
            custom_metric,
            y_label=custom_metric_name,
            title=title,
            output_filename=f"{image_base_name}_2var_dict_size.png",
            return_fig=True,
            baseline_value=baseline_value,
            baseline_label=baseline_label,
        )

    return fig


def plot_results(
    eval_filenames: list[str],
    core_filenames: list[str],
    eval_type: str,
    image_base_name: str,
    k: int | None = None,
    trainer_markers: dict[str, str] | None = None,
    trainer_colors: dict[str, str] | None = None,
    title_prefix: str = "",
    return_fig: bool = False,
    baseline_sae_path: str | None = None,
    baseline_label: str | None = None,
    connect_points: bool = False,
):
    eval_results = get_eval_results(eval_filenames)
    core_results = get_core_results(core_filenames)
    print("\n\n\ncore_results:")
    for k,v in core_results.items():
        print(k)
        print(v)
        print('\n')

    for sae in eval_results:
        eval_results[sae].update(core_results[sae])

    custom_metric, custom_metric_name = get_custom_metric_key_and_name(eval_type, k)

    if baseline_sae_path:
        baseline_results = get_eval_results([baseline_sae_path])

        baseline_filename = os.path.basename(baseline_sae_path)
        baseline_results_key = baseline_filename.replace("_eval_results.json", "")

        core_baseline_filename = baseline_sae_path.replace(eval_type, "core")

        baseline_results[baseline_results_key].update(
            get_core_results([core_baseline_filename])[baseline_results_key]
        )

        baseline_value = baseline_results[baseline_results_key][custom_metric]
        assert baseline_label, "Please provide a label for the baseline"
    else:
        baseline_value = None
        assert baseline_label is None, "Please do not provide a label for the baseline"


    title_2var = f"{title_prefix}L0 vs {custom_metric_name}"
    fig = plot_2var_graph(
        eval_results,
        custom_metric,
        y_label=custom_metric_name,
        title=title_2var,
        output_filename=f"{image_base_name}_2var_sae_type.png",
        trainer_markers=trainer_markers,
        trainer_colors=trainer_colors,
        baseline_value=baseline_value,
        baseline_label=baseline_label,
        connect_points=connect_points,
    )


    if return_fig:
        return fig


def plot_best_of_ks_results(
    selected_saes: list[tuple[str, str]],
    results_path: str,
    core_results_path: str,
    image_base_name: str,
    ks: list[int],
    trainer_markers: dict[str, str] | None = None,
    title_prefix: str = "",
    baseline_sae: tuple[str, str] | None = None,
    baseline_label: str | None = None,
):
    dummy_k = 0

    eval_results = get_eval_results(selected_saes, results_path)
    core_results = get_core_results(selected_saes, core_results_path)

    for sae in eval_results:
        eval_results[sae].update(core_results[sae])

    custom_metric, custom_metric_name = get_custom_metric_key_and_name(
        results_path, dummy_k
    )

    custom_metric = "best_custom_metric"
    custom_metric_name = custom_metric_name.replace(f"Top {dummy_k}", f"Best of {ks}")
    eval_results = get_best_results(eval_results, results_path, ks)

    if baseline_sae:
        baseline_results = get_eval_results([baseline_sae], results_path)
        baseline_id = f"{baseline_sae[0]}_{baseline_sae[1]}"
        baseline_results[baseline_id].update(
            get_core_results([baseline_sae], core_results_path)[baseline_id]
        )

        baseline_results = get_best_results(baseline_results, results_path, ks)
        baseline_value = baseline_results[baseline_id]["best_custom_metric"]
        assert baseline_label, "Please provide a label for the baseline"
    else:
        baseline_value = None
        assert baseline_label is None, "Please do not provide a label for the baseline"

    title_3var = f"{title_prefix}L0 vs Loss Recovered vs {custom_metric_name}"
    title_2var = f"{title_prefix}L0 vs {custom_metric_name}"

    plot_3var_graph(
        eval_results,
        title_3var,
        custom_metric,
        colorbar_label="Custom Metric",
        output_filename=f"{image_base_name}_3var.png",
        trainer_markers=trainer_markers,
    )
    plot_2var_graph(
        eval_results,
        custom_metric,
        y_label=custom_metric_name,
        title=title_2var,
        output_filename=f"{image_base_name}_2var_sae_type.png",
        trainer_markers=trainer_markers,
        baseline_value=baseline_value,
        baseline_label=baseline_label,
    )

    plot_2var_graph_dict_size(
        eval_results,
        custom_metric,
        y_label=custom_metric_name,
        title=title_2var,
        output_filename=f"{image_base_name}_2var_dict_size.png",
        baseline_value=baseline_value,
        baseline_label=baseline_label,
    )


def get_custom_metric_key_and_name(
    eval_path: str, k: int | None = None
) -> tuple[str, str]:
    if "tpp" in eval_path:
        custom_metric = f"tpp_threshold_{k}_total_metric"
        custom_metric_name = f"TPP Top {k} Metric"
    elif "scr" in eval_path:
        custom_metric = f"scr_metric_threshold_{k}"
        custom_metric_name = f"SCR Top {k} Metric"
    elif "sparse_probing" in eval_path:
        custom_metric = f"sae_top_{k}_test_accuracy"
        custom_metric_name = f"Sparse Probing Top {k} Test Accuracy"
    elif "absorption" in eval_path:
        custom_metric = "mean_absorption_fraction_score"
        custom_metric_name = "Mean Absorption Score"
    elif "autointerp" in eval_path:
        custom_metric = "autointerp_score"
        custom_metric_name = "Autointerp Score"
    elif "unlearning" in eval_path:
        custom_metric = "unlearning_score"
        custom_metric_name = "Unlearning Score"
    elif "core" in eval_path:
        custom_metric = "ce_loss_score"
        custom_metric_name = "Loss Recovered"
    elif "ravel" in eval_path:
        custom_metric = "disentanglement_score"
        custom_metric_name = "Disentanglement Score"
    elif "fade" in eval_path:
        custom_metric = "fade_faithfulness"
        custom_metric_name = "FADE Faithfulness"
    else:
        raise ValueError("Please add the correct key for the custom metric")

    return custom_metric, custom_metric_name


def get_sae_bench_train_tokens(filename: str) -> int:
    """This is for SAE Bench internal use. The SAE cfg does not contain the number of training tokens, so we need to hardcode it."""

    if "sae_bench" not in filename:
        raise ValueError("This function is only for SAE Bench releases")

    batch_size = 2048

    if "step" not in filename:
        steps = 244140
        return steps * batch_size
    else:
        match = re.search(r"step_(\d+)", filename)
        if match:
            step = int(match.group(1))
            return step * batch_size
        else:
            raise ValueError("No step match found")


def get_d_sae_string(d_sae: int) -> str:
    rounded_d_sae = round(d_sae / 1000) * 1000

    if rounded_d_sae == 66000:
        rounded_d_sae = 65000

    if rounded_d_sae >= 1_000_000 and rounded_d_sae <= 1_060_000:
        return "1M"
    else:
        return f"{rounded_d_sae // 1000}k"


def get_eval_results(eval_filenames: list[str]) -> dict[str, dict]:
    """eval_filenames is assumed to be a list of filenames of this format:
    {sae_release}_{sae_id}_eval_results.json"""
    eval_results = {}
    for filepath in eval_filenames:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        with open(filepath) as f:
            single_sae_results = json.load(f)

        filename = os.path.basename(filepath)
        if "_ae.pt" in filename:
            results_key = filename.split("_ae.pt")[0] + "_ae.pt"
        elif "_checkpoints" in filename:
            results_key = filename.split("_checkpoints")[0] + "_checkpoints"
        else:
            results_key = filename.replace("_eval_results.json", "")

        if "tpp" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"][
                "tpp_metrics"
            ]
        elif "scr" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"][
                "scr_metrics"
            ]
        elif "absorption" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"][
                "mean"
            ]
        elif "autointerp" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"][
                "autointerp"
            ]
        elif "sparse_probing" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"]["sae"]
        elif "unlearning" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"][
                "unlearning"
            ]
        elif "core" in filepath:
            core_results = single_sae_results["eval_result_metrics"]
            eval_results[results_key] = {}
            for parent_key, child_dict in core_results.items():
                for metric_key, value in child_dict.items():
                    eval_results[results_key][metric_key] = value
        elif "ravel" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"]["ravel"]
        elif "fade" in filepath:
            metrics = {"clarity": [], "responsiveness": [], "purity": [], "faithfulness": []}
            for scores in single_sae_results.values():
                if not isinstance(scores, list): continue
                if len(scores) >= 1 and scores[0] is not None: metrics["clarity"].append(scores[0])
                if len(scores) >= 2 and scores[1] is not None: metrics["responsiveness"].append(scores[1])
                if len(scores) >= 3 and scores[2] is not None: metrics["purity"].append(scores[2])
                if len(scores) >= 4 and scores[3] is not None: metrics["faithfulness"].append(scores[3])

            fade_clarity = np.mean(metrics["clarity"]) if metrics["clarity"] else 0
            fade_responsiveness = np.mean(metrics["responsiveness"]) if metrics["responsiveness"] else 0
            fade_purity = np.mean(metrics["purity"]) if metrics["purity"] else 0
            fade_faithfulness = np.mean(metrics["faithfulness"]) if metrics["faithfulness"] else 0

            eval_results[results_key] = {
                "fade_clarity": fade_clarity,
                "fade_responsiveness": fade_responsiveness,
                "fade_purity": fade_purity,
                "fade_faithfulness": fade_faithfulness,
                "fade_average": (fade_clarity + fade_responsiveness + fade_purity + fade_faithfulness) / 4,
            }

            try:
                parts = filename.split('_')
                d_sae = int(parts[5])
            except:
                d_sae = 0

            try:
                sae_class = filename.split("___")[1]
            except:
                sae_class = "unknown"

            eval_results[results_key]["eval_config"] = {}
            eval_results[results_key]["sae_class"] = sae_class
            eval_results[results_key]["d_sae"] = get_d_sae_string(d_sae)
            eval_results[results_key]["train_tokens"] = 1e-6

            continue

        else:
            raise ValueError("Please add the correct key for the eval results")

        eval_results[results_key]["eval_config"] = single_sae_results["eval_config"]

        sae_config = single_sae_results["sae_cfg_dict"]

        eval_results[results_key]["sae_class"] = sae_config["architecture"]

        eval_results[results_key]["d_sae"] = get_d_sae_string(sae_config["d_sae"])

        if "training_tokens" in sae_config:
            eval_results[results_key]["train_tokens"] = sae_config["training_tokens"]
        else:
            eval_results[results_key]["train_tokens"] = 1e-6

    return eval_results


def get_core_results(core_filenames: list[str]) -> dict:
    core_results = {}
    for filepath in core_filenames:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        with open(filepath) as f:
            single_sae_results = json.load(f)

        l0 = single_sae_results["eval_result_metrics"]["sparsity"]["l0"]
        ce_score = single_sae_results["eval_result_metrics"][
            "model_performance_preservation"
        ]["ce_loss_score"]
        kl_div_score = single_sae_results["eval_result_metrics"][
            "model_behavior_preservation"
        ]["kl_div_score"]

        filename = os.path.basename(filepath)
        if "_ae.pt" in filename:
            results_key = filename.split("_ae.pt")[0] + "_ae.pt"
        elif "_checkpoints" in filename:
            results_key = filename.split("_checkpoints")[0] + "_checkpoints"
        else:
            results_key = filename.replace("_eval_results.json", "")
        core_results[results_key] = {"l0": l0, "ce_loss_score": ce_score, "kl_div_score": kl_div_score}
    return core_results


def find_eval_results_files(folders: list[str]) -> list[str]:
    """
    Recursively explores the given list of folder names and finds all file paths
    containing 'eval_results.json'. Returns a list of the full file paths.

    Args:
        folders (list[str]): A list of folder names to explore.

    Returns:
        list[str]: A list of full file paths containing 'eval_results.json'.
    """
    result_files = []

    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if "eval_results.json" in file:
                    result_files.append(os.path.join(root, file))

    return result_files


def update_trainer_markers_and_colors(
    results: dict[str, dict[str, Any]],
    trainer_markers: dict[str, str],
    trainer_colors: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    trainer_markers = deepcopy(trainer_markers)
    trainer_colors = deepcopy(trainer_colors)

    available_markers = list(set(trainer_markers.values()))
    available_colors = list(set(trainer_colors.values()))

    all_trainers = {v["sae_class"] for v in results.values()}

    for trainer in all_trainers:
        if trainer in trainer_markers and trainer_markers[trainer] in available_markers:
            available_markers.remove(trainer_markers[trainer])
        if trainer in trainer_colors and trainer_colors[trainer] in available_colors:
            available_colors.remove(trainer_colors[trainer])

    for trainer in all_trainers:
        if trainer not in trainer_markers:
            if available_markers:
                trainer_markers[trainer] = available_markers.pop(0)
            else:
                trainer_markers[trainer] = "o"

    for trainer in all_trainers:
        if trainer not in trainer_colors:
            if available_colors:
                trainer_colors[trainer] = available_colors.pop(0)
            else:
                trainer_colors[trainer] = "#000000"

    return trainer_markers, trainer_colors


def plot_3var_graph(
    results: dict[str, dict[str, float]],
    title: str,
    custom_metric: str,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    colorbar_label: str = "Average Diff",
    output_filename: str | None = None,
    legend_location: str = "lower right",
    x_axis_key: str = "l0",
    y_axis_key: str = "ce_loss_score",
    trainer_markers: dict[str, str] | None = None,
):
    if not trainer_markers:
        trainer_markers = TRAINER_MARKERS

    trainer_markers, _ = update_trainer_markers_and_colors(
        results, trainer_markers, TRAINER_COLORS
    )

    l0_values = [data[x_axis_key] for data in results.values()]
    frac_recovered_values = [data[y_axis_key] for data in results.values()]
    custom_metric_values = [data[custom_metric] for data in results.values()]

    fig, ax = plt.subplots(figsize=(10, 6))

    norm = Normalize(vmin=min(custom_metric_values), vmax=max(custom_metric_values))

    handles, labels = [], []

    for trainer, marker in trainer_markers.items():
        trainer_data = {k: v for k, v in results.items() if v["sae_class"] == trainer}

        if not trainer_data:
            continue

        l0_values = [data[x_axis_key] for data in trainer_data.values()]
        frac_recovered_values = [data[y_axis_key] for data in trainer_data.values()]
        custom_metric_values = [data[custom_metric] for data in trainer_data.values()]

        scatter = ax.scatter(
            l0_values,
            frac_recovered_values,
            c=custom_metric_values,
            cmap="viridis",
            marker=marker,
            s=100,
            label=trainer,
            norm=norm,
            edgecolor="black",
        )

        _handle, _ = scatter.legend_elements(prop="sizes")
        _handle[0].set_markeredgecolor("black")
        _handle[0].set_markerfacecolor("white")
        _handle[0].set_markersize(10)
        if marker == "d":
            _handle[0].set_markersize(13)
        handles += _handle

        if trainer in TRAINER_LABELS:
            trainer_label = TRAINER_LABELS[trainer]
        else:
            trainer_label = trainer.capitalize()

        labels.append(trainer_label)

    fig.colorbar(scatter, ax=ax, label=colorbar_label)

    ax.set_xlabel("L0 (Sparsity)")
    ax.set_ylabel("Loss Recovered (Fidelity)")
    ax.set_title(title)

    ax.legend(handles, labels, loc=legend_location)

    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")
    plt.show()


def plot_interactive_3var_graph(
    results: dict[str, dict[str, float]],
    custom_color_metric: str,
    xlims: tuple[float, float] | None = None,
    y_lims: tuple[float, float] | None = None,
    output_filename: str | None = None,
    x_axis_key: str = "l0",
    y_axis_key: str = "ce_loss_score",
    title: str = "",
):
    ae_paths = list(results.keys())
    l0_values = [data[x_axis_key] for data in results.values()]
    frac_recovered_values = [data[y_axis_key] for data in results.values()]

    custom_metric_value = [data[custom_color_metric] for data in results.values()]

    dict_size = [data["d_sae"] for data in results.values()]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=l0_values,
            y=frac_recovered_values,
            mode="markers",
            marker=dict(
                size=10,
                color=custom_metric_value,
                colorscale="Viridis",
                showscale=True,
            ),
            text=[
                f"AE Path: {ae}<br>L0: {l0:.4f}<br>Frac Recovered: {fr:.4f}<br>Custom Metric: {ad:.4f}<br>Dict Size: {d}"
                for ae, l0, fr, ad, d in zip(
                    ae_paths,
                    l0_values,
                    frac_recovered_values,
                    custom_metric_value,
                    dict_size,
                )
            ],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="L0 (Sparsity)",
        yaxis_title="Loss Recovered (Fidelity)",
        hovermode="closest",
    )

    if xlims:
        fig.update_xaxes(range=xlims)
    if y_lims:
        fig.update_yaxes(range=y_lims)

    if output_filename:
        fig.write_html(output_filename)

    fig.show()


def plot_2var_graph(
    results: dict[str, dict[str, float]],
    custom_metric: str,
    title: str = "L0 vs Custom Metric",
    y_label: str = "Custom Metric",
    larger_better: bool = True,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    output_filename: str | None = None,
    baseline_value: float | None = None,
    baseline_label: str | None = None,
    x_axis_key: str = "l0",
    x_label: str = "L0 (Sparsity)",
    x_larger_better: bool = False,
    trainer_markers: dict[str, str] | None = None,
    trainer_colors: dict[str, str] | None = None,
    return_fig: bool = False,
    connect_points: bool = False,
    trainer_labels=None,
    x_scale: str = "log",
    concise: bool = False,
):
    if not trainer_markers:
        trainer_markers = TRAINER_MARKERS

    if not trainer_colors:
        trainer_colors = TRAINER_COLORS

    trainer_markers, trainer_colors = update_trainer_markers_and_colors(
        results, trainer_markers, trainer_colors
    )

    if concise:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

    handles, labels = [], []

    if custom_metric == "ce_loss_score":
        ax.axhline(y=1, color="red", linestyle=":", alpha=0.7, linewidth=1)

    for trainer, marker in trainer_markers.items():
        trainer_data = {k: v for k, v in results.items() if v["sae_class"] == trainer}

        if not trainer_data:
            continue

        l0_values = [data[x_axis_key] for data in trainer_data.values()]
        custom_metric_values = [data[custom_metric] for data in trainer_data.values()]

        avg_value = sum(custom_metric_values) / len(custom_metric_values)

        if connect_points and len(l0_values) > 1:
            points = sorted(zip(l0_values, custom_metric_values))
            l0_values = [p[0] for p in points]
            custom_metric_values = [p[1] for p in points]

            ax.plot(
                l0_values,
                custom_metric_values,
                color=trainer_colors[trainer],
                linestyle="-",
                alpha=0.5,
                linewidth=2.5,
                zorder=1,
            )

        ax.scatter(
            l0_values,
            custom_metric_values,
            marker=marker,
            s=100,
            label=trainer,
            color=trainer_colors[trainer],
            edgecolor="black",
            zorder=2,
        )

        legend_handle = plt.scatter(
            [],
            [],
            marker=marker,
            s=100,
            color=trainer_colors[trainer],
            edgecolor="black",
        )
        handles.append(legend_handle)

        if trainer_labels and trainer in trainer_labels:
            base_label = trainer_labels[trainer]
        else:
            base_label = trainer.capitalize()

        if avg_value < 0.01 or avg_value >= 100:
            avg_str = f"{avg_value:.2e}"
        else:
            avg_str = f"{avg_value:.3f}".rstrip('0').rstrip('.')

        if concise:
            labels.append(base_label)
        else:
            labels.append(f"{base_label} (avg: {avg_str})")


    if x_larger_better:
        ax.set_xlabel(r"$\text{" + x_label + r"}\rightarrow\ $", fontsize=22 if concise else 15)
    else:
        ax.set_xlabel(r"$\leftarrow\ \text{" + x_label + r"}$", fontsize=22 if concise else 15)

    if larger_better:
        ax.set_ylabel(r"$\text{" + y_label + r"}\rightarrow\ $", fontsize=22 if concise else 15)
    else:
        ax.set_ylabel(r"$\leftarrow\ \text{" + y_label + r"}$", fontsize=22 if concise else 15)

    if not concise:
        ax.set_title(title)

    ax.set_xscale(x_scale)

    if concise:
        import matplotlib.ticker as ticker
        ax.set_xticks([60, 80, 100, 120])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major', labelsize=18)

    if baseline_value is not None:
        ax.axhline(baseline_value, color="red", linestyle="--", label=baseline_label)
        labels.append(baseline_label)
        handles.append(
            Line2D([0], [0], color="red", linestyle="--", label=baseline_label)
        )

    if handles and labels:
        if concise:
            order = ["TopK", "Batch TopK", "Ort", "Matryoshka", r"$C^2R$"]

            def get_order_index(label):
                for i, target in enumerate(order):
                    if target in label:
                        return i
                return len(order)

            sorted_indices = sorted(range(len(labels)), key=lambda i: get_order_index(labels[i]))
        else:
            base_labels_for_sorting = []
            for label in labels:
                if "(avg:" in label:
                    base_label = label.split("(avg:")[0].strip()
                else:
                    base_label = label
                base_labels_for_sorting.append(base_label)

            sorted_indices = sorted(range(len(base_labels_for_sorting)), key=lambda i: base_labels_for_sorting[i])

        labels = [labels[i] for i in sorted_indices]
        handles = [handles[i] for i in sorted_indices]

    ax.legend(handles, labels, loc="best", fontsize=20 if concise else 12)

    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)

    plt.tight_layout()

    if output_filename:
        base_filename = output_filename.rsplit('.', 1)[0] if '.' in output_filename else output_filename

        png_filename = f"{base_filename}.png"
        plt.savefig(png_filename, bbox_inches="tight", dpi=300)
        print(f"Saved as PNG: {png_filename}")

        pdf_filename = f"{base_filename}.pdf"
        plt.savefig(pdf_filename, bbox_inches="tight")
        print(f"Saved as PDF: {pdf_filename}")

    if return_fig:
        return fig

    plt.show()


def plot_2var_graph_dict_size(
    results: dict[str, dict[str, float]],
    custom_metric: str,
    title: str = "L0 vs Custom Metric",
    y_label: str = "Custom Metric",
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    output_filename: str | None = None,
    legend_location: str = "lower right",
    baseline_value: float | None = None,
    baseline_label: str | None = None,
    x_axis_key: str = "l0",
    return_fig: bool = False,
    trainer_markers: dict[str, str] | None = None,
):
    if not trainer_markers:
        trainer_markers = TRAINER_MARKERS

    l0_values = [data[x_axis_key] for data in results.values()]
    custom_metric_values = [data[custom_metric] for data in results.values()]

    fig, ax = plt.subplots(figsize=(10, 6))

    possible_sizes = ["4k", "16k", "65k", "131k", "1M"]

    colors = [plt.cm.Reds(x) for x in [0.1, 0.5, 0.9]]

    unique_sizes = [
        size
        for size in possible_sizes
        if size in set(v["d_sae"] for v in results.values())
    ]

    assert len(unique_sizes) <= len(colors), (
        "Too many unique dictionary sizes for color map"
    )

    size_to_color = {size: colors[i] for i, size in enumerate(unique_sizes)}

    handles, labels = [], []

    for dict_size in unique_sizes:
        size_data = {k: v for k, v in results.items() if v["d_sae"] == dict_size}

        l0_values = [data[x_axis_key] for data in size_data.values()]
        custom_metric_values = [data[custom_metric] for data in size_data.values()]
        sae_classes = [data["sae_class"] for data in size_data.values()]

        for l0, metric, sae_class in zip(l0_values, custom_metric_values, sae_classes):
            marker = trainer_markers[sae_class]
            ax.scatter(
                l0,
                metric,
                marker=marker,
                s=100,
                color=size_to_color[dict_size],
                edgecolor="black",
            )

        _handle = plt.scatter(
            [], [], marker="o", s=100, color=size_to_color[dict_size], edgecolor="black"
        )
        handles.append(_handle)
        labels.append(f"SAE Width: {dict_size}")

    ax.set_xlabel("L0 (Sparsity)")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if baseline_value:
        ax.axhline(baseline_value, color="red", linestyle="--", label=baseline_label)
        labels.append(baseline_label)
        handles.append(
            Line2D([0], [0], color="red", linestyle="--", label=baseline_label)
        )

    ax.legend(handles, labels, loc=legend_location)

    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)

    ax.set_xscale("log")

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    if return_fig:
        return fig
    plt.show()


def plot_steps_vs_average_diff(
    results_dict: dict,
    steps_key: str = "train_tokens",
    avg_diff_key: str = "average_diff",
    title: str | None = None,
    y_label: str | None = None,
    output_filename: str | None = None,
):
    trainer_data = defaultdict(lambda: {"train_tokens": [], "metric_scores": []})

    all_steps = set()

    for key, value in results_dict.items():
        trainer = key.split("/")[-1].split("_")[
            1
        ]
        layer = key.split("/")[-2].split("_")[-2]

        if "topk_ctx128" in key:
            trainer_type = "TopK SAE"
        elif "standard_ctx128" in key:
            trainer_type = "Standard SAE"
        else:
            raise ValueError(f"Unknown trainer type in key: {key}")

        step = int(value[steps_key])
        avg_diff = value[avg_diff_key]

        trainer_key = f"{trainer_type} Layer {layer} Trainer {trainer}"

        trainer_data[trainer_key]["train_tokens"].append(step)
        trainer_data[trainer_key]["metric_scores"].append(avg_diff)
        all_steps.add(step)

    average_trainer_data = {"train_tokens": [], "metric_scores": []}
    for step in sorted(all_steps):
        step_diffs = []
        for data in trainer_data.values():
            if step in data["train_tokens"]:
                idx = data["train_tokens"].index(step)
                step_diffs.append(data["metric_scores"][idx])
        if step_diffs:
            average_trainer_data["train_tokens"].append(step)
            average_trainer_data["metric_scores"].append(np.mean(step_diffs))

    trainer_data["Average"] = average_trainer_data

    plt.figure(figsize=(12, 6))

    for trainer_key, data in trainer_data.items():
        steps = data["train_tokens"]
        metric_scores = data["metric_scores"]

        sorted_data = sorted(zip(steps, metric_scores))
        steps, metric_scores = zip(*sorted_data)

        max_step = max(steps)

        step_percentages = [step / max_step * 100 for step in steps]

        if trainer_key == "Average":
            plt.plot(
                step_percentages,
                metric_scores,
                marker="o",
                label=trainer_key,
                linewidth=3,
                color="red",
                zorder=10,
            )
        else:
            plt.plot(
                step_percentages,
                metric_scores,
                marker="o",
                label=trainer_key,
                alpha=0.3,
                linewidth=1,
            )



    if not y_label:
        y_label = avg_diff_key.replace("_", " ").capitalize()

    plt.xlabel("Training Progess (%)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if len(trainer_data) < 50 and False:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    plt.show()


def plot_correlation_heatmap(
    plotting_results: dict[str, dict[str, float]],
    metric_names: list[str],
    ae_names: list[str] | None = None,
    title: str = "Metric Correlation Heatmap",
    output_filename: str = None,
    figsize: tuple = (12, 10),
    cmap: str = "coolwarm",
    annot: bool = True,
):
    if ae_names is None:
        ae_names = list(plotting_results.keys())


    data = []
    for ae in ae_names:
        row = [plotting_results[ae].get(metric, np.nan) for metric in metric_names]
        data.append(row)

    df = pd.DataFrame(data, index=ae_names, columns=metric_names)

    corr_matrix = df.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, vmin=-1, vmax=1, center=0)

    plt.title(title)
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    plt.show()


def plot_correlation_scatter(
    plotting_results: dict[str, dict[str, float]],
    metric_x: str,
    metric_y: str,
    x_label: str | None = None,
    y_label: str | None = None,
    ae_names: list[str] | None = None,
    title: str = "Metric Comparison Scatter Plot",
    output_filename: str | None = None,
    figsize: tuple = (10, 8),
):
    if ae_names is None:
        ae_names = list(plotting_results.keys())

    x_values = [plotting_results[ae].get(metric_x, float("nan")) for ae in ae_names]
    y_values = [plotting_results[ae].get(metric_y, float("nan")) for ae in ae_names]

    valid_data = [
        (x, y, ae)
        for x, y, ae in zip(x_values, y_values, ae_names)
        if not (np.isnan(x) or np.isnan(y))
    ]
    if not valid_data:
        print("No valid data points after removing NaN values.")
        return

    x_values, y_values, valid_ae_names = zip(*valid_data)

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    r, p_value = stats.pearsonr(x_values, y_values)
    r_squared = r**2

    plt.figure(figsize=figsize)
    sns.scatterplot(x=x_values, y=y_values, label="SAE", color="blue")

    if x_label is None:
        x_label = metric_x
    if y_label is None:
        y_label = metric_y

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    sns.regplot(
        x=x_values, y=y_values, scatter=False, color="red", label=f"r = {r:.4f}"
    )

    plt.legend()

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    plt.show()

    print(f"Pearson correlation coefficient (r): {r:.4f}")
    print(f"Coefficient of determination (r²): {r_squared:.4f}")
    print(f"P-value: {p_value:.4f}")


def plot_training_steps(
    results_dict: dict,
    metric_key: str,
    steps_key: str = "train_tokens",
    title: str | None = None,
    y_label: str | None = None,
    output_filename: str | None = None,
    break_fraction: float = 0.15,
):
    trainer_data = defaultdict(lambda: {"train_tokens": [], "metric_scores": []})
    all_steps = set()
    all_trainers = set()

    for key, value in results_dict.items():
        trainer = key.split("_trainer_")[-1].split("_")[0]
        trainer_class = value["sae_class"]
        step = int(value[steps_key])
        metric_scores = value[metric_key]
        trainer_key = f"{trainer_class} Trainer {trainer}"

        trainer_data[trainer_key]["train_tokens"].append(step)
        trainer_data[trainer_key]["metric_scores"].append(metric_scores)
        trainer_data[trainer_key]["sae_class"] = trainer_class
        all_steps.add(step)
        all_trainers.add(trainer_class)

    average_trainer_data = {"train_tokens": [], "metric_scores": []}
    for step in sorted(all_steps):
        step_diffs = [
            data["metric_scores"][data["train_tokens"].index(step)]
            for data in trainer_data.values()
            if step in data["train_tokens"]
        ]
        if step_diffs:
            average_trainer_data["train_tokens"].append(step)
            average_trainer_data["metric_scores"].append(np.mean(step_diffs))
    trainer_data["Average"] = average_trainer_data

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(15, 6),
        gridspec_kw={"width_ratios": [break_fraction, 1 - break_fraction]},
    )
    fig.subplots_adjust(wspace=0.01)

    steps_break_point = min([s for s in all_steps if s > 0]) / 2
    break_point = steps_break_point

    for trainer_key, data in trainer_data.items():
        steps = data["train_tokens"]
        metric_scores = data["metric_scores"]

        if trainer_key == "Average":
            color, trainer_class = "black", "Average"
        elif (
            data["sae_class"] == "standard"
            or data["sae_class"] == "Vanilla"
            or data["sae_class"] == "standard_april_update"
        ):
            color, trainer_class = "red", data["sae_class"]
        elif data["sae_class"] == "topk":
            color, trainer_class = "blue", data["sae_class"]
        else:
            raise ValueError(f"Trainer type not recognized for {trainer_key}")

        sorted_data = sorted(zip(steps, metric_scores))
        steps, metric_scores = zip(*sorted_data)

        ax1.plot(
            steps,
            metric_scores,
            marker="o",
            label=trainer_class,
            linewidth=4 if trainer_key == "Average" else 2,
            color=color,
            alpha=1 if trainer_key == "Average" else 0.3,
            zorder=10 if trainer_key == "Average" else 1,
        )
        ax2.plot(
            steps,
            metric_scores,
            marker="o",
            label=trainer_class,
            linewidth=4 if trainer_key == "Average" else 2,
            color=color,
            alpha=1 if trainer_key == "Average" else 0.3,
            zorder=10 if trainer_key == "Average" else 1,
        )

    ax1.set_xlim(-break_point / 4, break_point)
    ax2.set_xscale("log")

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, lw=4)

    ax1.plot((1, 1), (-d, +d), **kwargs)
    ax1.plot((1, 1), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((0, 0), (-d, +d), **kwargs)
    ax2.plot((0, 0), (1 - d, 1 + d), **kwargs)

    if not y_label:
        y_label = metric_key.replace("_", " ").capitalize()
    ax1.set_ylabel(y_label)
    fig.text(0.5, 0.01, "Training Steps", ha="center", va="center")
    if title is not None:
        fig.suptitle(title)


    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    legend_elements = []
    legend_elements.append(Line2D([0], [0], color="black", lw=3, label="Average"))
    if "standard" in all_trainers or "Vanilla" in all_trainers:
        legend_elements.append(Line2D([0], [0], color="red", lw=3, label="Standard"))
    if "topk" in all_trainers:
        legend_elements.append(Line2D([0], [0], color="blue", lw=3, label="TopK"))
    ax2.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    plt.show()


def get_sae_class_archived(sae_cfg: dict, sae_release) -> str:
    """For results pre Jan 2025"""
    if "sae_bench" in sae_release and "panneal" in sae_release:
        return "p_anneal"

    if sae_cfg["activation_fn_str"] == "topk":
        return "topk"

    return sae_cfg["architecture"]


def get_sae_bench_train_tokens_archived(sae_release: str, sae_id: str) -> int:
    """For results pre Jan 2025.
    This is for SAE Bench internal use. The SAE cfg does not contain the number of training tokens, so we need to hardcode it."""

    if "sae_bench" not in sae_release:
        raise ValueError("This function is only for SAE Bench releases")

    if "pythia" in sae_release:
        batch_size = 4096
    else:
        batch_size = 2048

    if "step" not in sae_id:
        if "pythia" in sae_release:
            steps = 48828
        elif "2pow14" in sae_release:
            steps = 146484
        elif "2pow12" or "2pow16" in sae_release:
            steps = 97656
        else:
            raise ValueError(f"sae release {sae_release} not recognized")

        return steps * batch_size
    else:
        match = re.search(r"step_(\d+)", sae_id)
        if match:
            step = int(match.group(1))
            return step * batch_size
        else:
            raise ValueError("No step match found")



def get_metrics(
    results: dict[str, dict[str, float]],
    custom_metric: str,
    title: str = "L0 vs Custom Metric",
    y_label: str = "Custom Metric",
    larger_better: bool = True,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    output_filename: str | None = None,
    baseline_value: float | None = None,
    baseline_label: str | None = None,
    x_axis_key: str = "l0",
    x_label: str = "L0 (Sparsity)",
    x_larger_better: bool = False,
    trainer_markers: dict[str, str] | None = None,
    trainer_colors: dict[str, str] | None = None,
    return_fig: bool = False,
    connect_points: bool = False,
    trainer_labels=None,
):


    trainer_markers, trainer_colors = update_trainer_markers_and_colors(
        results, trainer_markers, trainer_colors
    )

    handles, labels = [], []

    custom_metric_values_list = []
    l0_values_list = []

    for trainer, marker in trainer_markers.items():
        trainer_data = {k: v for k, v in results.items() if v["sae_class"] == trainer}

        if not trainer_data:
            continue

        l0_values = [data[x_axis_key] for data in trainer_data.values()]
        custom_metric_values = [data[custom_metric] for data in trainer_data.values()]

        if connect_points and len(l0_values) > 1:
            points = sorted(zip(l0_values, custom_metric_values))
            l0_values = [p[0] for p in points]
            custom_metric_values = [p[1] for p in points]

        if trainer_labels and trainer in trainer_labels:
            base_label = trainer_labels[trainer]
        else:
            base_label = trainer.capitalize()

        labels.append(base_label)
        custom_metric_values_list.append(custom_metric_values)
        l0_values_list.append(l0_values)

    return labels, custom_metric_values_list, l0_values_list

