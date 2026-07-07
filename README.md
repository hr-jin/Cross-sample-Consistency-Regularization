# C²R: Cross-sample Consistency Regularization

Official code release for the ICML 2026 paper:

> **C²R: Cross-sample Consistency Regularization Mitigates Feature Splitting and Absorption in Sparse Autoencoders**
> Haoran Jin, Xiting Wang, Shijie Ren, Hong Xie, Defu Lian.

Paper: https://arxiv.org/abs/2606.30609

## Overview

Sparse Autoencoders (SAEs) decompose language-model activations into sparse,
human-interpretable latents. As dictionaries grow, two failure modes emerge:

- **Feature splitting** — a single coherent concept is fragmented across several
  near-duplicate latents.
- **Feature absorption** — a general latent silently "absorbs" specific cases,
  creating arbitrary exceptions.

C²R traces both to *inconsistent latent assignment across samples*: because SAE
training optimizes each sample independently, one concept can be scattered over
multiple redundant or interfering latents. C²R adds a batch-level regularizer
that penalizes the co-activation of directionally similar latents, encouraging
each semantic feature to be represented by a single, unified latent — without
degrading reconstruction fidelity.

## Repository layout

```
.
├── dictionary_learning_demo/   # SAE training (fork of saprmarks/dictionary_learning)
│   ├── demo.py                 # entry point for training sweeps
│   ├── demo_config.py          # architectures, sparsities, and C²R hyperparameters
│   └── dictionary_learning/dictionary_learning/trainers/
│       ├── standard.py         # <-- C²R loss: `compute_c2r_loss`
│       ├── top_k.py, batch_top_k.py, matryoshka_batch_top_k.py,
│       ├── jumprelu.py, gdm.py (gated), ort.py, ...
├── SAEBench/                   # evaluation suite (fork of adamkarvonen/SAEBench)
└── pipeline.sh                 # end-to-end train + eval example
```

## The C²R loss

The regularizer is implemented once in
[`compute_c2r_loss`](dictionary_learning_demo/dictionary_learning/dictionary_learning/trainers/standard.py)
and reused by every supported trainer (`standard`, `standard_new`, `top_k`,
`batch_top_k`, `matryoshka_batch_top_k`, `jump_relu`, `gated`, `ort`).

Given the batch of sparse activations `f` (shape `[batch, d_dict]`) and the
decoder matrix `W_dec` (shape `[d_dict, d_model]`), for each latent it:

1. L2-normalizes decoder columns and computes pairwise cosine similarities.
2. Finds each latent's most directionally-similar neighbor (`max` cosine sim,
   clamped to be non-negative; `topTauPerFeatSquare` squares it).
3. Weights that similarity by the summed batch activation norms of the latent
   and its neighbor, so the penalty only bites when *both* co-activate.

The per-latent penalties are averaged over the dictionary and added to the
training objective, scaled by `c2r_penalty` (`--lambda_c2r`). Similarity is
computed in shuffled chunks (`chunk_size = 8192`) to bound memory on large
dictionaries.

### Hyperparameters

| flag | trainer arg | meaning | default |
|------|-------------|---------|---------|
| `--lambda_c2r` | `c2r_penalty` | C²R loss weight (`0` disables) | `0.0` |
| `--c2r_alpha`  | `c2r_alpha`   | exponent on the activation-norm weight | `1.0` |
| `--aux_loss_start_step` | | step to begin applying the C²R loss | `0` |
| `--aux_loss_interval`   | | apply C²R every N steps (loss is scaled by N) | `1` |

## Installation

```bash
# Training
cd dictionary_learning_demo
git submodule update --init --recursive
pip install -e .

# Evaluation
cd ../SAEBench
pip install -e .
```

See the READMEs in each subdirectory for details
([`dictionary_learning_demo/README.md`](dictionary_learning_demo/README.md),
[`SAEBench/README.md`](SAEBench/README.md)).

## Training with C²R

Train a BatchTopK SAE on Gemma-2-2B with C²R enabled:

```bash
cd dictionary_learning_demo
python demo.py \
    --save_dir trained_saes \
    --model_name google/gemma-2-2b --layers 12 \
    --architectures batch_top_k \
    --sae_batch_size 2048 --num_tokens 500000000 --dtype float32 \
    --target_l0s '[60]' \
    --lambda_c2r '[5]' --c2r_alpha 1 \
    --aux_loss_start_step 0 --aux_loss_interval 5
```

Set `--lambda_c2r '[0]'` (or omit it) to train an identical baseline without
C²R for comparison.

## Reproducing the paper

`pipeline.sh` runs the full train-then-evaluate loop used in the paper (training
via `dictionary_learning_demo`, evaluation via `SAEBench` on the `absorption`,
`core`, `autointerp`, and `ravel` benchmarks). Edit the paths, dataset, device
IDs, and API keys at the top of the script before running:

```bash
bash pipeline.sh
```

## Acknowledgements

This code builds on
[`saprmarks/dictionary_learning`](https://github.com/saprmarks/dictionary_learning)
for SAE training and
[`adamkarvonen/SAEBench`](https://github.com/adamkarvonen/SAEBench) for
evaluation. We thank the authors of both projects.

## Citation

```bibtex
@inproceedings{jin2026c2r,
  title     = {C\textsuperscript{2}R: Cross-sample Consistency Regularization
               Mitigates Feature Splitting and Absorption in Sparse Autoencoders},
  author    = {Jin, Haoran and Wang, Xiting and Ren, Shijie and Xie, Hong and Lian, Defu},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
