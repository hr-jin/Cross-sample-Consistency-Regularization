"""
This file will be deleted once the encode method is merged into SAELens (currently it's not possible to encode with only
a slice of the SAE latents).
"""

import torch
from jaxtyping import Float
from sae_lens import SAE


def encode_subset(
    self: SAE, x: torch.Tensor, latents: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Calculate SAE latents from inputs. Includes optional `latents` argument to only calculate a subset.
    """
    encode_fn = {
        "standard": encode_standard,
        "gated": encode_gated,
        "jumprelu": encode_jumprelu,
    }[self.cfg.architecture]

    return (
        encode_fn(self, x, latents=None)[..., latents]
        if self.cfg.activation_fn_str == "topk"
        else encode_fn(self, x, latents)
    )


def encode_gated(
    self: SAE,
    x: Float[torch.Tensor, "... d_in"],
    latents: torch.Tensor | None = None,
) -> Float[torch.Tensor, "... d_sae"]:
    """
    Computes the latent values of the Sparse Autoencoder (SAE) using a gated architecture. The activation values are
    computed as the product of the masking term & the post-activation function magnitude term:

        1[(x - b_dec) @ W_gate + b_gate > 0] * activation_fn((x - b_dec) @ W_enc + b_enc)

    The `latents` argument allows for the computation of a specific subset of the hidden values. If `latents` is not
    provided, all latent values will be computed.
    """
    latents_tensor = torch.arange(self.cfg.d_sae) if latents is None else latents

    x = x.to(self.dtype)
    x = self.reshape_fn_in(x)
    x = self.hook_sae_input(x)
    x = self.run_time_activation_norm_fn_in(x)
    sae_in = x - self.b_dec * self.cfg.apply_b_dec_to_input

    gating_pre_activation = (
        sae_in @ self.W_enc[:, latents_tensor] + self.b_gate[latents_tensor]
    )
    active_features = (gating_pre_activation > 0).to(self.dtype)

    magnitude_pre_activation = self.hook_sae_acts_pre(
        sae_in @ (self.W_enc[:, latents_tensor] * self.r_mag[latents_tensor].exp())
        + self.b_mag[latents_tensor]
    )
    feature_magnitudes = self.activation_fn(magnitude_pre_activation)

    feature_acts = self.hook_sae_acts_post(active_features * feature_magnitudes)

    return feature_acts


def encode_jumprelu(
    self: SAE,
    x: Float[torch.Tensor, "... d_in"],
    latents: torch.Tensor | None = None,
) -> Float[torch.Tensor, "... d_sae"]:
    """
    Computes the latent values of the Sparse Autoencoder (SAE) using a gated architecture. The activation values are
    computed as:

        activation_fn((x - b_dec) @ W_enc + b_enc) * 1[(x - b_dec) @ W_enc + b_enc > threshold]

    The `latents` argument allows for the computation of a specific subset of the hidden values. If `latents` is not
    provided, all latent values will be computed.
    """
    latents_tensor = torch.arange(self.cfg.d_sae) if latents is None else latents

    x = x.to(self.dtype)

    x = self.reshape_fn_in(x)

    x = self.run_time_activation_norm_fn_in(x)

    sae_in = self.hook_sae_input(x - (self.b_dec * self.cfg.apply_b_dec_to_input))

    hidden_pre = self.hook_sae_acts_pre(
        sae_in @ self.W_enc[:, latents_tensor] + self.b_enc[latents_tensor]
    )

    feature_acts = self.hook_sae_acts_post(
        self.activation_fn(hidden_pre) * (hidden_pre > self.threshold[latents_tensor])
    )

    return feature_acts


def encode_standard(
    self: SAE,
    x: Float[torch.Tensor, "... d_in"],
    latents: torch.Tensor | None = None,
) -> Float[torch.Tensor, "... d_sae"]:
    """
    Computes the latent values of the Sparse Autoencoder (SAE) using a gated architecture. The activation values are
    computed as:

        activation_fn((x - b_dec) @ W_enc + b_enc)

    The `latents` argument allows for the computation of a specific subset of the hidden values. If `latents` is not
    provided, all latent values will be computed.
    """
    latents_tensor = torch.arange(self.cfg.d_sae) if latents is None else latents

    x = x.to(self.dtype)
    x = self.reshape_fn_in(x)
    x = self.hook_sae_input(x)
    x = self.run_time_activation_norm_fn_in(x)

    sae_in = x - (self.b_dec * self.cfg.apply_b_dec_to_input)

    hidden_pre = self.hook_sae_acts_pre(
        sae_in @ self.W_enc[:, latents_tensor] + self.b_enc[latents_tensor]
    )
    feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    return feature_acts
