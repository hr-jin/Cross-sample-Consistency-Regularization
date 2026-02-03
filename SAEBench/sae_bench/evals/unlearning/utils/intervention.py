import torch
from jaxtyping import Float
from sae_lens import SAE
from torch import Tensor
from transformer_lens.hook_points import HookPoint


def anthropic_clamp_resid_SAE_features(
    resid: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    sae: SAE,
    features_to_ablate: list[int],
    multiplier: float = 1.0,
    random: bool = False,
) -> Float[Tensor, "batch seq d_model"] | None:
    """
    Given a list of feature indices, this hook function removes feature activations in a manner similar to the one
    used in "Scaling Monosemanticity": https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#appendix-methods-steering
    This version clamps the feature activation to the value(s) specified in multiplier
    """

    if len(features_to_ablate) > 0:
        with torch.no_grad():
            feature_activations = sae.encode(resid)

            feature_activations[:, 0, :] = (
                0.0
            )

            reconstruction = sae.decode(feature_activations)



            error = resid - reconstruction

            non_zero_features_BLD = feature_activations[:, :, features_to_ablate] > 0



            if not random:
                if isinstance(multiplier, float) or isinstance(multiplier, int):
                    feature_activations[:, :, features_to_ablate] = torch.where(
                        non_zero_features_BLD,
                        -multiplier,
                        feature_activations[:, :, features_to_ablate],
                    )
                else:
                    raise NotImplementedError("Currently deprecated")
                    feature_activations[:, :, features_to_ablate] = torch.where(
                        non_zero_features_BLD,
                        -multiplier.unsqueeze(dim=0).unsqueeze(dim=0),
                        feature_activations[:, :, features_to_ablate],
                    )

            else:
                raise NotImplementedError("Currently deprecated")
                assert isinstance(multiplier, float) or isinstance(multiplier, int)

                next_features_to_ablate = [
                    (f + 1) % feature_activations.shape[-1] for f in features_to_ablate
                ]
                feature_activations[:, :, next_features_to_ablate] = torch.where(
                    feature_activations[:, :, features_to_ablate] > 0,
                    -multiplier,
                    feature_activations[:, :, next_features_to_ablate],
                )

            modified_reconstruction = sae.decode(feature_activations)

            resid = error + modified_reconstruction
        return resid
    return None
