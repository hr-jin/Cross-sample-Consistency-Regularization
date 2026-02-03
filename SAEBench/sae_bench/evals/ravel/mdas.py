import sae_lens
import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

import sae_bench.sae_bench_utils.activation_collection as activation_collection
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig


class MDAS(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: RAVELEvalConfig,
        sae: sae_lens.SAE,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.layer_intervened = sae.cfg.hook_layer

        hidden_dim = model.config.hidden_size


        self.transform_matrix = torch.nn.Parameter(
            torch.eye(hidden_dim, device=model.device, dtype=torch.float32),
            requires_grad=True,
        )
        self.binary_mask = torch.nn.Parameter(
            torch.zeros(hidden_dim, device=model.device, dtype=torch.float32),
            requires_grad=True,
        )

        self.batch_size = config.llm_batch_size
        self.device = model.device
        self.temperature = 1e-2

    def create_intervention_hook(
        self,
        source_rep_BD: torch.Tensor,
        base_pos_B: torch.Tensor,
        training_mode: bool = False,
    ):
        def intervention_hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                resid_BLD = outputs[0]
                rest = outputs[1:]
            else:
                raise ValueError("Unexpected output shape")

            if resid_BLD.shape[1] == 1:
                return outputs

            resid_BD = resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :]

            rotated_source_BD = torch.matmul(
                source_rep_BD.to(dtype=torch.float32), self.transform_matrix
            )
            rotated_resid_BD = torch.matmul(
                resid_BD.to(dtype=torch.float32), self.transform_matrix
            )

            if not training_mode:
                mask_values_D = (self.binary_mask > 0).to(dtype=self.binary_mask.dtype)
            else:
                mask_values_D = torch.sigmoid(self.binary_mask / self.temperature)


            modified_resid_BD = (
                1 - mask_values_D
            ) * rotated_resid_BD + mask_values_D * rotated_source_BD

            modified_resid_BD = torch.matmul(modified_resid_BD, self.transform_matrix.T)

            resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :] = (
                modified_resid_BD.to(dtype=resid_BLD.dtype)
            )

            return (resid_BLD, *rest)

        return intervention_hook

    def forward(
        self,
        base_encoding_BL,
        source_encoding_BL,
        base_pos_B,
        source_pos_B,
        training_mode: bool = False,
    ):
        with torch.no_grad():
            source_rep = activation_collection.get_layer_activations(
                self.model, self.layer_intervened, source_encoding_BL, source_pos_B
            )

        intervention_hook = self.create_intervention_hook(
            source_rep,
            base_pos_B,
            training_mode,
        )

        handle = activation_collection.get_module(
            self.model, self.layer_intervened
        ).register_forward_hook(intervention_hook)

        logits_BV = self.model(
            input_ids=base_encoding_BL["input_ids"].to(self.model.device),
            attention_mask=base_encoding_BL.get("attention_mask", None),
        ).logits[:, -1, :]

        handle.remove()

        predicted_B = logits_BV.argmax(dim=-1)

        predicted_text = []
        for i in range(logits_BV.shape[0]):
            predicted_text.append(self.tokenizer.decode(predicted_B[i]))

        return logits_BV, predicted_text
