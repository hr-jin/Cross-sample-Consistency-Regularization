"""
Implements the OrtSAE training scheme based on Batch TopK SAE.
"""
import torch as t
import torch.nn.functional as F
from typing import Optional
from collections import namedtuple

from ..trainers.batch_top_k import BatchTopKTrainer, BatchTopKSAE
from ..trainers.standard import compute_c2r_loss

def get_orthogonal_loss(decoder_weight, chunk_size=8192):
    """
    Computes the OrtSAE orthogonality penalty using chunk-wise strategy.

    Args:
        decoder_weight (torch.Tensor): The decoder weight matrix of shape [d_model, n_features].
        chunk_size (int): The size of each random chunk (default: 8192).

    Returns:
        torch.Tensor: The scalar orthogonality loss.
    """
    w_norm = F.normalize(decoder_weight, p=2, dim=0)

    n_features = w_norm.shape[1]

    shuffled_indices = t.randperm(n_features, device=w_norm.device)

    loss_accum = 0.0
    num_chunks = 0

    for i in range(0, n_features, chunk_size):
        chunk_indices = shuffled_indices[i : i + chunk_size]

        w_chunk = w_norm[:, chunk_indices]

        sim_matrix = t.matmul(w_chunk.T, w_chunk)

        sim_matrix.fill_diagonal_(-1.0)

        max_sim_values, _ = t.max(sim_matrix, dim=1)

        loss_chunk = t.mean(t.clamp(max_sim_values, min=0) ** 2)

        loss_accum += loss_chunk
        num_chunks += 1

    return loss_accum / num_chunks

class OrtTrainer(BatchTopKTrainer):
    """
    Ort training scheme based on Batch TopK SAE.
    """
    def __init__(self,
                 steps: int,
                 activation_dim: int,
                 dict_size: int,
                 k: int,
                 layer: int,
                 lm_name: str,
                 dict_class=BatchTopKSAE,
                 lr:Optional[float]=None,
                 orthogonality_penalty:float=0.25,
                 chunk_size:int=8192,
                 c2r_penalty:float=0.0,
                 c2r_alpha:float=1.0,
                 aux_loss_start_step: int = 0,
                 aux_loss_interval: int = 1,
                 auxk_alpha: float = 1 / 32,
                 warmup_steps:int=1000,
                 decay_start:Optional[int]=None,
                 threshold_beta: float = 0.999,
                 threshold_start_step: int = 1000,
                 k_anneal_steps: Optional[int] = None,
                 seed:Optional[int]=None,
                 device=None,
                 wandb_name:Optional[str]='OrtTrainer',
                 submodule_name:Optional[str]=None,
                 buffer_tokens: int = 256_000,
                 batch_tokens: int = 2048,
                 dtype: t.dtype = t.float32,
    ):
        super().__init__(
            steps=steps,
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            layer=layer,
            lm_name=lm_name,
            dict_class=dict_class,
            lr=lr,
            c2r_penalty=c2r_penalty,
            c2r_alpha=c2r_alpha,
            aux_loss_start_step=aux_loss_start_step,
            aux_loss_interval=aux_loss_interval,
            auxk_alpha=auxk_alpha,
            warmup_steps=warmup_steps,
            decay_start=decay_start,
            threshold_beta=threshold_beta,
            threshold_start_step=threshold_start_step,
            k_anneal_steps=k_anneal_steps,
            seed=seed,
            device=device,
            wandb_name=wandb_name,
            submodule_name=submodule_name,
            buffer_tokens=buffer_tokens,
            batch_tokens=batch_tokens,
            dtype=dtype
        )
        self.orthogonality_penalty = orthogonality_penalty
        self.chunk_size = chunk_size

    def loss(self, x, step=None, logging=False):
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x, return_active=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(f)

        x_hat = self.ae.decode(f)

        e = x - x_hat

        self.effective_l0 = self.ae.k.item()

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)

        loss = l2_loss + self.auxk_alpha * auxk_loss

        apply_aux_loss = False
        aux_loss_multiplier = 1.0

        if step >= self.aux_loss_start_step:
            if (step - self.aux_loss_start_step) % self.aux_loss_interval == 0:
                apply_aux_loss = True
                aux_loss_multiplier = float(self.aux_loss_interval)

        if self.orthogonality_penalty > 0 and apply_aux_loss:
            orth_loss = orth_loss = get_orthogonal_loss(self.ae.decoder.weight, chunk_size=self.chunk_size)
            loss += self.orthogonality_penalty * orth_loss * aux_loss_multiplier
        else:
            orth_loss = t.tensor(0.0)

        if self.c2r_penalty > 0 and apply_aux_loss:
            c2r_loss = compute_c2r_loss(f, self.ae.decoder.weight.T, self.c2r_alpha)
            loss += self.c2r_penalty * c2r_loss * aux_loss_multiplier
        else:
            c2r_loss = t.tensor(0.0)

        
        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "orth_loss": orth_loss.item(),
                    "c2r_loss": c2r_loss.item(),
                    "loss": loss.item(),
                    "threshold": self.ae.threshold,
                },
            )

    @property
    def config(self):
        return {
            "trainer_class": "OrtTrainer",
            "dict_class": "BatchTopKSAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "buffer_tokens": self.buffer_tokens,
            "batch_tokens": self.batch_tokens,
            "c2r_penalty": self.c2r_penalty,
            "c2r_alpha": self.c2r_alpha,
            "aux_loss_start_step": self.aux_loss_start_step,
            "aux_loss_interval": self.aux_loss_interval,
            "orthogonality_penalty": self.orthogonality_penalty,
            "chunk_size": self.chunk_size,
        }
