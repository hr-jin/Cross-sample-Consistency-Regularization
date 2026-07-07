"""
Implements the training scheme for a gated SAE described in https://arxiv.org/abs/2404.16014
"""

import torch as t
from typing import Optional

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import GatedAutoEncoder
from ..trainers.standard import compute_c2r_loss
from collections import namedtuple

class GatedSAETrainer(SAETrainer):
    """
    Gated SAE training scheme.
    """
    def __init__(self,
                 steps: int,
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class = GatedAutoEncoder,
                 lr: float = 5e-5,
                 l1_penalty: float = 1e-1,
                 c2r_penalty: float = 0.0,
                 c2r_alpha: float = 1.0,
                 warmup_steps: int = 1000,
                 sparsity_warmup_steps: Optional[int] = 2000,
                 decay_start:Optional[int]=None,
                 seed: Optional[int] = None,
                 device: Optional[str] = None,
                 wandb_name: Optional[str] = 'GatedSAETrainer',
                 submodule_name: Optional[str] = None,
                 buffer_tokens: int = 256_000,
                 batch_tokens: int = 2048,
                 dtype: t.dtype = t.float32,
                 **kwargs
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae = dict_class(activation_dim, dict_size, dtype=dtype, device=self.device)
        self.ae.to(self.device)

        self.buffer_tokens = buffer_tokens
        self.batch_tokens = batch_tokens



        self.lr = lr
        self.l1_penalty=l1_penalty
        self.c2r_penalty = c2r_penalty
        self.c2r_alpha = c2r_alpha
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)

        self.optimizer = ConstrainedAdam(
            self.ae.parameters(),
            self.ae.decoder.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
        )

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None, sparsity_warmup_steps=sparsity_warmup_steps)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_fn)
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)


    def loss(self, x:t.Tensor, step:int, logging:bool=False, **kwargs):

        sparsity_scale = self.sparsity_warmup_fn(step)

        f, f_gate = self.ae.encode(x, return_gate=True)
        x_hat = self.ae.decode(f)
        x_hat_gate = f_gate @ self.ae.decoder.weight.detach().T + self.ae.decoder_bias.detach()

        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        L_sparse = t.linalg.norm(f_gate, ord=1, dim=-1).mean()
        L_aux = (x - x_hat_gate).pow(2).sum(dim=-1).mean()

        loss = L_recon + (self.l1_penalty * L_sparse * sparsity_scale) + L_aux


        if self.c2r_penalty > 0:
            c2r_loss = compute_c2r_loss(f_gate, self.ae.decoder.weight.T, self.c2r_alpha)
            loss += self.c2r_penalty * c2r_loss
        else:
            c2r_loss = t.tensor(0.0)

        

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'mse_loss' : L_recon.item(),
                    'sparsity_loss' : L_sparse.item(),
                    'aux_loss' : L_aux.item(),
                    'c2r_loss' : c2r_loss.item(),
                    'loss' : loss.item()
                }
            )

    def update(self, step, x):
        x = x.to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss(x, step)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    @property
    def config(self):
        return {
            'dict_class': 'GatedAutoEncoder',
            'trainer_class' : 'GatedSAETrainer',
            'activation_dim' : self.ae.activation_dim,
            'dict_size' : self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'c2r_penalty' : self.c2r_penalty,
            'c2r_alpha' : self.c2r_alpha,
            'warmup_steps' : self.warmup_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'decay_start' : self.decay_start,
            'seed' : self.seed,
            'device' : self.device,
            'layer' : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            "buffer_tokens": self.buffer_tokens,
            "batch_tokens": self.batch_tokens
        }
