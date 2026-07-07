"""
Implements the standard SAE training scheme.
"""
import torch as t
from typing import Optional

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import AutoEncoder
from collections import namedtuple

def compute_c2r_loss(feature_acts, decoder_weight, alpha=1.0):
    """
    Calculates the Cross-sample Consistency Regularization (C2R) loss.

    For each latent, finds its most directionally-similar neighbor in the
    dictionary and penalizes their co-activation, weighted by the summed
    batch activation norms of the pair.

    Args:
        feature_acts: [Batch, d_dict] - Sparse activations
        decoder_weight: [d_dict, d_model] - Decoder weights
        alpha: exponent for the weight term (suggested 1.0)
    """
    W_norm = t.nn.functional.normalize(decoder_weight, p=2, dim=1)
    d_dict = W_norm.shape[0]
    device = decoder_weight.device

    feature_norms = feature_acts.norm(p=2, dim=0)

    total_loss = 0.0
    chunk_size = 8192

    shuffled_indices = t.randperm(d_dict, device=device)

    for i in range(0, d_dict, chunk_size):
        end = min(i + chunk_size, d_dict)

        chunk_indices = shuffled_indices[i:end]

        W_chunk = W_norm[chunk_indices]
        sim_chunk = t.mm(W_chunk, W_chunk.T)

        sim_chunk.fill_diagonal_(-2.0)

        n_chunk = feature_norms[chunk_indices]

        max_vals, max_inds = sim_chunk.max(dim=1)
        max_vals = max_vals.clamp(min=0).pow(2)

        n_neighbors = n_chunk[max_inds]

        if alpha == 1.0:
            total_loss += ((n_chunk + n_neighbors) * max_vals).sum()
        else:
            N_sum_pow = (n_chunk + n_neighbors).pow(alpha)
            total_loss += (max_vals * N_sum_pow).sum()
    total_loss = total_loss / d_dict

    return total_loss

class StandardTrainer(SAETrainer):
    """
    Standard SAE training scheme following Towards Monosemanticity. Decoder column norms are constrained to 1.
    """
    def __init__(self,
                 steps: int,
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=AutoEncoder,
                 lr:float=1e-3,
                 l1_penalty:float=1e-1,
                 c2r_penalty:float=0.0,
                 c2r_alpha:float=1.0,
                 aux_loss_start_step:int=0,
                 aux_loss_interval:int=1,
                 warmup_steps:int=1000,
                 sparsity_warmup_steps:Optional[int]=2000,
                 decay_start:Optional[int]=None,
                 resample_steps:Optional[int]=None,
                 seed:Optional[int]=None,
                 device=None,
                 wandb_name:Optional[str]='StandardTrainer',
                 submodule_name:Optional[str]=None,
                 buffer_tokens: int = 256_000,
                 batch_tokens: int = 2048,
                 dtype: t.dtype = t.float32,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.l1_penalty = l1_penalty
        self.c2r_penalty = c2r_penalty
        self.c2r_alpha = c2r_alpha
        self.aux_loss_start_step = aux_loss_start_step
        self.aux_loss_interval = aux_loss_interval
        self.buffer_tokens = buffer_tokens
        self.batch_tokens = batch_tokens

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae = dict_class(activation_dim, dict_size, dtype=dtype, device=self.device)
        self.ae.to(self.device)

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            self.steps_since_active = t.zeros(self.ae.dict_size, dtype=int).to(self.device)
            self.last_resampled_step = t.zeros(self.ae.dict_size, dtype=int).to(self.device) - 10000
        else:
            self.steps_since_active = None
            self.last_resampled_step = None

        self.optimizer = ConstrainedAdam(self.ae.parameters(), self.ae.decoder.parameters(), lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

    def resample_neurons(self, deads, activations, step=None):
        with t.no_grad():
            if deads.sum() == 0: return
            print(f"resampling {deads.sum().item()} neurons")

            losses = (activations - self.ae(activations)).norm(dim=-1)

            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()

            deads[deads.nonzero()[n_resample:]] = False

            if self.last_resampled_step is not None and step is not None:
                self.last_resampled_step[deads] = step

            self.ae.encoder.weight[deads] = sampled_vecs * alive_norm * 0.2
            self.ae.decoder.weight[:,deads] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.encoder.bias[deads] = 0.


            state_dict = self.optimizer.state_dict()['state']
            state_dict[1]['exp_avg'][deads] = 0.
            state_dict[1]['exp_avg_sq'][deads] = 0.
            state_dict[2]['exp_avg'][deads] = 0.
            state_dict[2]['exp_avg_sq'][deads] = 0.
            state_dict[3]['exp_avg'][:,deads] = 0.
            state_dict[3]['exp_avg_sq'][:,deads] = 0.

    def loss(self, x, step: int, logging=False, **kwargs):

        sparsity_scale = self.sparsity_warmup_fn(step)

        x_hat, f = self.ae(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()

        if self.steps_since_active is not None:
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0

        loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        apply_aux_loss = False
        aux_loss_multiplier = 1.0

        if step >= self.aux_loss_start_step:
            if (step - self.aux_loss_start_step) % self.aux_loss_interval == 0:
                apply_aux_loss = True
                aux_loss_multiplier = float(self.aux_loss_interval)

        if self.c2r_penalty > 0 and apply_aux_loss:
            c2r_loss = compute_c2r_loss(f, self.ae.decoder.weight.T, alpha=self.c2r_alpha)
            loss += self.c2r_penalty * c2r_loss * aux_loss_multiplier
        else:
            c2r_loss = t.tensor(0.0)


        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss' : l2_loss.item(),
                    'mse_loss' : recon_loss.item(),
                    'sparsity_loss' : l1_loss.item(),
                    'c2r_loss' : c2r_loss.item(),
                    'loss' : loss.item()
                }
            )


    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations, step=step)

    @property
    def config(self):
        return {
            'dict_class': 'AutoEncoder',
            'trainer_class' : 'StandardTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'c2r_penalty' : self.c2r_penalty,
            'c2r_alpha' : self.c2r_alpha,
            'aux_loss_start_step' : self.aux_loss_start_step,
            'aux_loss_interval' : self.aux_loss_interval,
            'warmup_steps' : self.warmup_steps,
            'resample_steps' : self.resample_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'steps' : self.steps,
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


class StandardTrainerAprilUpdate(SAETrainer):
    """
    Standard SAE training scheme following the Anthropic April update. Decoder column norms are NOT constrained to 1.
    This trainer does not support resampling or ghost gradients. This trainer will have fewer dead neurons than the standard trainer.
    """
    def __init__(self,
                 steps: int,
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=AutoEncoder,
                 lr:float=1e-3,
                 l1_penalty:float=1e-1,
                 c2r_penalty:float=0.0,
                 c2r_alpha:float=1.0,
                 aux_loss_start_step:int=0,
                 aux_loss_interval:int=1,
                 warmup_steps:int=1000,
                 sparsity_warmup_steps:Optional[int]=2000,
                 decay_start:Optional[int]=None,
                 seed:Optional[int]=None,
                 device=None,
                 wandb_name:Optional[str]='StandardTrainerAprilUpdate',
                 submodule_name:Optional[str]=None,
                 buffer_tokens: int = 256_000,
                 batch_tokens: int = 2048,
                 dtype: t.dtype = t.float32,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.lr = lr
        self.l1_penalty = l1_penalty
        self.c2r_penalty = c2r_penalty
        self.c2r_alpha = c2r_alpha
        self.aux_loss_start_step = aux_loss_start_step
        self.aux_loss_interval = aux_loss_interval
        self.warmup_steps = sparsity_warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.buffer_tokens = buffer_tokens
        self.batch_tokens = batch_tokens

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae = dict_class(activation_dim, dict_size, dtype=dtype, device=self.device)
        self.ae.to(self.device)

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None, sparsity_warmup_steps=sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

    def loss(self, x, step: int, logging=False, **kwargs):

        sparsity_scale = self.sparsity_warmup_fn(step)

        x_hat, f = self.ae(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = (f * self.ae.decoder.weight.norm(p=2, dim=0)).sum(dim=-1).mean()

        loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        apply_aux_loss = False
        aux_loss_multiplier = 1.0

        if step >= self.aux_loss_start_step:
            if (step - self.aux_loss_start_step) % self.aux_loss_interval == 0:
                apply_aux_loss = True
                aux_loss_multiplier = float(self.aux_loss_interval)

        if self.c2r_penalty > 0 and apply_aux_loss:
            c2r_loss = compute_c2r_loss(f, self.ae.decoder.weight.T, alpha=self.c2r_alpha)
            loss += self.c2r_penalty * c2r_loss * aux_loss_multiplier
        else:
            c2r_loss = t.tensor(0.0)

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss' : l2_loss.item(),
                    'mse_loss' : recon_loss.item(),
                    'sparsity_loss' : l1_loss.item(),
                    'c2r_loss' : c2r_loss.item(),
                    'loss' : loss.item()
                }
            )


    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

    @property
    def config(self):

        return {
            'dict_class': 'AutoEncoder',
            'trainer_class' : 'StandardTrainerAprilUpdate',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'c2r_penalty' : self.c2r_penalty,
            'c2r_alpha' : self.c2r_alpha,
            'aux_loss_start_step' : self.aux_loss_start_step,
            'aux_loss_interval' : self.aux_loss_interval,
            'warmup_steps' : self.warmup_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'steps' : self.steps,
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

