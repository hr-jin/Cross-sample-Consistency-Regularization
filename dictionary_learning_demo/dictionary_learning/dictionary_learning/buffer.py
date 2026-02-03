import torch as t
from nnsight import LanguageModel
import gc
from tqdm import tqdm

from .config import DEBUG

if DEBUG:
    tracer_kwargs = {'scan' : True, 'validate' : True}
else:
    tracer_kwargs = {'scan' : False, 'validate' : False}


class ActivationBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.

    max_activation_norm_multiple: remove all activations with norm greater than median norm * max_activation_norm_multiple. 10 is a good default.
    This is useful for models like Qwen which have random, unpredictable high norm activation sinks which reduce training effectiveness.
    """
    def __init__(self,
                 data,
                 model : LanguageModel,
                 submodule,
                 d_submodule=None,
                 io='out',
                 n_ctxs=3e4,
                 ctx_len=128,
                 refresh_batch_size=512,
                 out_batch_size=8192,
                 device='cpu',
                 remove_bos: bool = False,
                 add_special_tokens: bool = True,
                 max_activation_norm_multiple: int | None = None,
                ):

        if io not in ['in', 'out']:
            raise ValueError("io must be either 'in' or 'out'")

        if d_submodule is None:
            try:
                if io == 'in':
                    d_submodule = submodule.in_features
                else:
                    d_submodule = submodule.out_features
            except:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")
        self.activations = t.empty(0, d_submodule, device=device, dtype=model.dtype)
        self.read = t.zeros(0).bool()

        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.activation_buffer_size = n_ctxs * ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.remove_bos = remove_bos
        self.remove_high_norm = max_activation_norm_multiple

        if remove_bos and self.model.tokenizer.bos_token_id is None:
            print(
                "\n\n\nWARNING: remove_bos is True but tokenizer does not have a bos token. We are removing the first non-pad token instead. Don't use sequence packing.\n\n\n"
            )


    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]

    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True,
            add_special_tokens=self.add_special_tokens
        )

    def refresh(self):
        gc.collect()
        t.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = t.empty(self.activation_buffer_size, self.d_submodule, device=self.device, dtype=self.model.dtype)

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations


        while current_idx < self.activation_buffer_size:
            with t.no_grad():
                tokens = self.tokenized_batch()
                with self.model.trace(
                    tokens,
                    **tracer_kwargs,
                    invoker_args={"truncation": True, "max_length": self.ctx_len},
                ):
                    if self.io == "in":
                        hidden_states = self.submodule.inputs[0].save()
                    else:
                        hidden_states = self.submodule.output.save()
                    input = self.model.inputs.save()

                    self.submodule.output.stop()

            mask = (input.value[1]["attention_mask"] != 0)
            hidden_states = hidden_states.value
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            if self.remove_bos:
                if self.model.tokenizer.bos_token_id is not None:
                    bos_mask = input.value[1]["input_ids"] == self.model.tokenizer.bos_token_id
                    mask = mask & ~bos_mask
                else:
                    assert mask.dim() == 2, "expected shape (batch_size, seq_len)"
                    first_one = (mask.to(t.int64).cumsum(dim=1) == 1) & mask
                    mask = mask & ~first_one

            if self.remove_high_norm is not None:
                norms_BL = hidden_states.norm(dim=-1)
                median_norm = norms_BL.median()
                norm_mask = norms_BL > median_norm * self.remove_high_norm
                mask = mask & ~norm_mask

            hidden_states = hidden_states[mask]

            remaining_space = self.activation_buffer_size - current_idx
            assert remaining_space > 0
            hidden_states = hidden_states[:remaining_space]

            self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states.to(
                self.device
            )
            current_idx += len(hidden_states)


        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            'd_submodule' : self.d_submodule,
            'io' : self.io,
            'n_ctxs' : self.n_ctxs,
            'ctx_len' : self.ctx_len,
            'refresh_batch_size' : self.refresh_batch_size,
            'out_batch_size' : self.out_batch_size,
            'device' : self.device
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()


class HeadActivationBuffer:
    """
    This is specifically designed for training SAEs for individual attn heads in Llama3.
    Much redundant code; can eventually be merged to ActivationBuffer.
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """
    def __init__(self,
                 data,
                 model : LanguageModel,
                 layer,
                 n_ctxs=3e4,
                 ctx_len=128,
                 refresh_batch_size=512,
                 out_batch_size=8192,
                 device='cpu',
                 apply_W_O = False,
                 remote = False,
                 ):

        self.layer = layer
        self.n_heads = model.config.num_attention_heads
        self.resid_dim = model.config.hidden_size
        self.head_dim = self.resid_dim //self.n_heads
        self.data = data
        self.model = model
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.apply_W_O = apply_W_O
        self.remote = remote

        self.activations = t.empty(0, self.n_heads, self.head_dim, device=device)
        self.read = t.zeros(0).bool()

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
                self.refresh()

            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]

    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )

    def refresh(self):
        self.activations = self.activations[~self.read]

        while len(self.activations) < self.n_ctxs * self.ctx_len:
            with t.no_grad():
                with self.model.trace(self.text_batch(), **tracer_kwargs, invoker_args={'truncation': True, 'max_length': self.ctx_len}, remote=self.remote):
                    input = self.model.inputs.save()
                    hidden_states = self.model.model.layers[self.layer].self_attn.o_proj.inputs[0][0]
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]

                    new_shape = hidden_states.size()[:-1] + (self.n_heads, self.head_dim)
                    hidden_states = hidden_states.view(*new_shape)

                    if self.apply_W_O:
                        hidden_states_W_O_shape = hidden_states.size()[:-1] + (self.model.config.hidden_size,)
                        hidden_states_W_O = t.zeros(hidden_states_W_O_shape, device=hidden_states.device)
                        for h in range (self.n_heads):
                            start = h*self.head_dim
                            end = (h+1)*self.head_dim
                            hidden_states_W_O[..., h, start:end] = hidden_states[..., h, :]
                        hidden_states = self.model.model.layers[self.layer].self_attn.o_proj(hidden_states_W_O).save()

            attn_mask = input.value[1]['attention_mask']
            hidden_states = hidden_states[attn_mask != 0]

            self.activations = t.cat([self.activations, hidden_states.to(self.device)], dim=0)
            self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            'layer': self.layer,
            'n_ctxs' : self.n_ctxs,
            'ctx_len' : self.ctx_len,
            'refresh_batch_size' : self.refresh_batch_size,
            'out_batch_size' : self.out_batch_size,
            'device' : self.device
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()


class NNsightActivationBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """

    def __init__(
        self,
        data,
        model: LanguageModel,
        submodule,
        d_submodule=None,
        io="out",
        n_ctxs=3e4,
        ctx_len=128,
        refresh_batch_size=512,
        out_batch_size=8192,
        device="cpu",
    ):

        if io not in ["in", "out", "in_and_out"]:
            raise ValueError("io must be either 'in' or 'out' or 'in_and_out'")

        if d_submodule is None:
            try:
                if io == "in":
                    d_submodule = submodule.in_features
                else:
                    d_submodule = submodule.out_features
            except:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")

        if io in ["in", "out"]:
            self.activations = t.empty(0, d_submodule, device=device)
        elif io == "in_and_out":
            self.activations = t.empty(0, 2, d_submodule, device=device)

        self.read = t.zeros(0).bool()

        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
                self.refresh()

            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]


    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts, return_tensors="pt", max_length=self.ctx_len, padding=True, truncation=True
        )

    def token_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return t.tensor([next(self.data) for _ in range(batch_size)], device=self.device)
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        return self.token_batch(batch_size)

    def _reshaped_activations(self, hidden_states):
        hidden_states = hidden_states.value
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * seq_len, d_model)
        return hidden_states

    def refresh(self):
        self.activations = self.activations[~self.read]

        while len(self.activations) < self.n_ctxs * self.ctx_len:

            with t.no_grad(), self.model.trace(
                self.token_batch(),
                **tracer_kwargs,
                invoker_args={"truncation": True, "max_length": self.ctx_len},
            ):
                if self.io in ["in", "in_and_out"]:
                    hidden_states_in = self.submodule.inputs[0].save()
                if self.io in ["out", "in_and_out"]:
                    hidden_states_out = self.submodule.output.save()

            if self.io == "in":
                hidden_states = self._reshaped_activations(hidden_states_in)
            elif self.io == "out":
                hidden_states = self._reshaped_activations(hidden_states_out)
            elif self.io == "in_and_out":
                hidden_states_in = self._reshaped_activations(hidden_states_in).unsqueeze(1)
                hidden_states_out = self._reshaped_activations(hidden_states_out).unsqueeze(1)
                hidden_states = t.cat([hidden_states_in, hidden_states_out], dim=1)
            self.activations = t.cat([self.activations, hidden_states.to(self.device)], dim=0)
            self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            "d_submodule": self.d_submodule,
            "io": self.io,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()
