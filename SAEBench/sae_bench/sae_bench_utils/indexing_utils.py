import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor

def get_zero_activation_sequences(
    x: Float[Tensor, "batch seq"],
    k: int,
    buffer: int,
) -> Int[Tensor, "k 2"]:
    """
    Finds k sequence indices where all activation values within the
    sequence (including buffer) are less than the specified threshold.

    Returns indices for the *center* of the zero-activation sequences.
    """

    dataset_size, seq_len = x.shape

    is_active = (x > 0)



    is_entirely_inactive = (is_active.sum(dim=1) == 0)

    inactive_indices = torch.where(is_entirely_inactive)[0]

    if inactive_indices.numel() < k:
        num_to_take = inactive_indices.numel()
    else:
        perm = torch.randperm(inactive_indices.numel())
        num_to_take = k
        inactive_indices = inactive_indices[perm][:k]

    if num_to_take == 0:
        return torch.empty((0, 2), dtype=torch.int64, device=x.device)

    col_start = buffer
    col_end = seq_len - buffer

    rand_cols = torch.randint(
        col_start, col_end, (num_to_take,), device=x.device
    )

    rows = inactive_indices[:num_to_take]

    return torch.stack((rows, rand_cols), dim=1)


def get_k_largest_indices(
    x: Float[Tensor, "batch seq"],
    k: int,
    buffer: int = 0,
    no_overlap: bool = False,
) -> Int[Tensor, "k 2"]:
    """
    Args:
        x:          The 2D tensor to get the top k largest elements from.
        k:          The number of top elements to get.
        buffer:     We won't choose any elements within `buffer` from the start or end of their seq (this helps if we
                    want more context around the chosen tokens).
        no_overlap: If True, this ensures that no 2 top-activating tokens are in the same seq and within `buffer` of
                    each other.

    Returns:
        indices: The index positions of the top k largest elements.
    """
    x = x[:, buffer:-buffer]
    indices = x.flatten().argsort(-1, descending=True)
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer

    if no_overlap:
        unique_indices = []
        seen_positions = set()
        for row, col in zip(rows.tolist(), cols.tolist()):
            if (row, col) not in seen_positions:
                unique_indices.append((row, col))
                for offset in range(-buffer, buffer + 1):
                    seen_positions.add((row, col + offset))
            if len(unique_indices) == k:
                break
        rows, cols = torch.tensor(
            unique_indices, dtype=torch.int64, device=x.device
        ).unbind(dim=-1)

    return torch.stack((rows, cols), dim=1)[:k]


def get_iw_sample_indices(
    x: Float[Tensor, "batch seq"],
    k: int,
    buffer: int = 0,
    use_squared_values: bool = True,
) -> Int[Tensor, "k 2"]:
    """
    This function returns k indices from x, importance-sampled (i.e. chosen with probabilities in proportion to their
    values). This is mean to be an alternative to topTauPerFeatSquare sampling, which accomplishes a similar thing.

    Also includes an optional threshold above which we won't sample.
    """
    x = x[:, buffer:-buffer]
    if use_squared_values:
        x = x.pow(2)

    probabilities = x.flatten() / x.sum()
    indices = torch.multinomial(probabilities, k, replacement=False)

    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer
    return torch.stack((rows, cols), dim=1)[:k]


def index_with_buffer(
    x: Float[Tensor, "batch seq"],
    indices: Int[Tensor, "k 2"],
    buffer: int = 0,
) -> Float[Tensor, "k buffer_x2_plus1"]:
    """
    This function returns the tensor you get when indexing into `x` with indices, and taking a +-buffer range around
    each index. For example, if `indices` is a list of the top activating tokens (returned by `get_k_largest_indices`),
    then this function can get you the sequence context.
    """
    assert indices.ndim == 2, "indices must have 2 dimensions"
    assert indices.shape[1] == 2, "indices must have 2 columns"
    rows, cols = indices.unbind(dim=-1)
    rows = einops.repeat(rows, "k -> k buffer", buffer=buffer * 2 + 1)
    cols = einops.repeat(cols, "k -> k buffer", buffer=buffer * 2 + 1) + torch.arange(
        -buffer, buffer + 1, device=cols.device
    )
    return x[rows, cols]
