import torch


def _row_histogram(x: torch.Tensor, bins: int):
    """Build one equal-width histogram for every row on x.device."""
    row_min = x.amin(dim=1)
    row_max = x.amax(dim=1)
    is_constant = row_min == row_max
    low = torch.where(is_constant, row_min - 0.5, row_min)
    high = torch.where(is_constant, row_max + 0.5, row_max)

    indices = torch.floor(
        (x - low[:, None]) * (bins / (high - low)[:, None])
    ).long()
    indices.clamp_(0, bins - 1)
    counts = torch.zeros(
        (x.shape[0], bins), dtype=torch.float32, device=x.device
    )
    counts.scatter_add_(1, indices, torch.ones_like(x, dtype=torch.float32))

    steps = torch.arange(bins + 1, dtype=x.dtype, device=x.device)
    edges = low[:, None] + steps[None, :] * ((high - low)[:, None] / bins)
    return counts, edges


def _smooth(values: torch.Tensor, valid: torch.Tensor, eps: float):
    zeros = valid & (values == 0)
    nonzeros = valid & (values != 0)
    n_zeros = zeros.sum(dim=-1, keepdim=True).to(values.dtype)
    n_nonzeros = nonzeros.sum(dim=-1, keepdim=True).clamp_min(1).to(values.dtype)
    eps1 = eps * n_zeros / n_nonzeros
    return torch.where(
        valid,
        values + eps * zeros.to(values.dtype) - eps1 * nonzeros.to(values.dtype),
        torch.zeros((), dtype=values.dtype, device=values.device),
    )


def _best_cut_batch(
    side_counts: torch.Tensor,
    side_length: torch.Tensor,
    target_bins: torch.Tensor,
    bins: int,
    eps: float = 1e-4,
):
    """Evaluate all KL cut candidates for every row in parallel."""
    device = side_counts.device
    cut = torch.arange(bins, device=device).view(1, bins, 1)
    col = torch.arange(bins, device=device).view(1, 1, bins)
    candidate = (
        (cut.squeeze(-1) > target_bins[:, None])
        & (cut.squeeze(-1) < side_length[:, None])
    )
    valid = col < cut
    in_side = col < side_length[:, None, None]
    base = side_counts[:, None, :].expand(-1, bins, -1)

    # counts[:i], with the discarded tail accumulated into bin i - 1.
    sliced = torch.where(valid, base, torch.zeros_like(base))
    tail = torch.where(
        (col >= cut) & in_side, base, torch.zeros_like(base)
    ).sum(dim=-1, keepdim=True)
    sliced = sliced + (col == cut - 1).to(sliced.dtype) * tail

    # Reproduce the original merge and expand loops.
    safe_target = target_bins.clamp(1, bins)
    merged_width = torch.div(
        cut, safe_target[:, None, None], rounding_mode="floor"
    ).clamp_min(1)
    group = torch.div(col, merged_width, rounding_mode="floor")
    group = torch.minimum(group, (safe_target - 1)[:, None, None]).long()
    group = group.expand(-1, bins, -1)

    nonzero = valid & (sliced != 0)
    merged = torch.zeros_like(sliced)
    merged.scatter_add_(
        2, group, torch.where(valid, sliced, torch.zeros_like(sliced))
    )
    norms = torch.zeros_like(sliced)
    norms.scatter_add_(2, group, nonzero.to(sliced.dtype))
    expanded = merged.gather(2, group) / norms.gather(2, group).clamp_min(1)
    expanded = torch.where(nonzero, expanded, torch.zeros_like(expanded))

    # SciPy entropy uses double precision, so keep KL evaluation in float64.
    p = _smooth(sliced, valid, eps).to(torch.float64)
    q = _smooth(expanded, valid, eps).to(torch.float64)
    tiny = torch.finfo(torch.float64).tiny
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(tiny)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(tiny)
    kl = torch.sum(torch.special.xlogy(p, p / q.clamp_min(tiny)), dim=-1)
    kl = torch.where(candidate, kl, torch.full_like(kl, torch.inf))
    return torch.argmin(kl, dim=1), candidate.any(dim=1)


def _process_chunk(x: torch.Tensor, maxq: int, bins: int):
    counts, edges = _row_histogram(x, bins)
    mode = torch.argmax(counts, dim=1)
    positions = torch.arange(bins, device=x.device)[None, :]
    left_length = mode
    right_length = bins - mode

    left_indices = (mode[:, None] - 1 - positions).clamp(0, bins - 1)
    right_indices = (mode[:, None] + positions).clamp(0, bins - 1)
    left_counts = torch.where(
        positions < left_length[:, None], counts.gather(1, left_indices), 0
    )
    right_counts = torch.where(
        positions < right_length[:, None], counts.gather(1, right_indices), 0
    )

    target_left = torch.div(maxq * left_length, bins, rounding_mode="floor")
    target_left = torch.where(
        target_left == 0, torch.ones_like(target_left), target_left
    )
    target_right = maxq - target_left
    best_left, has_left = _best_cut_batch(
        left_counts, left_length, target_left, bins
    )
    best_right, has_right = _best_cut_batch(
        right_counts, right_length, target_right, bins
    )

    # Original kl_index + target_bin + 3 equals candidate cut + 2.
    left_offset = best_left + 2
    right_offset = best_right + 2
    left_index = (mode - left_offset).clamp(0, bins)
    right_index = (mode + right_offset).clamp(0, bins)
    left_edge = edges.gather(1, left_index[:, None]).squeeze(1)
    right_edge = edges.gather(1, right_index[:, None]).squeeze(1)
    first_edge = edges[:, 0]
    last_edge = edges[:, -1]

    left_regular = torch.where(
        left_offset >= left_length, first_edge * 0.99, left_edge
    )
    right_regular = torch.where(
        right_offset >= right_length + 1, last_edge * 0.99, right_edge
    )
    lower = torch.where(
        left_length <= 2,
        first_edge,
        torch.where(has_left, left_regular, first_edge * 0.99),
    )
    upper = torch.where(
        right_length <= 2,
        last_edge,
        torch.where(has_right, right_regular, last_edge * 0.99),
    )
    return upper.to(torch.float32), lower.to(torch.float32)


@torch.no_grad()
def find_outlier_batch(
    maxq: int,
    data: torch.Tensor,
    bins: int,
    chunk_size: int = 256,
):
    """Return independent KL upper/lower thresholds for all CUDA rows."""
    if not isinstance(data, torch.Tensor):
        raise TypeError("data must be a torch.Tensor")
    if data.ndim != 2:
        raise ValueError("data must have shape [N, D]")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("data must not be empty")
    if not data.is_cuda:
        raise ValueError("data must already be on a CUDA device")
    if not isinstance(bins, int) or bins < 3:
        raise ValueError("bins must be an integer >= 3")
    if not isinstance(maxq, int) or not 2 <= maxq <= bins:
        raise ValueError("maxq must be an integer in [2, bins]")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")

    x = data.detach()
    upper_parts = []
    lower_parts = []
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        chunk = x[start:stop].to(dtype=torch.float32)
        upper, lower = _process_chunk(chunk, maxq, bins)
        upper_parts.append(upper)
        lower_parts.append(lower)
    return torch.cat(upper_parts), torch.cat(lower_parts)


@torch.no_grad()
def find_outlier(maxq: int, f2: torch.Tensor, bins: int):
    """Backward-compatible single-row CUDA wrapper."""
    upper, lower = find_outlier_batch(
        maxq=maxq,
        data=f2.reshape(1, -1),
        bins=bins,
        chunk_size=1,
    )
    return upper[0], lower[0]
