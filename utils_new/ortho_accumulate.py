import torch
from typing import Optional, Tuple

@torch.no_grad()
def update_Q_lambda_union(
    Q_union: Optional[torch.Tensor],
    lambda_union: Optional[torch.Tensor],
    Q_new: torch.Tensor,
    lambda_new: torch.Tensor,
    k_max: Optional[int] = None,
    eps_overlap: float = 1e-3,
    eps_rank: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update the union of protected subspaces and corresponding lambdas
    using a monotone (max-based) retention rule.

    Args:
        Q_union: (n, r) existing union basis, or None
        lambda_union: (r,) existing lambdas, or None
        Q_new: (n, k_new) new task eigenvectors (assumed approx orthonormal)
        lambda_new: (k_new,) new task eigenvalues
        k_max: optional cap on total rank
        eps_overlap: threshold for detecting directional overlap
        eps_rank: threshold for discarding numerically empty directions

    Returns:
        Q_union_new: (n, r_new) updated union basis
        lambda_union_new: (r_new,) updated lambdas
    """

    device = Q_new.device
    dtype = Q_new.dtype

    Q_new = Q_new.to(device, dtype)
    lambda_new = lambda_new.to(device, dtype)

    # --------------------------------------------------
    # First task: initialize
    # --------------------------------------------------
    if Q_union is None or lambda_union is None:
        Q_init, _ = torch.linalg.qr(Q_new, mode="reduced")
        return Q_init, lambda_new.clone()

    Q_union = Q_union.to(device, dtype)
    lambda_union = lambda_union.to(device, dtype)

    # --------------------------------------------------
    # Step 1: handle overlap with existing union space
    # --------------------------------------------------
    overlap = Q_union.T @ Q_new   # (r × k_new)

    lambda_union_new = lambda_union.clone()

    for j in range(Q_new.shape[1]):
        weights = overlap[:, j].abs()
        max_w, idx = weights.max(dim=0)
        if max_w > eps_overlap:
            # Monotone (max) update
            lambda_union_new[idx] = torch.max(
                lambda_union_new[idx], lambda_new[j]
            )

    # --------------------------------------------------
    # Step 2: extract genuinely new directions
    # --------------------------------------------------
    Q_new_orth = Q_new - Q_union @ overlap
    Q_new_orth, R = torch.linalg.qr(Q_new_orth, mode="reduced")

    diag = torch.abs(torch.diag(R))
    keep = diag > eps_rank

    if keep.sum() == 0:
        return Q_union, lambda_union_new

    Q_new_orth = Q_new_orth[:, keep]
    lambda_new_orth = lambda_new[keep]

    # --------------------------------------------------
    # Step 3: append
    # --------------------------------------------------
    Q_union_new = torch.cat([Q_union, Q_new_orth], dim=1)
    lambda_union_new = torch.cat(
        [lambda_union_new, lambda_new_orth], dim=0
    )

    # --------------------------------------------------
    # Step 4: enforce capacity (optional)
    # --------------------------------------------------
    if k_max is not None and Q_union_new.shape[1] > k_max:
        Q_union_new = Q_union_new[:, :k_max]
        lambda_union_new = lambda_union_new[:k_max]

    return Q_union_new, lambda_union_new
