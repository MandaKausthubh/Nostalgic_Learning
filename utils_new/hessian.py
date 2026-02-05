import torch
import time
from utils_new.hvp import HessianVectorProduct
from utils_new.lanczos import Lanczos


def compute_Q_for_task(
    model,
    train_loader,
    device,
    k=32,
):
    """
    Computes (Q_t, Lambda_t) for a single task using Lanczos.
    MPS-safe: small eigendecompositions are done on CPU.
    """

    # --------------------------------------------------
    # 1. One fixed representative batch
    # --------------------------------------------------
    inputs, targets = next(iter(train_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    model.eval()

    # --------------------------------------------------
    # 2. Trainable backbone parameters only
    # --------------------------------------------------
    backbone_params = model.get_backbone_params()

    # --------------------------------------------------
    # 3. Hessian–Vector Product helper
    # --------------------------------------------------
    hvp = HessianVectorProduct(
        model=model,
        loss_fn=model.criterion,
        device=device,
        trainable_params=backbone_params,
    )

    # --------------------------------------------------
    # 4. Lanczos on MPS (HVP-heavy part)
    # --------------------------------------------------
    lanczos = Lanczos(hvp, device=device)

    start = time.time()
    T, Q_basis = lanczos.run(inputs, targets, k)
    end = time.time()

    T = (T + T.T) / 2.0
    eps = 1e-6 * torch.trace(T) / T.shape[0]
    T = T + eps * torch.eye(T.shape[0], device=T.device)

    data_type = T.dtype

    print(f"[Lanczos] k={Q_basis.shape[1]} | time={end - start:.2f}s")

    # --------------------------------------------------
    # 5. Move small matrices to CPU for eigendecomposition
    # --------------------------------------------------
    if T.device.type == ('mps'):
        print("[Lanczos] Moving T and Q to CPU for eigendecomposition... MPS compatibility issue")
        T = T.detach().cpu()
        Q_basis = Q_basis.detach().cpu()

    eigvals, eigvecs = torch.linalg.eigh(T.to(torch.float64))
    eigvals = eigvals.to(dtype=data_type)
    eigvecs = eigvecs.to(dtype=data_type)

    # Sort descending
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    k_eff = min(k, eigvecs.shape[1])
    eigvals = eigvals[:k_eff]
    eigvecs = eigvecs[:, :k_eff]

    # --------------------------------------------------
    # 6. Lift eigenvectors back to full space (CPU → MPS)
    # --------------------------------------------------
    Q_t_cpu = Q_basis @ eigvecs          # (N × k)
    Q_t = Q_t_cpu.to(device)

    Lambda_t = eigvals.to(device)

    return Q_t, Lambda_t
