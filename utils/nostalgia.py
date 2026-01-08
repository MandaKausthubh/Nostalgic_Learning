import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Any


class NostalgiaOptimizer(Optimizer):
    """
    Wraps a base optimizer and applies a Nostalgia-style gradient projection:
        g' = g - Q (Q^T g)

    Projection is applied ONLY to the specified projection_params.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        base_optimizer: Optimizer,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # Important: we do NOT pass params to super(); base_optimizer owns them
        self.base_optimizer = base_optimizer

        self.projection_params = list(params)
        self.device = device
        self.dtype = dtype

        self.nostalgia_Q: Optional[torch.Tensor] = None
        self.scaling: Optional[torch.Tensor] = None

        # Fixed parameter layout (ordering matters!)
        self.param_numels = [p.numel() for p in self.projection_params]
        self.num_params = sum(self.param_numels)

    # ------------------------------------------------------------------
    def set_Q(self, Q: torch.Tensor, scaling: Optional[torch.Tensor] = None):
        """
        Q: [num_params, k] matrix of eigenvectors
        scaling: optional [k] or [k, k] eigenvalue-based scaling
        """
        if Q.shape[0] != self.num_params:
            raise ValueError(
                f"Q has {Q.shape[0]} rows, expected {self.num_params} "
                f"(sum of projection parameter sizes)."
            )

        self.nostalgia_Q = Q.to(self.device, self.dtype)
        self.scaling = scaling.to(self.device, self.dtype) if scaling is not None else None

    # ------------------------------------------------------------------
    def _flatten_grads(self) -> torch.Tensor:
        """
        Flatten gradients of ALL projection params in fixed order.
        Missing gradients are treated as zeros.
        """
        flat_grads = []
        for p in self.projection_params:
            if p.grad is None:
                flat_grads.append(torch.zeros(
                    p.numel(), device=self.device, dtype=self.dtype
                ))
            else:
                flat_grads.append(p.grad.view(-1))
        return torch.cat(flat_grads)

    # ------------------------------------------------------------------
    def _unflatten_to_grads(self, flat_grad: torch.Tensor):
        """
        Write projected flat gradient back into parameter .grad fields.
        """
        pointer = 0
        for p, n in zip(self.projection_params, self.param_numels):
            grad_slice = flat_grad[pointer:pointer + n].view_as(p)
            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.copy_(grad_slice)
            pointer += n

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Optional[Any] = None): #type: ignore
        if self.nostalgia_Q is not None:
            g = self._flatten_grads()

            # Q^T g
            coeffs = self.nostalgia_Q.T @ g

            # Optional eigenvalue-aware scaling
            if self.scaling is not None:
                if self.scaling.ndim == 1:
                    coeffs = coeffs * self.scaling
                else:
                    coeffs = self.scaling @ coeffs

            # Q (Q^T g)
            projection = self.nostalgia_Q @ coeffs

            g_projected = g - projection
            self._unflatten_to_grads(g_projected)

        return self.base_optimizer.step(closure)

    # ------------------------------------------------------------------
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
