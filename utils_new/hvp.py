from typing import List, Optional
import torch


class HessianVectorProduct:
    """
    Computes exact Hessian-vector products using autograd (Pearlmutter trick).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        device: torch.device,
        trainable_params: Optional[List[torch.nn.Parameter]] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

        if trainable_params is None:
            self.params = [p for p in model.parameters() if p.requires_grad]
        else:
            self.params = list(trainable_params)

        self.param_numels = [p.numel() for p in self.params]
        self.num_params = sum(self.param_numels)

    # --------------------------------------------------
    def _unflatten(self, flat: torch.Tensor) -> List[torch.Tensor]:
        """Convert flat vector → parameter-shaped tensors"""
        out = []
        ptr = 0
        for p, n in zip(self.params, self.param_numels):
            out.append(flat[ptr:ptr + n].view_as(p))
            ptr += n
        return out

    # --------------------------------------------------
    def hvp(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        v_flat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute H @ v, where v is a flat vector.
        """
        assert v_flat.numel() == self.num_params

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        v_structured = self._unflatten(v_flat)

        inputs = self.model.preprocess_inputs(inputs)  # type: ignore

        # Forward
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)

        # First gradient
        grads = torch.autograd.grad(
            loss,
            self.params,
            create_graph=True,
            retain_graph=False,
        )


        # Inner product <grad, v>
        grad_dot_v = sum(
            (g.view(-1) * v.view(-1)).sum()
            for g, v in zip(grads, v_structured)
        )

        # Second gradient (HVP)
        hvp = torch.autograd.grad(
            grad_dot_v,  #type: ignore
            self.params,
            retain_graph=False,
        )

        # Flatten output
        hvp_flat = torch.cat([h.contiguous().view(-1) for h in hvp])
        return hvp_flat.detach()
