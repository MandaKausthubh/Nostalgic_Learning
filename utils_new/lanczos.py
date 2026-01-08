import torch
from utils_new.hvp import HessianVectorProduct

class Lanczos:
    def __init__(self, hvp_computer: HessianVectorProduct, device):
        self.hvp = hvp_computer
        self.device = device
        self.n = hvp_computer.num_params

    def run(self, inputs, targets, k):
        k = min(k, self.n)

        Q = torch.zeros(self.n, k, device=self.device)
        alpha = torch.zeros(k, device=self.device)
        beta = torch.zeros(k, device=self.device)

        # q₀
        q = torch.randn(self.n, device=self.device)
        q = q / q.norm()
        Q[:, 0] = q

        for j in range(k):
            v = self.hvp.hvp(inputs, targets, Q[:, j])

            alpha[j] = torch.dot(Q[:, j], v)
            v = v - alpha[j] * Q[:, j]

            if j > 0:
                v = v - beta[j - 1] * Q[:, j - 1]

            # Full re-orthogonalization (important!)
            for i in range(j):
                proj = torch.dot(Q[:, i], v)
                v = v - proj * Q[:, i]

            beta_j = v.norm()
            if beta_j < 1e-6:
                Q = Q[:, :j+1]
                alpha = alpha[:j+1]
                beta = beta[:j]
                break

            beta[j] = beta_j
            if j + 1 < k:
                Q[:, j + 1] = v / beta_j

        # Tridiagonal matrix
        T = torch.diag(alpha)
        for i in range(len(beta)-1):
            T[i, i + 1] = beta[i]
            T[i + 1, i] = beta[i]

        return T, Q
