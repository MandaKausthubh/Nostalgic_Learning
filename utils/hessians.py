import torch
from utils.hvp import HessianVectorProduct
from typing import List, Tuple, Union

VectorComponentLists = List[Union[torch.Tensor, torch.nn.Parameter]]

def flatten_tensors(tensors: VectorComponentLists) -> torch.Tensor:
    """Concatenates a list of tensors into a single flat tensor."""
    return torch.cat([t.contiguous().view(-1) for t in tensors])

def unflatten_tensors(flat_tensor: torch.Tensor, tensor_list: VectorComponentLists) -> List[torch.Tensor]:
    """Splits a flat tensor back into a list of tensors with the original shapes."""
    output_tensors = []
    pointer = 0
    for tensor in tensor_list:
        num_elements = tensor.numel()
        output_tensors.append(flat_tensor[pointer:pointer + num_elements].view_as(tensor).clone())
        pointer += num_elements
    return output_tensors

def dot_product(tensor_list_1: List[torch.Tensor], tensor_list_2: List[torch.Tensor]) -> torch.Tensor:
    """Computes the dot product between two lists of tensors."""
    return sum([t1.view(-1) @ t2.view(-1) for t1, t2 in zip(tensor_list_1, tensor_list_2)]) # type: ignore


class Lanczos:
    """
    Computes the largest eigenvalues and eigenvectors of the Hessian using the Lanczos algorithm.
    It relies on a Hessian-Vector Product (HvP) function.
    """
    def __init__(self, hvp_computer: 'HessianVectorProduct', device: torch.device):
        self.hvp_computer = hvp_computer
        self.device = device
        self.num_params = self.hvp_computer.get_num_trainable_params()
        self.param_shapes = [p.shape for p in self.hvp_computer.trainable_params]

    def build_tridiagonal(self, inputs: torch.Tensor, targets: torch.Tensor, k: int, num_iters: int) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """
        Runs the Lanczos iteration to build the tridiagonal matrix T and the basis Q.

        Args:
            inputs: Model inputs (for HvP).
            targets: Model targets (for HvP).
            k: The number of eigenvalues/vectors to compute (dimensionality of T).
            num_iters: The number of power iterations to use for the initial vector.

        Returns:
            A tuple (T, Q) where T is the tridiagonal matrix and Q is the list of Lanczos basis vectors.
        """
        k = min(k, self.num_params)

        T = torch.zeros(k, k, device=self.device)
        Q = []
        q_prev = unflatten_tensors(
            torch.randn(self.num_params, device=self.device), 
            self.hvp_computer.trainable_params  # type:ignore
        )
        # Normalize the initial vector q_0
        q_norm = torch.sqrt(dot_product(q_prev, q_prev))
        q_prev = [q / q_norm for q in q_prev]
        Q.append(q_prev)

        beta = 0.0

        for j in range(k):
            q_j = Q[-1]
            w = self.hvp_computer.hvp(inputs, targets, q_j)
            alpha = dot_product(w, q_j)
            T[j, j] = alpha # Set diagonal element
            w = [w_i - alpha * q_j_i for w_i, q_j_i in zip(w, q_j)]

            if j > 0:
                q_j_minus_1 = Q[-2] # q_{j-1} is second-to-last in Q
                w = [w_i - beta * q_prev_i for w_i, q_prev_i in zip(w, q_j_minus_1)]

            beta = torch.sqrt(dot_product(w, w))
            if beta < 1e-6:
                T = T[:j+1, :j+1]
                return T, Q

            if j + 1 < k:
                T[j, j + 1] = beta
                T[j + 1, j] = beta

            q_next = [w_i / beta for w_i in w]
            Q.append(q_next) # Store q_{j+1}

        return T, Q[:-1]


    def compute_nostalgia_matrix(self, inputs: torch.Tensor, targets: torch.Tensor, k: int = 10):
        """
        Computes the top-k eigenvalues and eigenvectors of the Hessian to form the Q matrix.
        """
        T, Q_list = self.build_tridiagonal(inputs, targets, k=k, num_iters=1)

        eigenvalues, eig_vectors = torch.linalg.eigh(T)
        top_k_eig_vectors = eig_vectors[:, -k:]

        Q_tensor = [flatten_tensors(q) for q in Q_list]
        Q_tensor = torch.stack(Q_tensor, dim=1)

        V_H = Q_tensor @ top_k_eig_vectors
        return V_H, eigenvalues[-k:]


def reshape_eigenvectors_to_model_structure(
    V_H: torch.Tensor,
    trainable_params: VectorComponentLists
) -> List[VectorComponentLists]:
    """
    Transforms the flat eigenvector matrix V_H (N x k) back into a list of 
    k eigenvectors, each in the model's native list-of-tensors format.

    Args:
        V_H: The matrix of eigenvectors (N x k).
        trainable_params: The reference list of parameters to define the shapes.

    Returns:
        A list containing k eigenvectors, where each eigenvector is a 
        list of tensors matching the model's structure.
    """
    N, k = V_H.shape

    V_flat_eigenvectors = [V_H[:, i] for i in range(k)]
    V_structured_list = []

    for flat_vector in V_flat_eigenvectors:
        structured_vector = unflatten_tensors(flat_vector, trainable_params)
        V_structured_list.append(structured_vector)

    return V_structured_list
