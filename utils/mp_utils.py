from typing import Optional
import torch
import torch.nn as nn

def normalize(input: torch.Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Performs :math:`L_p` normalization of inputs over specified dimension. This is fixed for mixed precision.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
        out (Tensor, optional): the output tensor. If :attr:`out` is used, this
                                operation won't be differentiable.
    """
    if nn.functional.has_torch_function_unary(input):
        return nn.functional.handle_torch_function(normalize, (input,), input, p=p, dim=dim, eps=eps, out=out)
    if out is None:
        denom = input.norm(p, dim, keepdim=True).float().clamp_min(eps).expand_as(input)
        return input / denom
    else:
        denom = input.norm(p, dim, keepdim=True).float().clamp_min_(eps).expand_as(input)
        return torch.div(input, denom, out=out)