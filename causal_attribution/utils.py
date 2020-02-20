"""Pytorch functions."""

from torch.autograd import Function
import torch.nn as nn
import torch

class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input

class ReversalLayer(nn.Module):
    def __init__(self):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """

        super().__init__()

    def forward(self, input_):
        return RevGrad.apply(input_)


def glue_dense_vectors(tensors_info, use_gpu=False):
    """ Glue together a bunch of (possibly categorical/dense) vectors.

    Args:
        tensors_info: [(tensor, information about the variable), ...]
        use_gpu: bool. Whether to use gpu.
    Returns:
        torch.FloatTensor [vatch, feature size] -- all of the vectorized
            variables concatted together.
    """
    out = []
    for tensor, info in tensors_info:
        if info['type'] == 'categorical':
            vec = make_bow_vector(
                torch.unsqueeze(tensor, -1), len(info['vocab']), use_gpu=use_gpu)
            out.append(vec)
        else:
            out.append(torch.unsqueeze(tensor, -1))

    return torch.cat(out, 1)


def make_bow_vector(ids, vocab_size, use_counts=False, use_gpu=False):
    """ Make a sparse BOW vector from a tensor of dense ids.

    Args:
        ids: torch.LongTensor [batch, features]. Dense tensor of ids.
        vocab_size: vocab size for this tensor.
        use_counts: if true, the outgoing BOW vector will contain
            feature counts. If false, will contain binary indicators.
        use_gpu: bool. Whether to use gpu
    Returns:
        The sparse bag-of-words representation of ids.
    """
    vec = torch.zeros(ids.shape[0], vocab_size)
    ones = torch.ones_like(ids, dtype=torch.float)
    if use_gpu:
        vec = vec.cuda()
        ones = ones.cuda()
        ids = ids.cuda()
    vec.scatter_add_(1, ids, ones)
    vec[:, 1] = 0.0  # zero out pad
    if not use_counts:
        vec = (vec != 0).float()
    return vec
