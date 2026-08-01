from ..._core import builtin
from .._ops import _wmma
from ..cdna3 import buffer_load, buffer_store

__all__ = ["buffer_load", "buffer_store", "wmma"]


@builtin
def wmma(a, b, acc, _semantic=None):
    """
    Computes matrix-multiplication of a * b + acc using AMD WMMA instruction.

    Args:
        a (tensor): The operand a to be multiplied.
        b (tensor): The operand b to be multiplied.
        acc (tensor): The accumulator tensor.
    """
    return _wmma(1, a, b, acc, _semantic)
