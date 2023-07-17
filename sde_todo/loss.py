import torch
import torch.nn as nn
from jaxtyping import Array
from typing import Callable

def get_div(y: Array, x: Array):
    """
    (Optional)
    Return the divergence of y wrt x. Let y = f(x). get_div return div_x(y).

    Args:
        x Input of a differentiable function
        y Output of a differentiable function

    Returns:
        div_x(y)
    """
    pass

class DSMLoss():

    def __init__(self, alpha: float, diff_weight: bool):
        """
        (TODO) Initialize the DSM Loss.

        Args:
            alpha: regularization weight
            diff_weight: scale loss by square of diffusion

        Returns:
            None
        """

    def __call__(self,
                 t: Array,
                 x: Array,
                 model: Callable[[Array], Array],
                 s: Array,
                 diff_sq: Float):
        """
        Args:
            t: uniformly sampled time period
            x: samples after t diffusion
            model: score prediction function s(x,t)
            s: ground truth score

        Returns:
            loss: average loss value
        """
        loss = None
        return loss


class ISMLoss():
    """
    (Optional) Implicit Score Matching Loss
    """

    def __init__(self):
        pass

    def __call__(self, t, x, model):
        """
        Args:
            t: uniformly sampled time period
            x: samples after t diffusion
            model: score prediction function s(x,t)

        Returns:
            loss: average loss value
        """
        return loss


class SBJLoss():
    """
    (Optional) Joint Schrodinger Bridge Loss

    hint: You will need to implement the divergence.
    """

    def __init__(self):
        pass

    def __call__(self, t, xf, zf, zb_fn):
        """
        Initialize the SBJLoss Loss.

        Args:
            t: uniformly sampled time period
            xf: samples after t forward diffusion
            zf: ground truth forward value
            zb_fn: backward Z function

        Returns:
            loss: average loss value
        """
        return loss


class SBALoss():
    """
    (Optional) Alternating Schrodinger Bridge Loss

    hint: You will need to implement the divergence.
    """

    def __init__(self):
        pass

    def __call__(self, t, xf, zf, zb_fn):
        """
        Initialize the SBALoss Loss.

        Args:
            t: uniformly sampled time period
            xf: samples after t forward diffusion
            zf: ground truth forward value
            zb_fn: backward Z function

        Returns:
            loss: average loss value
        """
        return loss
