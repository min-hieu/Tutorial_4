import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from jaxtyping import Array, Int, Float

class PositionalEncoding(nn.Module):

    def __init__(self, t_channel: Int):
        """
        (Optional) Initialize positional encoding network

        Args:
            t_channel: number of modulation channel
        """
        super().__init__()

    def forward(self, t: Float):
        """
        Return the positional encoding of

        Args:
            t: input time

        Returns:
            emb: time embedding
        """
        emb = None
        return emb


class MLP(nn.Module):

    def __init__(self,
                 in_dim: Int,
                 out_dim: Int,
                 hid_shapes: Int[Array]):
        '''
        (TODO) Build simple MLP

        Args:
            in_dim: input dimension
            out_dim: output dimension
            hid_shapes: array of hidden layers' dimension
        '''
        super().__init__()
        self.model = None

    def forward(self, x: Array):
        return self.model(x)



class SimpleNet(nn.Module):

    def __init__(self,
                 in_dim: Int,
                 enc_shapes: Int[Array],
                 dec_shapes: Int[Array],
                 z_dim: Int):
        super().__init__()
        '''
        (TODO) Build Score Estimation network.
        You are free to modify this function signature.
        You can design whatever architecture.

        hint: it's recommended to first encode the time and x to get
        time and x embeddings then concatenate them before feeding it
        to the decoder.

        Args:
            in_dim: dimension of input
            enc_shapes: array of dimensions of encoder
            dec_shapes: array of dimensions of decoder
            z_dim: output dimension of encoder
        '''

    def forward(self, t: Array, x: Array):
        '''
        (TODO) Implement the forward pass. This should output
        the score s of the noisy input x.

        hint: you are free

        Args:
            t: the time that the forward diffusion has been running
            x: the noisy data after t period diffusion
        '''
        s = None
        return s
