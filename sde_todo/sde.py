import abc
import torch
import numpy as np
from jaxtyping import Array

class SDE(abc.ABC):
    def __init__(self, N: int, T: int):
        super().__init__()
        self.N = N         # number of discretization steps
        self.T = T         # terminal time
        self.dt = T / N
        self.is_reverse = False
        self.is_bridge = False

    @abc.abstractmethod
    def sde_coeff(self, t, x):
        return NotImplemented

    @abc.abstractmethod
    def marginal_prob(self, t, x):
        return NotImplemented

    @abc.abstractmethod
    def predict_fn(self, x):
        return NotImplemented

    @abc.abstractmethod
    def correct_fn(self, t, x):
        return NotImplemented

    def dw(self, x, dt=None):
        """
        (TODO) Return the differential of Brownian motion

        Args:
            x: input data

        Returns:
            dw (same shape as x)
        """
        dt = self.dt if dt is None else dt
        return None

    def prior_sampling(self, x: Array):
        """
        Sampling from prior distribution. Default to unit gaussian.

        Args:
            x: input data

        Returns:
            z: random variable with same shape as x
        """
        return torch.randn_like(x)

    def predict_fn(self,
                   t: Array,
                   x: Array,
                   dt: Float=None):
        """
        (TODO) Perform single step diffusion.

        Args:
            t: current diffusion time
            x: input with noise level at time t
            dt: the discrete time step. Default to T/N

        Returns:
            x: input at time t+dt
        """
        dt = self.dt if dt is None else dt
        pred = None
        return pred

    def correct_fn(self, t: Array, x: Array):
        return None

    def reverse(self, model):
        N = self.N
        T = self.T
        forward_sde_coeff = self.sde_coeff

        class RSDE(self.__class__):
            def __init__(self, score_fn):
                super().__init__(N, T)
                self.score_fn = score_fn
                self.is_reverse = True
                self.forward_sde_coeff = forward_sde_coeff

            def sde_coeff(self, t: Array, x: Array):
                """
                (TODO) Return the reverse drift and diffusion terms.

                Args:
                    t: current diffusion time
                    x: current input at time t

                Returns:
                    reverse_f: reverse drift term
                    g: reverse diffusion term
                """
                reverse_f = None
                g         = None
                return reverse_f, g

            def ode_coeff(self, t: Array, x: Array):
                """
                (Optional) Return the reverse drift and diffusion terms in
                ODE sampling.

                Args:
                    t: current diffusion time
                    x: current input at time t

                Returns:
                    reverse_f: reverse drift term
                    g: reverse diffusion term
                """
                reverse_f = None
                g         = None
                return reverse_f, g

            def predict_fn(self,
                           t: Array,
                           x,
                           dt=None,
                           ode=False):
                """
                (TODO) Perform single step reverse diffusion

                """
                return x

        return RSDE(model)

class OU(SDE):
    def __init__(self, N=1000, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        f, g = None, None
        return f, g

    def marginal_prob(self, t, x):
        mean, std = None, None
        return mean, std

class VESDE(SDE):
    def __init__(self, N=100, T=1, sigma_min=0.01, sigma_max=50):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        f, g = None, None
        return f, g

    def marginal_prob(self, t, x):
        mean, std = None, None
        return mean, std


class VPSDE(SDE):
    def __init__(self, N=1000, T=1, beta_min=0.1, beta_max=20):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        f, g = None, None
        return f, g

    def marginal_prob(self, t, x):
        mean, std = None, None
        return mean, std


class SB(abc.ABC):
    def __init__(self, N=1000, T=1, zf_model=None, zb_model=None):
        super().__init__()
        self.N = N         # number of time step
        self.T = T         # end time
        self.dt = T / N

        self.is_reverse = False
        self.is_bridge  = True

        self.zf_model = zf_model
        self.zb_model = zb_model

    def dw(self, x, dt=None):
        dt = self.dt if dt is None else dt
        return torch.randn_like(x) * (dt**0.5)

    @abc.abstractmethod
    def sde_coeff(self, t, x):
        return NotImplemented

    def sb_coeff(self, t, x):
        """
        (Optional) Return the SB reverse drift and diffusion terms.

        Args:
        """
        sb_f = None
        g    = None
        return sb_f, g

    def predict_fn(self,
                   t: Array,
                   x: Array,
                   dt:Float =None):
        """
        Args:
            t:
            x:
            dt:
        """
        return x

    def correct_fn(self, t, x, dt=None):
        return x

    def reverse(self, model):
        """
        (Optional) Initialize the reverse process
        """

        class RSB(self.__class__):
            def __init__(self, model):
                super().__init__(N, T, zf_model, zb_model)
                """
                (Optional) Initialize the reverse process
                """

            def sb_coeff(self, t, x):
                """
                (Optional) Return the SB reverse drift and diffusion terms.
                """
                sb_f = None
                g    = None
                return sb_f, g

        return RSDE(model)

class OUSB(SB):
    def __init__(self, N=1000, T=1, zf_model=None, zb_model=None):
        super().__init__(N, T, zf_model, zb_model)

    def sde_coeff(self, t, x):
        f = -0.5 * x
        g = torch.ones(x.shape)
        return f, g
