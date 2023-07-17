import abc
import torch
from tqdm import tqdm
from jaxtyping import Float

class Sampler():

    def __init__(self, eps: Float):
        self.eps = eps

    def get_sampling_fn(self, sde, dataset):

        def sampling_fn(N_samples: Int):
            """
            return the final denoised sample, number of step,
                   timesteps, and trajectory.

            Args:
                N_samples: number of samples

            Returns:
                out: the final denoised samples (out == x_traj[-1])
                ntot (int): the total number of timesteps
                timesteps Int[Array]: the array of timesteps used
                x_traj: the entire diffusion trajectory
            """
            x = dataset[range(N_samples)] # initial sample
            timesteps = torch.linspace(0, sde.T-self.eps, sde.N)

            x_traj = torch.zeros((sde.N, *x.shape))
            with torch.no_grad():
                for i, t in enumerate(tqdm(timesteps, desc='sampling')):
                    # TODO
                    pass

            out = x
            ntot = sde.N
            return out, ntot, timesteps, x_traj

        return sampling_fn
