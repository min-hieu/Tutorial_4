from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int = 1000, beta_1: float = 1e-4, beta_T: float = 0.02, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        # TODO: Compute alphas and alphas_cumprod
        # alphas and alphas_cumprod correspond to $\alpha$ and $\bar{\alpha}$ in the DDPM paper (https://arxiv.org/abs/2006.11239).
        alphas = alphas_cumprod = betas

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts


class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
    
        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor(1.0), self.alphas_cumprod[-1:]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    def step(self, sample: torch.Tensor, timestep: int, noise_pred: torch.Tensor):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            sample (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep t.
            timestep (`int`): current timestep in a reverse process.
            noise_pred (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= x_{t-1})
        """

        # TODO: Implement the DDPM's one step denoising function.
        # Refer to Algorithm 2 in the DDPM paper (https://arxiv.org/abs/2006.11239).

        sample_prev = sample

        return sample_prev

    def add_noise(
        self,
        original_sample: torch.Tensor,
        timesteps: torch.IntTensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            sample (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            timesteps: (`torch.IntTensor [B]`)
            noise: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_noisy: (`torch.Tensor [B,C,H,W]`): noisy samples
            noise: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        # TODO: Implement the function that samples $\mathbf{x}_t$ from $\mathbf{x}_0$.
        # Refer to Equation 4 in the DDPM paper (https://arxiv.org/abs/2006.11239).

        noisy_sample = noise = original_sample

        return noisy_sample, noise


class DDIMScheduler(BaseScheduler):
    def __init__(self, num_train_timesteps, beta_1, beta_T, mode="linear"):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)

    def set_timesteps(
        self, num_inference_timesteps: int, device: Union[str, torch.device] = None
    ):
        """
        Sets the timesteps of a diffusion Markov chain. It is for accelerated generation process (Sec. 4.2) in the DDIM paper (https://arxiv.org/abs/2010.02502).
        """
        if num_inference_timesteps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_timesteps ({num_inference_timesteps}) cannot exceed self.num_train_timesteps ({self.num_train_timesteps})"
            )

        self.num_inference_timesteps = num_inference_timesteps

        step_ratio = self.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def step(
        self,
        sample: torch.Tensor,
        timestep: int,
        noise_pred: torch.Tensor,
        eta: float = 0.0,
    ):
        """
        One step denoising function of DDIM: $x_{\tau_i}$ -> $x_{\tau_{i-1}}$.

        Input:
            sample (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep $\tau_i$.
            timestep (`int`): current timestep in a reverse process.
            noise_pred (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
            eta (float): correspond to η in DDIM which controls the stochasticity of a reverse process.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= $x_{\tau_{i-1}}$)
        """
        # TODO: Implement the DDIM's one step denoising function.
        # Refer to Equation 12 in the DDIM paper (https://arxiv.org/abs/2010.02502).

        sample_prev = sample

        return sample_prev
