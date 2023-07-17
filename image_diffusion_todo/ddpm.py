from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from scheduler import BaseScheduler


class DiffusionModule(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: BaseScheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)
        x_noisy, noise = self.var_scheduler.add_noise(x0, timestep)
        noise_pred = self.network(x_noisy, timestep=timestep, class_label=class_label)

        loss = F.mse_loss(noise_pred.flatten(), noise.flatten(), reduction="mean")
        return loss

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        return_traj: bool = False,
    ):
        """
        Sample x_0 from a learned diffusion model.
        """
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(
            self.device
        )

        traj = [x_T]
        for t in self.var_scheduler.timesteps:
            x_t = traj[-1]
            noise_pred = self.network(x_t, timestep=t.to(self.device))

            x_t_prev = self.var_scheduler.step(x_t, t, noise_pred)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
                "network": self.network,
                "var_scheduler": self.var_scheduler,
                } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)

