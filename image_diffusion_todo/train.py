import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dataset import AFHQDataModule, get_data_iterator, tensor_to_pil_image
from dotmap import DotMap
from ddpm import DiffusionModule
from network import UNet
from pytorch_lightning import seed_everything
from scheduler import DDIMScheduler, DDPMScheduler
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    now = get_current_time()
    save_dir = Path(f"results/diffusion-{now}")
    save_dir.mkdir(exist_ok=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    image_resolution = config.image_resolution
    ds_module = AFHQDataModule(
        "./data",
        batch_size=config.batch_size,
        num_workers=4,
        max_num_images_per_cat=config.max_num_images_per_cat,
        image_resolution=image_resolution
    )

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    var_scheduler = DDPMScheduler(
        config.num_diffusion_train_timesteps,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        mode="linear",
    )
    if isinstance(var_scheduler, DDIMScheduler):
        var_scheduler.set_timesteps(20)  # 20 steps are enough in the case of DDIM.

    network = UNet(
        T=config.num_diffusion_train_timesteps,
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=getattr(ds_module, "num_classes", None),
    )

    ddpm = DiffusionModule(network, var_scheduler)
    ddpm = ddpm.to(config.device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    step = 0
    losses = []
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                ddpm.eval()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                samples = ddpm.sample(4, return_traj=False)
                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                ddpm.save(f"{save_dir}/last.ckpt")
                ddpm.train()

            img, label = next(train_it)
            img, label = img.to(config.device), label.to(config.device)
            loss = ddpm.get_loss(img, class_label=label)
            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)

    print(f"last.ckpt is saved at {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument(
        "--max_num_images_per_cat",
        type=int,
        default=-1,
        help="max number of images per category for AFHQ dataset",
    )
    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
        help="diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--image_resolution", type=int, default=64)

    args = parser.parse_args()
    main(args)
