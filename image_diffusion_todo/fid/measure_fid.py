import numpy as np
import os
import torch
import torch.nn as nn
import sys
from PIL import Image
from scipy import linalg
from torchvision import transforms
from itertools import chain
from pathlib import Path
from inception import InceptionV3

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, img_size):
        self.files = files
        self.img_size = img_size
        self.transforms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_eval_loader(path, img_size, batch_size):
    def listdir(dname):
        fnames = list(
            chain(
                *[
                    list(Path(dname).rglob("*." + ext))
                    for ext in ["png", "jpg", "jpeg", "JPG"]
                ]
            )
        )
        return fnames

    files = listdir(path)
    ds = ImagePathDataset(files, img_size)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    return dl

def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
    return np.real(dist)



@torch.no_grad()
def calculate_fid_given_paths(paths, img_size=256, batch_size=50):
    print("Calculating FID given paths %s and %s..." % (paths[0], paths[1]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = InceptionV3(for_train=False)
    current_dir = Path(os.path.realpath(__file__)).parent
    ckpt = torch.load(current_dir / "afhq_inception_v3.ckpt")
    inception.load_state_dict(ckpt)
    inception = inception.eval().to(device)
    loaders = [get_eval_loader(path, img_size, batch_size) for path in paths]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, total=len(loader)):
            actv = inception(x.to(device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value

if __name__ == "__main__":
    # python measure_fid /path/to/dir1 /path/to/dir2

    paths = [sys.argv[1], sys.argv[2]]
    fid_value = calculate_fid_given_paths(paths, img_size=256, batch_size=64)
    print("FID:", fid_value)

