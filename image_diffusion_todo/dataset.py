from multiprocessing import Pool
import os
from itertools import chain
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image


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


def tensor_to_pil_image(x: torch.Tensor, single_image=False):
    """
    x: [B,C,H,W]
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
        single_image = True

    x = (x * 0.5 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (x * 255).round().astype("uint8")
    images = [Image.fromarray(image) for image in images]
    if single_image:
        return images[0]
    return images


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class AFHQDataset(torch.utils.data.Dataset):
    def __init__(
        self, root: str, split: str, transform=None, max_num_images_per_cat=-1, label_offset=1
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.max_num_images_per_cat = max_num_images_per_cat
        self.label_offset = label_offset

        categories = os.listdir(os.path.join(root, split))
        self.num_classes = len(categories)

        fnames, labels = [], []
        for idx, cat in enumerate(sorted(categories)):
            category_dir = os.path.join(root, split, cat)
            cat_fnames = listdir(category_dir)
            cat_fnames = sorted(cat_fnames)
            if self.max_num_images_per_cat > 0:
                cat_fnames = cat_fnames[: self.max_num_images_per_cat]
            fnames += cat_fnames
            labels += [idx + label_offset] * len(cat_fnames)  # label 0 is for null class.

        self.fnames = fnames
        self.labels = labels

    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx]).convert("RGB")
        label = self.labels[idx]
        assert label >= self.label_offset
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)


class AFHQDataModule(object):
    def __init__(
        self,
        root: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        max_num_images_per_cat: int = -1,
        image_resolution: int = 64,
        label_offset=1,
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.afhq_root = os.path.join(root, "afhq")
        self.max_num_images_per_cat = max_num_images_per_cat
        self.image_resolution = image_resolution
        self.label_offset = label_offset

        if not os.path.exists(self.afhq_root):
            print(f"{self.afhq_root} is empty. Downloading AFHQ dataset...")
            self._download_dataset()

        self._set_dataset()

    def _set_dataset(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_resolution, self.image_resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.train_ds = AFHQDataset(
            self.afhq_root,
            "train",
            self.transform,
            max_num_images_per_cat=self.max_num_images_per_cat,
            label_offset=self.label_offset
        )
        self.val_ds = AFHQDataset(
            self.afhq_root,
            "val",
            self.transform,
            max_num_images_per_cat=self.max_num_images_per_cat,
            label_offset=self.label_offset,
        )

        self.num_classes = self.train_ds.num_classes

    def _download_dataset(self):
        URL = "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0"
        ZIP_FILE = f"./{self.root}/afhq.zip"
        os.system(f"mkdir -p {self.root}")
        os.system(f"wget -N {URL} -O {ZIP_FILE}")
        os.system(f"unzip {ZIP_FILE} -d {self.root}")
        os.system(f"rm {ZIP_FILE}")


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

if __name__ == "__main__":
    data_module = AFHQDataModule("data", 32, 4, -1, 64, 1)

    eval_dir = Path(data_module.afhq_root) / "eval"
    eval_dir.mkdir(exist_ok=True)
    def func(path):
        fn = path.name
        cmd = f"cp {path} {eval_dir / fn}"
        os.system(cmd)
        img = Image.open(str(eval_dir / fn))
        img = img.resize((64,64))
        img.save(str(eval_dir / fn))
        print(fn)

    with Pool(8) as pool:
        pool.map(func, data_module.val_ds.fnames)

    print(f"Constructed eval dir at {eval_dir}")
