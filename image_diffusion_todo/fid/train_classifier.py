from inception import InceptionV3
import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
from dataset import AFHQDataset, AFHQDataModule
from tqdm import tqdm
import cv2
import numpy as np

data_module = AFHQDataModule("/home/juil/workspace/23summer_tutorial/HelloScore/image_diffusion/data/", 32, 4, -1, 256, 0)

train_dl = data_module.train_dataloader()
val_dl = data_module.val_dataloader()

device = f"cuda:1"

net = InceptionV3(for_train=True)
net = net.to(device)
net.train()
for n, p in net.named_parameters():
    p.requires_grad_(True)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

epochs = 10
for epoch in range(epochs):
    pbar = tqdm(train_dl)
    net.train()
    for img, label in pbar:
        img, label = img.to(device), label.to(device)
        pred = net(img)
        loss = F.cross_entropy(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_label = pred.max(-1)[1]
        acc = (pred_label == label).float().mean()

        pbar.set_description(f"E {epoch} | loss: {loss:.4f} acc: {acc*100:.2f}%")

    net.eval()
    val_accs = []
    for img, label in val_dl:
        img, label = img.to(device), label.to(device)
        pred = net(img)
        pred_label = pred.max(-1)[1]

        acc = (pred_label == label).float().mean()
        val_accs.append(acc)
    print(f"Val Acc: {sum(val_accs) / len(val_accs) * 100:.2f}%")

torch.save(net.state_dict(), "afhq_inception_v3.ckpt")
print("Saved model")

