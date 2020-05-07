import argparse
import os
import torch
import yaml
import matplotlib.pyplot as plt
import data
import models

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
args = parser.parse_args()

file_loc = os.path.join(args.ckpt, "opts.yaml")
opts = yaml.safe_load(open(file_loc, "r"))
opts["device"] = "cpu"

model = models.AffordanceModel(opts)
model.load(args.ckpt, "_best", 1)

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.FirstLevelDataset(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)
objects = iter(loader).next()["object"]
model.encoder1.eval()
with torch.no_grad():
    codes = model.encoder1(objects)

fig, ax = plt.subplots(5, 10, figsize=(12, 7))
for i in range(5):
    for j in range(10):
        idx = i * 10 + j
        ax[i, j].imshow(objects[idx, 12]*0.0094+0.279)
        ax[i, j].axis("off")
        ax[i, j].set_title(codes[idx].numpy())
plt.show()
