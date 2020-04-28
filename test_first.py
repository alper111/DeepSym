import torch
import data
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
args = parser.parse_args()

trainset = data.FirstLevelDataset()
codes = torch.load(os.path.join(args.ckpt, "codes_first.torch"))

fig, ax = plt.subplots(5, 10, figsize=(12, 7))

for i in range(5):
    for j in range(10):
        idx = i * 10 + j
        ax[i, j].imshow(trainset.objects[idx, 12])
        ax[i, j].axis("off")
        ax[i, j].set_title(codes[idx].numpy())
plt.show()
