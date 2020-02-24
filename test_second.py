import torch
import numpy as np
import data
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
args = parser.parse_args()

trainset = data.SecondLevelDataset()
codes = torch.load(os.path.join(args.ckpt, "codes_second.torch"))

fig, ax = plt.subplots(10, 4, figsize=(18, 18))
unnormalized = (trainset.objects.reshape(-1, 128*128) * (trainset.obj_std + 1e-6) + trainset.obj_mu)
unnormalized = unnormalized.reshape(-1, 1, 128, 128)

for i in range(10):
    for j in range(4):
        idx = np.random.randint(0, len(trainset.relations))
        ax[i, j].imshow(unnormalized[trainset.relations[idx], 0].permute(1, 0, 2).reshape(128, 256))
        ax[i, j].axis("off")
        ax[i, j].set_title(codes[idx].numpy())
plt.show()
