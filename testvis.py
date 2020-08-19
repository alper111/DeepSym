import argparse
import os
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import data
from models import EffectRegressorConv

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
args = parser.parse_args()

file_loc = os.path.join(args.ckpt, "opts.yaml")
opts = yaml.safe_load(open(file_loc, "r"))
opts["device"] = "cpu"

model = EffectRegressorConv(opts)
model.load(args.ckpt, "_best")

trainset = data.ImageFirstLevel()
loader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)
objects = iter(loader).next()["observation"]
model.encoder.eval()
with torch.no_grad():
    codes = model.encoder(objects)

fig, ax = plt.subplots(5, 10, figsize=(12, 7))
for i in range(5):
    for j in range(10):
        idx = i * 10 + j
        ax[i, j].imshow(objects[idx, 0]*(trainset.obs_std+1e-6) + trainset.obs_mu)
        # ax[i, j].imshow(objects[idx, 0])
        ax[i, j].axis("off")
        ax[i, j].set_title(codes[idx].numpy())
plt.show()
pp = PdfPages("codes.pdf")
pp.savefig(fig)
pp.close()
