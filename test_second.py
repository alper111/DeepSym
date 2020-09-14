import argparse
import os
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import data
from models import EffectRegressorMLP

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
args = parser.parse_args()

file_loc = os.path.join(args.ckpt, "opts.yaml")
opts = yaml.safe_load(open(file_loc, "r"))
opts["device"] = "cpu"

model = EffectRegressorMLP(opts)
model.load(args.ckpt, "_best", 1)
model.load(args.ckpt, "_best", 2)
model.encoder1.eval()
model.encoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.PairedObjectData(transform=transform)
trainset.train = False
loader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=True)
objects = iter(loader).next()["observation"]
with torch.no_grad():
    codes = model.encoder2(objects)

fig, ax = plt.subplots(6, 6, figsize=(10, 6))
for i in range(6):
    for j in range(6):
        idx = i * 6 + j
        ax[i, j].imshow(objects[idx].permute(1, 0, 2).reshape(objects.shape[3], objects.shape[3]*2)*0.0094+0.279)
        ax[i, j].axis("off")
        ax[i, j].set_title(codes[idx].numpy())
plt.show()
pp = PdfPages("paired.pdf")
pp.savefig()
pp.close()
