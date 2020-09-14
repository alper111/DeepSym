import argparse
import os
import torch
import torchvision
import yaml
import data
from models import EffectRegressorMLP
import matplotlib
import matplotlib.cm as cm

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
args = parser.parse_args()

file_loc = os.path.join(args.ckpt, "opts.yaml")
opts = yaml.safe_load(open(file_loc, "r"))
opts["device"] = "cpu"

model = EffectRegressorMLP(opts)
model.load(args.ckpt, "_best", 1)
model.load(args.ckpt, "_best", 2)
model.encoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=2400, shuffle=False)
objects = iter(loader).next()["observation"]
objects = objects.reshape(5, 10, 3, 4, 4, opts["size"], opts["size"])

dist = torch.zeros(25, 3, 10, 10)

minima = -1
maxima = 1
norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap("bwr"))

for i in range(5):
    for j in range(5):
        x = objects[j, :, 0, 2, 2].repeat(10, 1, 1).reshape(-1, 1, 42, 42)
        y = objects[i, :, 0, 2, 2].repeat_interleave(10, 0).reshape(-1, 1, 42, 42)

        xy = torch.cat([x, y], dim=1)
        with torch.no_grad():
            codes = model.encoder2(xy)
        dist[i*5+j, 0] = codes.reshape(10, 10).flip([0])
        for r in range(10):
            for c in range(10):
                dist[i*5+j, :, r, c] = torch.tensor(mapper.to_rgba(dist[i*5+j, 0, r, c]), dtype=torch.float)[:3]

dist = dist.repeat_interleave(6, 2)
dist = dist.repeat_interleave(6, 3)
torchvision.utils.save_image(dist, "comparisons.png", nrow=5, padding=10, pad_value=1.0)
