import argparse
import os
import torch
import yaml
import data
from models import EffectRegressorMLP
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
trainset = data.ImageFirstLevel(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=2400, shuffle=False)
objects = iter(loader).next()["observation"]
objects = objects.reshape(5, 10, 3, 4, 4, opts["size"], opts["size"])

obj_list = ["Sphere", "Cube", "Cylinder", "Rot. cylinder", "Box"]
idx_1 = 0
idx_2 = 4

x = objects[idx_1, :, 0].repeat(10, 1, 1, 1, 1).reshape(-1, 1, 42, 42)
y = objects[idx_2, :, 0].repeat_interleave(10, 0).reshape(-1, 1, 42, 42)

xy = torch.cat([x, y], dim=1)
with torch.no_grad():
    codes = model.encoder2(xy)
codes = codes.reshape(10, 10, 4, 4)
plt.imshow(codes.mean([2, 3]).flip([0]), vmin=-1, vmax=1)
plt.xticks(list(range(10)), list(range(10, 20)))
plt.yticks(list(range(10)), list(reversed(range(10, 20))))
plt.xlabel("%s size (cm)" % obj_list[idx_1])
plt.ylabel("%s size (cm)" % obj_list[idx_2])
pp = PdfPages("out/%s-%s.pdf" % (obj_list[idx_1], obj_list[idx_2]))
pp.savefig()
pp.close()
