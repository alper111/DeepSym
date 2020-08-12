import os
import argparse
import torch
import yaml
from models import EffectRegressorMLP
import data


parser = argparse.ArgumentParser("Save categories.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
device = torch.device(opts["device"])

model = EffectRegressorMLP(opts)
model.load(opts["save"], "_best", 1)
model.load(opts["save"], "_best", 2)
model.encoder1.eval()
model.encoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
relations = torch.load("data/relations.pt")
X = torch.load("data/objectsZ.pt")
B, C, H, W = X.shape
Y = torch.empty(B*C, 1, opts["size"], opts["size"])
X = X.reshape(B*C, 1, H, W)

for i in range(B*C):
    Y[i] = transform(X[i])

with torch.no_grad():
    category1 = model.encoder1(Y.to(device))
category1 = category1.int()
category2 = torch.empty((B*C)**2, 1, device=device, dtype=torch.int)
# we cannot batch process this.
for i in range(B*C):
    y_i = Y[i].repeat(B*C, 1, 1, 1)
    concat = torch.cat([y_i, Y], dim=1)
    with torch.no_grad():
        category2[i*B*C:(i+1)*B*C] = model.encoder2(concat.to(device))
    if (i+1) % 50 == 0:
        print("%d/%d" % (i+1, B*C))


left = category1.repeat_interleave(B*C, 0)
right = category1.repeat(B*C, 1)
category_all = torch.cat([left, right, category2], dim=-1)
torch.save(category_all.cpu(), os.path.join(opts["save"], "category.pt"))
