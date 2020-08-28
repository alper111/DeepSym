import os
import argparse
import time
import yaml
import torch
from models import EffectRegressor
import data
import utils

parser = argparse.ArgumentParser("Train effect prediction models.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
if not os.path.exists(opts["save"]):
    os.makedirs(opts["save"])
opts["time"] = time.asctime(time.localtime(time.time()))
file = open(os.path.join(opts["save"], "opts.yaml"), "w")
yaml.dump(opts, file)
file.close()
print(yaml.dump(opts))

device = torch.device(opts["device"])

# load the first level data
transform = data.default_transform(size=42, affine=True, mean=0.279, std=0.0094)
# transform = None
trainset = data.ImageFirstLevel(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size"], shuffle=True)

model = EffectRegressor(opts)
if opts["load"] is not None:
    model.load(opts["load"], ext="")
model.print_model()
model.train(opts["epoch"], loader)

cd = opts["code1_dim"]
d = 2**cd
x = -torch.ones(3*d, 3+cd, device=opts["device"])
for i in range(3):
    for j in range(d):
        x[i*d+j, :cd] = torch.tensor(utils.decimal_to_binary(j, length=cd), dtype=torch.float)
x[:d, cd:] = torch.tensor([1., -1., -1.]).repeat(d, 1)
x[d:2*d, cd:] = torch.tensor([-1., 1., -1.]).repeat(d, 1)
x[2*d:, cd:] = torch.tensor([-1., -1., 1.]).repeat(d, 1)

with torch.no_grad():
    print(model.decoder(x))
