import os
import argparse
import time
import yaml
import torch
from models import EffectRegressorConv
import data

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
# transform = data.default_transform(size=112, affine=True)
transform = None
trainset = data.ImageFirstLevel(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size"], shuffle=True)

model = EffectRegressorConv(opts)
if opts["load"] is not None:
    model.load(opts["load"], ext="")
model.print_model()
model.train(opts["epoch"], loader)
