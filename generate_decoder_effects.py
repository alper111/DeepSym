import argparse
import os
import torch
import yaml
import data
import utils
from models import EffectRegressorMLP

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
args = parser.parse_args()

file_loc = os.path.join(args.ckpt, "opts.yaml")
opts = yaml.safe_load(open(file_loc, "r"))
opts["device"] = "cpu"

model = EffectRegressorMLP(opts)
model.load(args.ckpt, "_best", 2)
model.encoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.SecondLevelDataset(transform=transform)

model.decoder2.eval()
effects = []
with torch.no_grad():
    for i in range(32):
        x = torch.tensor(utils.decimal_to_binary(i, 5), dtype=torch.float)
        effects.append(model.decoder2(x)*(trainset.eff_std+1e-6)+trainset.eff_mu)

effects = torch.stack(effects)
centroids, assigns, mse, it = utils.kmeans(effects, k=6)
print(centroids)
