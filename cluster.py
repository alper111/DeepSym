import argparse
import os
import yaml
import torch
import numpy as np
import utils
import data
from models import EffectRegressorMLP

parser = argparse.ArgumentParser("Cluster effects.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
opts["device"] = "cpu"
device = opts["device"]

model = EffectRegressorMLP(opts)
model.load(opts["save"], "_best", 2)
model.encoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.PairedObjectData(transform=transform)

K = 6
ok = False
while not ok:
    ok = True
    centroids, assigns, mse, _ = utils.kmeans(trainset.effect, k=K)
    print(mse)
    centroids = centroids * (trainset.eff_std + 1e-6) + trainset.eff_mu
    effect_names = []
    for i, c_i in enumerate(centroids):
        print("Centroid %d: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f" %
              (i, c_i[0], c_i[1], c_i[2], c_i[3], c_i[4], c_i[5]))

    for i, c_i in enumerate(centroids):
        print("Centroid %d: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f" %
              (i, c_i[0], c_i[1], c_i[2], c_i[3], c_i[4], c_i[5]))
        print("What is this effect?")
        print(">>>", end="")
        name = input()
        if name == "reset":
            ok = False
            break
        effect_names.append(name)
effect_names = np.array(effect_names)
print("Effect names are:")
print(effect_names)
torch.save(assigns.cpu(), os.path.join(opts["save"], "label.pt"))
np.save(os.path.join(opts["save"], "effect_names.npy"), effect_names)
