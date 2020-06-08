import argparse
import os
import yaml
import torch
import numpy as np
import utils

parser = argparse.ArgumentParser("Cluster effects.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
device = opts["device"]

sorted_idx = torch.load("data/sorted_effidx.pt")
effects = torch.load("data/effects_2.pt").to(device)
effects = effects.abs()
eff_mu = effects.mean(dim=0)
eff_std = effects.std(dim=0)
effects = (effects - eff_mu) / (eff_std + 1e-6)

K = 6
ok = False
while not ok:
    ok = True
    centroids, assigns, mse, _ = utils.kmeans(effects, k=K)
    centroids = centroids * (eff_std + 1e-6) + eff_mu
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
torch.save(assigns[sorted_idx].cpu(), os.path.join(opts["save"], "label.pt"))
np.save(os.path.join(opts["save"], "effect_names.npy"), effect_names)
