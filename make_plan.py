import os
import argparse
import yaml
import numpy as np
import torch
import models
import data
import utils


parser = argparse.ArgumentParser("Make plan.")
parser.add_argument("-ckpt", help="model path", type=str, required=True)
args = parser.parse_args()

file_loc = os.path.join(args.ckpt, "opts.yaml")
opts = yaml.safe_load(open(file_loc, "r"))
device = torch.device(opts["device"])

model = models.AffordanceModel(opts)
model.load(args.ckpt, "_best", 1)
model.load(args.ckpt, "_best", 2)
model.encoder1.eval()
model.encoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)

obj_names = list(filter(lambda x: x[-3:] == "txt", os.listdir("data/depth3")))
file = open("data/depth3/"+np.random.choice(obj_names), "r")
lines = list(map(lambda x: x.rstrip(), file.readlines()))
lines = np.array(list(map(lambda x: x.split(" "), lines)), dtype=np.float)
x = torch.tensor(lines, dtype=torch.float, device=device)
x = x[8:120, 8:120]
objs, locs = utils.find_objects(x.clone(), 40)
with torch.no_grad():
    for i, obj in enumerate(objs):
        obj = obj.unsqueeze(0).unsqueeze(0)
        cat = model.encoder1(obj)
        print("Category: (%d %d), Location: (%d %d)" % (cat[0, 0], cat[0, 1], locs[i, 0], locs[i, 1]))