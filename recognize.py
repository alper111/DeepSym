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
opts["device"] = torch.device("cpu")
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
objs, locs = utils.find_objects(x.clone(), 42)
objs = transform(objs)
obj_infos = []
comparisons = []
with torch.no_grad():
    for i, obj in enumerate(objs):
        cat = model.encoder1(obj.unsqueeze(0).unsqueeze(0))
        print("Category: (%d %d), Location: (%d %d)" % (cat[0, 0], cat[0, 1], locs[i, 0], locs[i, 1]))
        info = {}
        info["name"] = "obj{}".format(i)
        info["loc"] = (int(locs[i, 0]), int(locs[i, 1]))
        info["type"] = "objtype{}".format(utils.binary_to_decimal([int(cat[0, 0]), int(cat[0, 1])]))
        obj_infos.append(info)
        for j in range(i+1, len(objs)):
            rel = model.encoder2(torch.stack([obj, objs[j]]).unsqueeze(0))[0, 0]
            if rel == -1:
                comparisons.append("(relation0 obj%d obj%d)" % (i, j))
            else:
                comparisons.append("(relation0 obj%d obj%d)" % (j, i))
print(obj_infos)
print(comparisons)
