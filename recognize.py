import os
import argparse
import yaml
import numpy as np
import torch
import models
import data
import utils


parser = argparse.ArgumentParser("Make plan.")
parser.add_argument("-opts", help="option file", type=str, required=True)
parser.add_argument("-img", help="image path", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
device = torch.device(opts["device"])

model = models.AffordanceModel(opts)
model.load(opts["save"], "_best", 1)
model.load(opts["save"], "_best", 2)
model.encoder1.eval()
model.encoder2.eval()
# Homogeneous transformation matrix
H = torch.load("H.pt")

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)

# transform image to tensor
file = open(args.img, "r")
lines = list(map(lambda x: x.rstrip(), file.readlines()))
lines = np.array(list(map(lambda x: x.split(" "), lines)), dtype=np.float)
x = torch.tensor(lines, dtype=torch.float)
x = x[8:120, 8:120]
objs, locs = utils.find_objects(x.clone(), 42)
objs = transform(objs)
objs = objs.to(device)

locs = torch.cat([locs.float(), torch.ones(locs.shape[0], 1, device=locs.device)], dim=1)
locs = torch.matmul(locs, H.T)

obj_infos = []
comparisons = []
with torch.no_grad():
    for i, obj in enumerate(objs):
        cat = model.encoder1(obj.unsqueeze(0).unsqueeze(0))
        print("Category: (%d %d), Location: (%.5f %.5f)" % (cat[0, 0], cat[0, 1], locs[i, 0].item(), locs[i, 1].item()))
        info = {}
        info["name"] = "obj{}".format(i)
        info["loc"] = (locs[i, 0].item(), locs[i, 1].item())
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

file_loc = os.path.join(opts["save"], "problem.pddl")
file_obj = os.path.join(opts["save"], "objects.txt")
if os.path.exists(file_loc):
    os.remove(file_loc)
if os.path.exists(file_obj):
    os.remove(file_obj)
print("(define (problem dom1) (:domain stack)", file=open(file_loc, "a"))
print("Objects:", file=open(file_obj, "a"))
object_str = "\t(:objects base"
init_str = "\t(:init  (stackloc base) (objtype2 base)\n"
for obj_i in obj_infos:
    print(obj_i["name"] + " : " + obj_i["type"] + " @ " + "%.5f" % obj_i["loc"][0] + " " + "%.5f" % obj_i["loc"][1], file=open(file_obj, "a"))
    object_str += " " + obj_i["name"]
    init_str += "\t\t(pickloc " + obj_i["name"] + ") (" + obj_i["type"] + " " + obj_i["name"] + ")\n"
object_str += ")"
for c_i in comparisons:
    init_str += "\t\t" + c_i + "\n"
for obj_i in obj_infos:
    init_str += "\t\t(relation0 %s base)\n" % obj_i["name"]
init_str += "\t\t(H0)\n"
init_str += "\t\t(S0)\n"
init_str += "\t)"
# TODO: take goal as parameter
goal_str = "\t(:goal (and (H3) (S0)))\n)"
print(object_str, file=open(file_loc, "a"))
print(init_str, file=open(file_loc, "a"))
print(goal_str, file=open(file_loc, "a"))
# need to handle base!
