import os
import argparse
import rospy
import yaml
import numpy as np
import torch
from models import EffectRegressorMLP
import data
import utils
from simtools.rosutils import RosNode


parser = argparse.ArgumentParser("Make plan.")
parser.add_argument("-opts", help="option file", type=str, required=True)
parser.add_argument("-goal", help="goal state", type=str, default="(H3) (S0)")
parser.add_argument("-uri", help="master uri", type=str, default="http://localhost:11311")
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
device = torch.device(opts["device"])

node = RosNode("recognize_scene", args.uri)
node.stopSimulation()
rospy.sleep(1.0)
node.startSimulation()
rospy.sleep(1.0)

model = EffectRegressorMLP(opts)
model.load(opts["save"], "_best", 1)
model.load(opts["save"], "_best", 2)
model.encoder1.eval()
model.encoder2.eval()
# Homogeneous transformation matrix
H = torch.load("H.pt")

# GENERATE A RANDOM SCENE
NUM_OBJECTS = 5
objTypes = np.random.randint(1, 6, (NUM_OBJECTS, ))
objSizes = np.random.uniform(1.0, 2, (5, )).tolist()
locations = np.array([
    [-0.69, -0.09],
    [-0.9, -0.35],
    [-0.45, 0.175],
    [-0.45, -0.35],
    [-0.9, 0.175]
])
locations = locations[np.random.permutation(5)]
locations = locations[:NUM_OBJECTS].tolist()

for i in range(NUM_OBJECTS):
    node.generateObject(objTypes[i], objSizes[i], locations[i]+[objSizes[i]*0.05+0.7])
rospy.sleep(1.0)
locations = torch.tensor(locations, dtype=torch.float)

x = torch.tensor(node.getDepthImage(8), dtype=torch.float)
objs, locs, _ = utils.find_objects(x, opts["size"])

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
for i, o in enumerate(objs):
    objs[i] = transform(o)[0]
objs = objs.to(device)

locs = torch.cat([locs.float(), torch.ones(locs.shape[0], 1, device=locs.device)], dim=1)
locs = torch.matmul(locs, H.T)
locs = locs / locs[:, 2].reshape(-1, 1)

_, indices = torch.cdist(locs[:, :2], locations).min(dim=1)
obj_infos = []
comparisons = []
with torch.no_grad():
    for i, obj in enumerate(objs):
        cat = model.encoder1(obj.unsqueeze(0).unsqueeze(0))
        # TODO: this uses true location and size.
        print("Category: (%d %d), Location: (%.5f %.5f)" % (cat[0, 0], cat[0, 1], locations[indices[i], 0], locations[indices[i], 1]))
        info = {}
        info["name"] = "O{}".format(i+1)
        info["loc"] = (locations[indices[i], 0].item(), locations[indices[i], 1].item())
        info["size"] = objSizes[indices[i]]*0.1
        info["type"] = "objtype{}".format(utils.binary_to_decimal([int(cat[0, 0]), int(cat[0, 1])]))

        obj_infos.append(info)
        for j in range(len(objs)):
            if i != j:
                rel = model.encoder2(torch.stack([obj, objs[j]]).unsqueeze(0))[0, 0]
                if rel == -1:
                    comparisons.append("(relation0 O%d O%d)" % (i+1, j+1))
                else:
                    comparisons.append("(relation1 O%d O%d)" % (i+1, j+1))
print(obj_infos)
print(comparisons)

file_loc = os.path.join(opts["save"], "problem.pddl")
file_obj = os.path.join(opts["save"], "objects.txt")
if os.path.exists(file_loc):
    os.remove(file_loc)
if os.path.exists(file_obj):
    os.remove(file_obj)
print("(define (problem dom1) (:domain stack)", file=open(file_loc, "a"))
print(str(len(obj_infos)), file=open(file_obj, "a"))
object_str = "\t(:objects"
init_str = "\t(:init\n"
for obj_i in obj_infos:
    print("%s %.5f %.5f %.5f" % (obj_i["name"], obj_i["loc"][0], obj_i["loc"][1], obj_i["size"]), file=open(file_obj, "a"))
    object_str += " " + obj_i["name"]
    init_str += "\t\t(pickloc " + obj_i["name"] + ") (" + obj_i["type"] + " " + obj_i["name"] + ")\n"
object_str += ")"
for c_i in comparisons:
    init_str += "\t\t" + c_i + "\n"
init_str += "\t\t(H0)\n"
init_str += "\t\t(S0)\n"
init_str += "\t)"

goal_str = "\t(:goal (and %s (not (stacked)) (not (inserted))))\n)" % args.goal
print(object_str, file=open(file_loc, "a"))
print(init_str, file=open(file_loc, "a"))
print(goal_str, file=open(file_loc, "a"))
