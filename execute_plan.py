import sys
import argparse
import rospy
import numpy as np
from scipy.spatial.transform import Rotation
from simtools.rosutils import RosNode


parser = argparse.ArgumentParser("Parse plan.")
parser.add_argument("-p", help="plan file", type=str, required=True)
parser.add_argument("-uri", help="master uri. optional", type=str, default="http://localhost:11311")
args = parser.parse_args()


node = RosNode("execute_plan", args.uri, wait_time=2.5)

file = open(args.p, "r")
lines = file.readlines()
N = int(lines[0])
objNames = []
objLocs = []
for i in range(N):
    name, x, y = lines[i+1].split()
    objNames.append(name)
    objLocs.append([float(x), float(y)])

sizes = [0.1, 0.18, 0.1]
base_level = 0.7
for p in lines[N+1:]:
    _, target, base = p.split()
    base_idx = objNames.index(base)
    target_idx = objNames.index(target)
    base_loc = objLocs[base_idx]
    target_loc = objLocs[target_idx]
    node.move(target_loc+[1.0, 0., 0., 0., 1.])
    node.move(target_loc+[0.7+0.75*sizes[target_idx], 0., 0., 0., 1.])
    node.handGraspPose()
    base_level += sizes[base_idx] + sizes[target_idx]
    node.move(target_loc+[base_level+0.1, 0., 0., 0., 1.])
    node.move(base_loc+[base_level+0.1, 0., 0., 0., 1.])

    node.handOpenPose()
    objLocs[target_idx] = objLocs[base_idx]
