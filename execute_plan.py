import argparse
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
objSizes = []
for i in range(N):
    name, x, y, size = lines[i+1].split()
    objNames.append(name)
    objLocs.append([float(x), float(y)])
    objSizes.append(float(size))

print("Plan success probability: %.2f" % float(lines[N+1].split(":")[1]))
if lines[N+1] == "not found.":
    print("Cannot find a plan which satisfies the objective.")
    exit()

base_level = 0.7
for p in lines[N+2:]:
    node.initArmPose()
    _, base, target = p.split()
    base_idx = objNames.index(base)
    target_idx = objNames.index(target)
    base_loc = objLocs[base_idx]
    target_loc = objLocs[target_idx]
    node.move(target_loc+[1.0, 0., 0., 0., 1.])
    node.move(target_loc+[0.7+0.85*objSizes[target_idx], 0., 0., 0., 1.])
    node.handGraspPose()
    base_level += objSizes[base_idx]
    node.move(target_loc+[base_level+objSizes[target_idx]+0.05, 0., 0., 0., 1.])
    node.move(base_loc+[base_level+objSizes[target_idx]+0.05, 0., 0., 0., 1.])

    node.handOpenPose()
    objLocs[target_idx] = objLocs[base_idx]
node.initArmPose()
