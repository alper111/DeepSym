import sys
import rospy
import rosutils
import numpy as np
from scipy.spatial.transform import Rotation

if len(sys.argv) > 1:
    master_uri = sys.argv[1]
else:
    master_uri = "http://localhost:11311"

node = rosutils.RosNode("exploration_push", master_uri)
node.stopSimulation()
rospy.sleep(1.0)
node.startSimulation()
rospy.sleep(1.0)
print("simulation started.")
node.initArmPose()
node.handOpenPose()
scales = np.linspace(1.0, 2.0, 10)
xrng = np.linspace(-0.4, -1.15, 6)
yrng = np.linspace(-0.4, 0.4, 6)
obs_prev = []
obs_next = []
pos_prev = []
pos_next = []
force_readings = []
action = []
for obj_i in range(4, 6):
    for scale in [1.0, 2.0]:
        for a in range(3):
            for x in [-0.66]:
                for y in [-0.11]:
                    size = scale * 0.1
                    loc = [x, y, 0.7+size/2]
                    node.generateObject(obj_i, scale, loc)
                    obs_prev.append(node.getDepthImage(8))
                    pos_prev.append(node.getObjectPosition())
                    # push top
                    if a == 0:
                        node.handPokePose()
                        node.move([x-0.05, y, 1.0, 0., 0., 0., 1.])
                        node.move([x-0.05, y, 0.77+size, 0., 0., 0., 1.])
                        force_readings.append(node.getFingerForce())
                        node.move([x-0.05, y, 1.0, 0., 0., 0., 1.])
                    # push front
                    elif a == 1:
                        node.handFistPose()
                        quat = Rotation.from_euler("z", 90, degrees=True).as_quat().tolist()
                        node.move([-0.4, y, 0.7+size/2] + quat)
                        node.move([-0.8, y, 0.7+size/2] + quat)
                        force_readings.append(node.getFingerForce())
                    # push side
                    elif a == 2:
                        node.handFistPose()
                        node.move([x, -0.4, 0.7+size/2, 0., 0., 0., 1.])
                        node.move([x, 0., 0.7+size/2, 0., 0., 0., 1.])
                        force_readings.append(node.getFingerForce())
                    node.initArmPose()
                    node.handOpenPose()
                    obs_next.append(node.getDepthImage(8))
                    pos_next.append(node.getObjectPosition())
                    action.append(a)
                    node.popObject()

np.save("obs_prev.npy", obs_prev)
np.save("obs_next.npy", obs_next)
np.save("pos_prev.npy", pos_prev)
np.save("pos_next.npy", pos_next)
np.save("force_readings.npy", force_readings)
np.save("actions.npy", action)
node.stopSimulation()

print("simulation stopped.")
