import sys
import rospy
import rosutils
import numpy as np
from scipy.spatial.transform import Rotation

if len(sys.argv) > 1:
    master_uri = sys.argv[1]
else:
    master_uri = "http://localhost:11311"

node = rosutils.RosNode("exploration_push", master_uri, wait_time=2.5)
node.stopSimulation()
rospy.sleep(1.0)
node.startSimulation()
rospy.sleep(1.0)
print("simulation started.")
node.initArmPose()
node.handOpenPose()
scales = np.linspace(1.0, 2.0, 10)
xrng = np.linspace(-0.45, -1.1, 5)[:4]
yrng = np.linspace(-0.35, 0.35, 5)[:4]
obs_prev = []
obs_next = []
pos_prev = []
pos_next = []
force_readings = []
action = []
for obj_i in range(1, 6):
    for scale in scales:
        for a in range(3):
            for x in xrng:
                for y in yrng:
                    print(obj_i, scale, a, x, y)
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
                        quat = [0., 0., 0., 1.]
                        node.move([x+0.17, y, 1.0] + quat)
                        node.move([x+0.17, y, 0.70+size/2] + quat)
                        node.move([x-0.05, y, 0.70+size/2] + quat)
                        force_readings.append(node.getFingerForce())
                    # push side
                    elif a == 2:
                        node.handFistPose()
                        quat = Rotation.from_euler("z", 270, degrees=True).as_quat().tolist()
                        node.move([x, y-0.17, 1.0] + quat)
                        node.move([x, y-0.17, 0.70+size/2] + quat)
                        node.move([x, y+0.05, 0.70+size/2] + quat)
                        force_readings.append(node.getFingerForce())
                    elif a == 3:
                        if obj_i == 4:
                            quat = Rotation.from_euler("z", 90, degrees=True).as_quat().tolist()
                        else:
                            quat = [0., 0., 0., 1.]
                        node.handOpenPose()
                        node.move([x, y, 1.0]+quat)
                        node.move([x, y, 0.7+0.75*size]+quat)
                        node.handGraspPose()
                        node.move([x, y, 1.0]+quat)
                        node.move([x, y+0.3, 1.0]+quat)
                        node.handOpenPose()
                    node.initArmPose()
                    # node.handOpenPose()
                    node.wait(1.0)
                    obs_next.append(node.getDepthImage(8))
                    pos_next.append(node.getObjectPosition())
                    action.append(a)
                    # node.popObject()
                    node.stopSimulation()
                    rospy.sleep(1.0)
                    node.startSimulation()
                    rospy.sleep(1.0)

                    np.save("data/exploration_first/obs_prev.npy", obs_prev)
                    np.save("data/exploration_first/obs_next.npy", obs_next)
                    np.save("data/exploration_first/pos_prev.npy", pos_prev)
                    np.save("data/exploration_first/pos_next.npy", pos_next)
                    np.save("data/exploration_first/force_readings.npy", force_readings)
                    np.save("data/exploration_first/actions.npy", action)
node.stopSimulation()

print("simulation stopped.")
