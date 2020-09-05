import sys
import time
import rospy
import rosutils
import numpy as np
from scipy.spatial.transform import Rotation

if len(sys.argv) > 1:
    master_uri = sys.argv[1]
else:
    master_uri = "http://localhost:11311"

node = rosutils.RosNode("exploration_stack", master_uri, wait_time=1.5)
node.stopSimulation()
rospy.sleep(1.0)
node.startSimulation()
rospy.sleep(1.0)
print("simulation started.")
node.initArmPose()
node.handOpenPose()
scales = np.linspace(1.0, 2.0, 10)
depth_prev = []
depth_next = []
rgb_prev = []
rgb_next = []

for obj_i in range(1, 6):
    for s_i in scales:
        for obj_j in range(6, 11):
            for s_j in scales:
                start = time.time()
                print(obj_i, s_i, obj_j, s_j)
                size_i = s_i * 0.1
                size_j = s_j * 0.1
                loc_i = [-0.75, -0.15, 0.7+size_i/2]
                loc_j = [-0.75, 0.16, 0.7+size_j/2]
                node.generateObject(obj_i, s_i, loc_i)
                node.generateObject(obj_j, s_j, loc_j)
                depth_prev.append(node.getDepthImage(8))
                rgb_prev.append(node.getRGBImage())
                if obj_j == 4:
                    quat = Rotation.from_euler("z", 90, degrees=True).as_quat().tolist()
                else:
                    quat = [0., 0., 0., 1.]
                node.handOpenPose()
                node.move([-0.75, 0.16, 1.0]+quat)
                if obj_j in [2, 3, 5]:
                    multiplier = 0.95
                else:
                    multiplier = 0.75
                node.move([-0.75, 0.16, 0.7+multiplier*size_j]+quat)
                node.handGraspPose()
                node.move([-0.75, 0.16, 0.7+size_i+size_j+0.05]+quat)
                node.move([-0.75, -0.15, 0.7+size_i+size_j+0.05]+quat)
                node.handOpenPose()
                node.initArmPose()
                node.wait(1.0)
                depth_next.append(node.getDepthImage(8))
                rgb_next.append(node.getRGBImage())
                node.stopSimulation()
                rospy.sleep(2)
                node.startSimulation()
                rospy.sleep(2)

                end = time.time()
                print(end-start)

np.save("data/exploration_second/depth_prev.npy", depth_prev)
np.save("data/exploration_second/depth_next.npy", depth_next)
np.save("data/exploration_second/rgb_prev.npy", rgb_prev)
np.save("data/exploration_second/rgb_next.npy", rgb_next)

node.stopSimulation()
print("simulation stopped.")
