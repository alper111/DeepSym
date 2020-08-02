import rospy
import rosutils
import numpy as np

node = rosutils.RosNode("exploration_push", "http://ahmetoglu.local:11311")
node.stopSimulation()
rospy.sleep(1.0)
node.startSimulation()
rospy.sleep(1.0)
print("simulation started.")
node.handOpenPose()
x = []
y = []
for i in range(1, 6):
    print("Object %d" % i)
    scale = 2.0
    loc = [-0.775, 0., 0.7+(scale*0.1)/2]
    node.generateObject(i, scale, loc)
    pick_loc = loc.copy()
    if (i == 1):
        pad = -0.04
    elif (i == 4):
        pad = -0.08
    else:
        pad = -0.02
    pick_loc[2] = 0.7 + (scale*0.1) + pad
    node.move(pick_loc+[0, 0, 0, 1])
    node.handGraspPose()
    currentPose = np.degrees(node.getHandPose())
    currentPose[2:5] += 10
    currentPose[5:] += 10
    node._command_hand(currentPose)

    drop_loc = loc.copy()
    drop_loc[2] = 1.0
    node.move(drop_loc+[0, 0, 0, 1])
    drop_loc[1] = 0.5
    node.move(drop_loc+[0, 0, 0, 1])
    node.handOpenPose()
    node.popObject()

node.stopSimulation()

print("simulation stopped.")
