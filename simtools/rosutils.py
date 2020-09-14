import os
import rospy
import numpy as np
from std_msgs.msg import Bool, Float32, Float32MultiArray
from geometry_msgs.msg import Pose


class RosNode:
    def __init__(self, name, rosmaster, wait_time=2.0):
        os.environ["ROS_MASTER_URI"] = rosmaster
        self._node = rospy.init_node(name)
        self._start_pub = rospy.Publisher("/startSimulation", Bool, queue_size=10)
        self._stop_pub = rospy.Publisher("/stopSimulation", Bool, queue_size=10)
        self._pose_pub = rospy.Publisher("/setPose", Pose, queue_size=10)
        self._hand_pub = rospy.Publisher("/setHand", Float32MultiArray, queue_size=10)
        self._grasp_pub = rospy.Publisher("/setClosing", Float32, queue_size=10)
        self._genobj_pub = rospy.Publisher("/genObject", Float32MultiArray, queue_size=10)
        self._popobj_pub = rospy.Publisher("/popObject", Bool, queue_size=10)
        self._wait_time = wait_time
        rospy.sleep(1.0)

    def startSimulation(self):
        msg = Bool()
        msg.data = True
        self._start_pub.publish(msg)

    def stopSimulation(self):
        msg = Bool()
        msg.data = True
        self._stop_pub.publish(msg)

    def move(self, pose):
        msg = Pose()
        msg.position.x = pose[0]
        msg.position.y = pose[1]
        msg.position.z = pose[2]
        msg.orientation.x = pose[3]
        msg.orientation.y = pose[4]
        msg.orientation.z = pose[5]
        msg.orientation.w = pose[6]
        self._pose_pub.publish(msg)
        self.wait()

    def getPose(self):
        msg = rospy.wait_for_message("/getPose", Pose, timeout=1.0)
        return [msg.position.x, msg.position.y, msg.position.z,
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

    def initArmPose(self):
        self.move([-0.2, -0.1139, 1.1856, 0, 0, 0, 1])

    def handGraspPose(self):
        self._command_hand([-90, -90, 180, 180, 180, 60, 60, 60])

    def handOpenPose(self):
        self._command_hand([-90, -90, 0, 0, 0, 45, 45, 45])

    def handPokePose(self):
        # self._command_hand([-90, -90, 90, 90, 90, 90, -11.5, 90])
        self._command_hand([-90, -90, 90, 90, 90, 90, -5.0, 90])

    def handFistPose(self):
        self._command_hand([-90, -90, 85, 85, 85, 90, 90, 90])

    def handOpenSimple(self):
        self._grasp_command(0.2)

    def handGraspSimple(self):
        self._grasp_command(-0.2)

    def getHandPose(self):
        msg = rospy.wait_for_message("/getHand", Float32MultiArray)
        return list(msg.data)

    def generateObject(self, objType, scale, loc):
        msg = Float32MultiArray()
        msg.data = [objType, scale, loc[0], loc[1], loc[2]]
        self._genobj_pub.publish(msg)
        self.wait(0.1)

    def popObject(self):
        msg = Bool()
        msg.data = True
        self._popobj_pub.publish(msg)
        self.wait(0.1)

    def getObjectPosition(self):
        data = rospy.wait_for_message("/getObjectPosition", Float32MultiArray).data
        return list(data)

    def wait(self, seconds=None):
        if seconds is None:
            seconds = self._wait_time
        start_time = rospy.wait_for_message("/simulationTime", Float32).data
        end_time = rospy.wait_for_message("/simulationTime", Float32).data
        while (end_time - start_time) < seconds:
            end_time = rospy.wait_for_message("/simulationTime", Float32).data

    def getDepthImage(self, margin):
        data = rospy.wait_for_message("/kinectDepth", Float32MultiArray).data
        data = np.array(data, dtype=np.float32).reshape(128, 128)
        return data[margin:(128-margin), margin:(128-margin)]

    def getRGBImage(self, margin=0):
        data = rospy.wait_for_message("/kinectRGB", Float32MultiArray).data
        data = np.array(data, dtype=np.float).reshape(480, 640, 3)
        return data[margin:(480-margin), margin:(640-margin)]

    def getFingerForce(self):
        data = rospy.wait_for_message("/getFingerForce", Float32MultiArray).data
        return list(data)

    def _command_hand(self, position):
        msg = Float32MultiArray()
        msg.data = np.radians(position)
        self._hand_pub.publish(msg)
        self.wait()

    def _grasp_command(self, x):
        msg = Float32()
        msg.data = x
        self._grasp_pub.publish(msg)
        self.wait()
