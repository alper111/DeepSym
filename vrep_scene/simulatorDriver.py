import math
import sys
import signal
import random
import matplotlib.pyplot as plt
import numpy as np
import rospy
import time
from shutil import copyfile
from std_msgs.msg import Int32
from std_msgs.msg import Bool
import os

def callback(data):
	global simulationState
	simulationState = data.data

def startSimulation():
	global start,rate,true
	for i in range(5):
		start.publish(true)
		rate.sleep()

def stopSimulation():
	global stop,rate,true
	stop.publish(true)
	rate.sleep()

def sigint_handler(signal, frame):
	print 'INTERRUPTED'
	stopSimulation()
	sys.exit(0)


true = Bool()
true.data = True
simulationState = -1

objects = ['sphere','cylinder','cylinder_side','hollow','cube']
#actions = ['push_front','push_side','push_top']
actions = ['push_front']
N = 10


start = rospy.Publisher('/startSimulation', Bool, queue_size=10)
stop = rospy.Publisher('/stopSimulation', Bool, queue_size=10)

rospy.init_node('UR10LeverUpExperiment')
rate = rospy.Rate(10)
rospy.Subscriber("/simulationState", Int32, callback)

signal.signal(signal.SIGINT, sigint_handler)
stopSimulation()

maxSize = 0.2
minSize = 0.1
scale = np.linspace(1,maxSize/minSize,N) # divide object sizes according to N, may need np.around after linspace (recommended)

x_range = np.linspace(-0.4, -1.15, 6)
y_range = np.linspace(-0.4, 0.4, 6)

#file = open(info.txt,"w")

for action in actions:
	for object in objects:
		for i in range(N):
			for x_idx in range(5):
				for y_idx in range(5):
					
					posx = random.uniform(x_range[x_idx], x_range[x_idx+1])
					posy = random.uniform(y_range[y_idx], y_range[y_idx+1])		
					
					pos_count = 5*y_idx + x_idx			

					print action, object, i, posx, posy, pos_count
					os.system('printf "'+object+'\n'+action+'\n'+str(i)+'\n'+str(scale[i])+ '\n'+str(posx)+ '\n'+str(posy)+ '\n'+str(pos_count)+ '\n" > info.txt')
					while(simulationState != 0):
						i=i
					startSimulation() #start sim
					while(simulationState != 2): #waits vrep to set itself simulation state = 2 (paused)
						i=i
					stopSimulation() #reset sim
