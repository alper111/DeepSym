import math
import sys
import signal
import random
import numpy as np
import rospy
import time
from shutil import copyfile
from std_msgs.msg import Int32
from std_msgs.msg import Bool
import os
from random import choice


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

with open("/home/serkanbugur/Workspace/affordance-learning/cnn-2.0/plan.txt", "w") as myfile:
	myfile.write('not yet\n')

if os.path.exists("test.txt"):
	os.remove("test.txt")

start = rospy.Publisher('/startSimulation', Bool, queue_size=10)
stop = rospy.Publisher('/stopSimulation', Bool, queue_size=10)

rospy.init_node('UR10LeverUpExperiment')
rate = rospy.Rate(10)
rospy.Subscriber("/simulationState", Int32, callback)

signal.signal(signal.SIGINT, sigint_handler)
stopSimulation()

#objects = ['sphere','cylinder','cylinder_side','hollow','cube']
objects = ['cylinder','hollow','cube']
#actions = ['push_front','push_side','push_top']
N = 10
obj_count = 4

maxSize = 0.2
minSize = 0.1
scale = np.linspace(1,maxSize/minSize,N) # divide object sizes according to N, may need np.around after linspace (recommended)

#x_range = np.linspace(-1, -0.5, 3)
#y_range = np.linspace(-0.3, 0.3, 4)

x_range = np.array([-1, -0.75, -0.5])
y_range = np.array([-0.3, -0.1, 0.1, 0.3])

selected_positions = []

with open("test.txt", "a") as myfile:
    myfile.write(str(obj_count) + '\n')


for i in range(obj_count):
	obj = objects[random.randint(0,2)]
	if obj == 'sphere':
		obj_scale = scale[random.randint(0,3)]
	
	else:
		obj_scale = scale[random.randint(0,6)]	

	rand_position = choice([i for i in range(0,6) if i not in selected_positions])
	print(rand_position)
	selected_positions.append(rand_position)

	if rand_position == 0:
		#posx = random.uniform(x_range[0], x_range[1])
		posx = x_range[0]
		posy = random.uniform(y_range[0], y_range[1])
	elif rand_position == 1:
		#posx = random.uniform(x_range[0], x_range[1])
		posx = x_range[2]
		posy = random.uniform(y_range[1], y_range[2])
	elif rand_position == 2:
		#posx = random.uniform(x_range[0], x_range[1])
		posx = x_range[1]
		posy = random.uniform(y_range[2], y_range[3])
	elif rand_position == 3:
		#posx = random.uniform(x_range[1], x_range[2])
		posx = x_range[2]
		posy = random.uniform(y_range[0], y_range[1])
	elif rand_position == 4:
		#posx = random.uniform(x_range[1], x_range[2])
		posx = x_range[1]
		posy = random.uniform(y_range[1], y_range[2])
	elif rand_position == 5:
		#posx = random.uniform(x_range[1], x_range[2])
		posx = x_range[0]
		posy = random.uniform(y_range[2], y_range[3])
	with open("test.txt", "a") as myfile:
	#os.system('printf "'+str(obj_count)+'\n'+obj+' '+str(obj_scale)+'\n'+str(posx)+ ' '+str(posy)+ '\n" > put_random_objects_table.txt')
		myfile.write(obj+' '+str(obj_scale)+' '+str(posx)+ ' '+str(posy) + '\n')

while(simulationState != 0):
	i=i
startSimulation() #start sim
#while(simulationState != 2): #waits vrep to set itself simulation state = 2 (paused)
#	i=i
#stopSimulation() #reset sim





	


