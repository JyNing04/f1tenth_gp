"""
Used to create dynamic speed profile for eahc racetrack
"""

from doctest import testmod
import enum
from tkinter import YView
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import math
import csv
from scipy import signal


def downsample_manual(suffix, data, skip_id, TYPE, data_path):
	# data: npz file, consists of states, dstates, inputs
	# numpy array, size n x m
	# skip_id: int
	if skip_id == 2:
		downsample_type = 'halfsample'
	elif skip_id == 3:
		downsample_type = 'downsample'
	else:
		downsample_type = 'full'
	if TYPE == 'DYN':
		states  = data['states']
		dstates = data['dstates']
		inputs  = data['inputs']
		time    = data['time']
		np.savez(data_path + 'f1tenth-{}-{}-{}-{}'.format(downsample_type, TYPE, CTYPE, track_name) + suffix,
					time=time[::skip_id],
					states=states[::skip_id],
					dstates=dstates[::skip_id],
					inputs=inputs[::skip_id],)
	if TYPE == 'KIN':
		states  = data['states']
		inputs  = data['inputs']
		time    = data['time']
		np.savez(data_path + 'f1tenth-{}-{}-{}-{}'.format(downsample_type, TYPE, CTYPE, track_name) + suffix,
					time=time[::skip_id],
					states=states[::skip_id],
					inputs=inputs[::skip_id],)
	print("Saving data to: ", data_path + 'f1tenth-{}-{}-{}-{}'.format(downsample_type, TYPE, CTYPE, track_name) + suffix)
	

def plot_velocity(states):
	vel = states[:,3]
	X   = states[:,0]
	Y   = states[:,1]
	# Plot original velocity 
	plt.figure()
	plt.hexbin(X,Y,vel, gridsize=120, cmap='brg') # Heatmap
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Velocity Heatmap [%s]' %map[track_id])
	cb = plt.colorbar()
	cb.set_label('Velocity (m/s)')

	plt.figure()
	plt.plot(time[:n_steps], vel, label='V')
	plt.xlabel('Time [s]')
	plt.ylabel('Speed [m/s]')
	plt.grid()
	plt.legend()

def find_max_vel_region(states, target_vel):
	vel = states[:,3]
	max_v_idx = np.where(vel >= target_vel)[0]
	region = []
	sub_ls = []
	for i in range(len(max_v_idx)-1):
		if max_v_idx[i] == max_v_idx[i+1] - 1:
			sub_ls.append(max_v_idx[i])
		else:
			region.append(sub_ls)
			sub_ls = []
		if i == len(max_v_idx) - 2:
			sub_ls.append(max_v_idx[-1])
			region.append(sub_ls)
	return region

def cal_acc(t, d):
	a = 2 * d / t**2
	return a

def change_speed(states, region, test_mode):
	vel = states[:, 3]
	for idx, ele in enumerate(region):
		len_ele = len(ele)
		vel_ele = vel[ele]
		t_ele   = time[ele]
		total_t = t_ele[-1] - t_ele[0]
		total_d = t_ele * vel_ele
		dist    = total_d[-1] - total_d[0]
		acc_scale = 10
		if len_ele <= 25:
			acc_scale += 50
		mid_point = 0.5
		if test_mode:
			# print("test mode speed")
			acc_scale += 25
			mid_point = 0.35
		accel = cal_acc(total_t/2, dist/2) / acc_scale # scaled acceleration 
		# Modify velocity profile
		for j in range(int(len_ele)-1):
			# if idx >= 4:
			# Select middle point of straight line
			if j <= int(len_ele * mid_point):
				vel_ele[j+1] = vel_ele[j] + accel
			else:
				vel_ele[j] = vel_ele[j-1] - accel #*1.05
			# else:
			#     if j <= int(len_ele/5*3):
			#         vel_ele[j+1] = vel_ele[j] + accel
			#     else:
			#         vel_ele[j] = vel_ele[j-1] - accel #*1.05
		vel[ele] = vel_ele
	return vel

# Cosntruct waypoint data with velocity profile
def construct_path(file_name): 
	race_track = []
	with open(file_name, 'r') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter =',')
		next(csv_reader)
		for waypoint in csv_reader:
			race_track.append(waypoint)
		# Force elements in racetrack to be float
		race_track = list(np.array(race_track, dtype='float'))
		track_size = len(race_track)
		return race_track, track_size

def find_nearest_goal(race_track, curr_x, curr_y):
	ranges = []
	for idx in range(len(race_track)):
		dist_x    = math.pow(curr_x - race_track[idx][0],2)
		dist_y    = math.pow(curr_y - race_track[idx][1],2)
		eucl_dist = math.sqrt(dist_x + dist_y)
		ranges.append(eucl_dist)
	return(ranges.index(min(ranges)))

def plot_trackV(track_x, track_y, track_v, track_size):
	plt.figure()
	plt.hexbin(track_x,track_y,track_v, gridsize=120, cmap='brg') # Heatmap
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Track Velocity Heatmap')
	cb = plt.colorbar()
	cb.set_label('Velocity (m/s)')

	plt.figure()
	plt.plot(np.arange(track_size), np.flip(track_v), label='V')
	plt.xlabel('Data sample')
	plt.ylabel('Speed [m/s]')
	plt.title('Track Velocity Profile')
	plt.grid()
	plt.legend()

if __name__ == '__main__':
	CTYPE      = 'PP'
	ORIGINAL   = False
	test_mode  = True # Will be changed in speed changing and data saving
	Downsample = True
	data_ds    = True # If downsampled data ready
	# constant_v = False
	v_mod_flag = True
	csv_file   = True
	track_id   = 2
	skip_id    = 3
	map        = ['Sepang', 'Shanghai', 'YasMarina']
	path       = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/tracks')
	race_type  = 'centerline' # centerline or raceline_ED
	track_name = map[track_id]+'_'+race_type # if not test_mode else 'test_'+map[track_id]+ '_'+race_type
	file_name  = os.path.join(path, track_name + '.csv')
	suffix     = '.npz' if not test_mode else '_test.npz'
	dyn_fname = 'f1tenth-DYN-{}-{}.npz'
	kin_fname = 'f1tenth-KIN-{}-{}.npz'
	# Where data has been stored
	if ORIGINAL:
		data_path = '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/original/'
	elif race_type == 'centerline':
		data_path = '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/centerline/'
	else:
		data_path = '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/raceline/'
	# In test mode
	if test_mode:
		data_path += 'test/'
		dyn_fname  = 'f1tenth-DYN-{}-{}_test.npz'
		kin_fname  = 'f1tenth-KIN-{}-{}_test.npz'

	# Loading full size data
	data_dyn = np.load(data_path + dyn_fname.format(CTYPE, track_name))
	data_kin = np.load(data_path + kin_fname.format(CTYPE, track_name))
	# Assigne data to different categories (states, inputs, time)
	states  = data_dyn['states']
	dstates = data_dyn['dstates']
	statesk = data_kin['states']
	inputs  = data_dyn['inputs']
	time    = data_dyn['time']
	n_steps = time.shape[0]
	print("Data size: ", data_kin['states'].shape[0])
	DS_list    = ['-full', '-halfsample', '-downsample']
	if Downsample:
		downsample_manual(suffix, data_dyn, skip_id, 'DYN', data_path)
		downsample_manual(suffix, data_kin, skip_id, 'KIN', data_path)
		data_ds = True
		v_mod_flag = False
		csv_file   = False
	# Load downsampled data 
	# states: ['x', 'y', '𝛿', 'v', 'phi', 'ω', 'β'] 
	# inputs: [acc, Δ𝛿]
	if data_ds:
		data_dyn_ds = np.load(data_path + 'f1tenth' + DS_list[skip_id-1] + '-DYN-{}-{}'.format(CTYPE, track_name) + suffix)
		data_kin_ds = np.load(data_path + 'f1tenth' + DS_list[skip_id-1] + '-KIN-{}-{}'.format(CTYPE, track_name) + suffix)
		# if skip_id == 1:
		#     data_dyn_ds = np.load(data_path + 'f1tenth-full-DYN-{}-{}.npz'.format(CTYPE, track_name))
		#     data_kin_ds = np.load(data_path + 'f1tenth-full-KIN-{}-{}.npz'.format(CTYPE, track_name))
		# if skip_id == 2:
		#     data_dyn_ds = np.load(data_path + 'f1tenth-halfsample-DYN-{}-{}.npz'.format(CTYPE, track_name))
		#     data_kin_ds = np.load(data_path + 'f1tenth-halfsample-KIN-{}-{}.npz'.format(CTYPE, track_name))
		# if skip_id == 3:
		#     data_dyn_ds = np.load(data_path + 'f1tenth-downsample-DYN-{}-{}.npz'.format(CTYPE, track_name))
		#     data_kin_ds = np.load(data_path + 'f1tenth-downsample-KIN-{}-{}.npz'.format(CTYPE, track_name))
		states      = data_dyn_ds['states']
		dstates     = data_dyn_ds['dstates']
		statesk     = data_kin_ds['states']
		inputs      = data_dyn_ds['inputs']
		time        = data_dyn_ds['time']
		n_steps     = time.shape[0]
		print("Data size: ", n_steps)
		# v_mod_flag  = False
		# csv_file    = False
	
	if v_mod_flag:
		# Find max velocity XY locations 
		target_v = 7.499999
		region   = find_max_vel_region(states, target_v)
		X        = states[:, 0]
		Y        = states[:, 1]
		# Visualize track regions with max speed
		# del region[4]
		# print(len(region[2]))
		# del region[2]
		plt.figure()
		for ele in region:
			plt.plot(X[ele], Y[ele], 'ro-')
			plt.plot(X, Y, 'g-')
			plt.xlabel('x')
			plt.ylabel('y')
			plt.title('Max Speed Regions: map {}'.format(map[track_id]))
		# Change speed with constant acceleration x = x0 + v0*t + 1/2*a*t^2
		vel = change_speed(states, region, test_mode=True)

		# Assign speed to track csv file
	if csv_file:
		test_mode = True
		race_track, track_size = construct_path(file_name)
		track_x = np.asarray(race_track)[:,0]
		track_y = np.asarray(race_track)[:,1]
		track_v = np.zeros_like(track_x)
		min_vel = 5.0
		for i in range(len(X)):
			if i <= 2:
				track_v[i] = min_vel
			min_idx = find_nearest_goal(race_track, X[i], Y[i]) % track_size
			# print("track_x: %.2f, current_x: %.2f"%(track_x[min_idx], X[i]))
			track_v[min_idx] = vel[i]
		# Ensure no zero speed exists, and set the lower bound
		for i, ele in enumerate(track_v):
			if ele <= 1.0:
				track_v[i] = track_v[i-1]
			elif ele <= 5.0:
				track_v[i] = min_vel
			# print("X: %.2f, Y: %.2f, Vel: %.2f" % (track_x[i], track_y[i], track_v[i]))

		# Save waypoint data
		print("Saving data.....")
		track_dict = {'X': track_x, 'Y': track_y, 'velocity':track_v}
		if test_mode:
			df = pd.DataFrame(track_dict).to_csv('/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/tracks_velocities/test/{}_test.csv'.format(track_name),index=False)
		else:
			df = pd.DataFrame(track_dict).to_csv('/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/tracks_velocities/{}.csv'.format(track_name),index=False)
		
		# Plot track velocity 
		plot_trackV(track_x, track_y, track_v, track_size)
	# Plot velocity all over the track
	plot_velocity(states)

	plt.show()