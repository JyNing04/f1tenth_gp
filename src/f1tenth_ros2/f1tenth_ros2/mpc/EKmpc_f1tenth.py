#!/usr/bin/env python3
import math
import rclpy
import rclpy.subscription
import rclpy.publisher
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from f1tenth_msgs.msg import Waypoints
from geometry_msgs.msg import PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
import _pickle as pickle
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from f1tenth_ros2.models import Ekinematic
from f1tenth_ros2.models import dynamics
from f1tenth_ros2.params import f110
from f1tenth_ros2.mpc.nmpc import setupMPC
from planner import MPCPlanner
import time as tm

class EKMPCControllerNode(Node):
	def __init__(self):
		super().__init__('EKmpc_controller')
		qos_profile_in  = rclpy.qos.qos_profile_system_default
		qos_profile_in.depth = 1
		qos_profile_out = qos_profile_in
		map_index       = 0
		map_list        = ['Sepang', 'Shanghai', 'YasMarina']
		map_name        = map_list[map_index]
		raceline_type   = 'raceline_ED' # raceline_ED or centerline
		self.race_type  = raceline_type
		self.track_list = [map_name, '_'+raceline_type]
		self.start_pos  = [0.58955, 0.112556] 
		self.original   = False
		speed_profile   = False if self.original else True # True or False
		self.test_mode  = False
		self.path       = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/tracks') if not speed_profile \
			else os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/tracks_velocities')
		self.track_name = '{}_{}_test'.format(map_name, raceline_type) if self.test_mode else '{}_{}'.format(map_name, raceline_type)
		self.file_name  = os.path.join(self.path, self.track_name + '.csv') if not self.test_mode else os.path.join(self.path+'/test', self.track_name + '.csv') # waypoints file
		self.CTYPE      = 'GPMPC'
		self.currentX   = self.start_pos[0] # current position of car
		self.currentY   = self.start_pos[1]
		self.currentθ   = 0.0 # heading of the car
		self.lookahead  = 0.55 # Lookahead distance (m)
		self.lg         = 1.0 # Lookahead gain
		self.idx        = 0
		self.race_track = []
		self.track_size = 0
		self.vel_cmd    = AckermannDriveStamped()
		self.max_steer  = 1.0
		self.distF_avg  = 1.5
		# Default settings: Horizon, time step, weight matrices, etc 
		self.params   = f110.F110()
		self.model 	  = Ekinematic.Kinematic()
		time_step     = 0.04
		self.Horizon  = 11
		COST_Q        = np.diag([0.12, 0.12])
		COST_P        = np.diag([0.05, 0.0])
		COST_R        = np.diag([0.65, 3.65])
		COST_S        = np.diag([0.08, 0.0])
		track_width   = 2.4
		self.odom_in: rclpy.subscription.Subscription = self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_callback, qos_profile_in)
		self.scan_in: rclpy.subscription.Subscription = self.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile_in)
		self.vel_pub: rclpy.publisher.Publisher       = self.create_publisher(AckermannDriveStamped, 'drive', qos_profile_out)
		self.pos_pub: rclpy.publisher.Publisher       = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', qos_profile_out)
		self.dist_pub: rclpy.publisher.Publisher	  = self.create_publisher(Float32, 'Front_distance', 1)
		self.mpcpdt_pub: rclpy.publisher.Publisher    = self.create_publisher(MarkerArray, 'prediction_path_viz', 1)
		# Initialize vehicle states, inputs, dx/dt
		n_states      = self.model.n_states
		n_inputs      = self.model.n_inputs
		n_steps       = 72 * 60 * 3
		self.states   = np.zeros([n_states, n_steps+1])
		self.dstates  = np.zeros([n_states, n_steps+1])
		self.inputs   = np.zeros([n_inputs, n_steps])
		self.step     = 0
		self.time     = 0.
		self.time_f   = np.zeros(n_steps+1)
		self.save_data= True
		self.hstates  = np.zeros([n_states,self.Horizon+1])
		self.hstates2 = np.zeros([n_states,self.Horizon+1])
		# Define controller
		self.mpc_sol = setupMPC(self.Horizon, time_step, COST_Q, COST_P, COST_R, COST_S, self.params, self.model, track_width, map_name, proj_cons=True, track_cons=False)
		# Load GP models
		ωmodel_path  = "/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/gp_models/raceline/Sepang_raceline_ED-full-ω-RQ+LINEAR_gp.pickle"
		βmodel_path  = "/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/gp_models/raceline/Sepang_raceline_ED-full-β-RQ+LINEAR_gp.pickle"
		self.ωmodel, self.xωscaler, self.yωscaler = self.load_gpmodel(ωmodel_path)
		self.βmodel, self.xβscaler, self.yβscaler = self.load_gpmodel(βmodel_path)
		# Record running
		check_point   = 1000 #18
		self.rec_idx  = 0
		self.posXY    = np.zeros((2, check_point))
		self.refXY    = np.zeros((check_point, 2, (self.Horizon+1)))
		self.predXY   = np.zeros((check_point, 2, (self.Horizon+1)))
	
	def load_gpmodel(self, path):
		with open(path, 'rb') as f:
			(gpmodel, xscaler, yscaler) = pickle.load(f)
		return gpmodel, xscaler, yscaler

	def gp_predict(self, model, xscaler, yscaler):
		Xinput = np.hstack([
			self.states[2, self.step],   			   # 𝛿
			self.states[3, self.step],  			   # v 
			self.states[5, self.step],				   # ω
			self.states[6, self.step], 			  	   # β
			self.inputs[0, self.step], 				   # acc
			self.inputs[1, self.step], 				   # Δ𝛿
			]).reshape(1, -1)
		Xinput  = xscaler.transform(Xinput)
		Youtput = model.predict(Xinput)
		Youtput = yscaler.inverse_transform(Youtput)
		return Youtput

	def construct_path(self): 
		with open(self.file_name, 'r') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter =',')
			next(csv_reader)
			for waypoint in csv_reader:
				self.race_track.append(waypoint)
			# Force elements in racetrack to be float
			self.race_track = list(np.array(self.race_track, dtype='float'))
			self.track_size = len(self.race_track)
	
	def vehicle_reset(self):
		"""	
		x is a 7x1 state vector: [x, y, 𝛿, v, phi, ω, β]^T
		u is a 2x1 input vector: [acc, Δ𝛿]^T
		"""
		self.init_time = tm.time()
		initial_pose = PoseWithCovarianceStamped()
		x_init    = np.zeros(self.states.shape[0])
		x_init[0] = self.start_pos[0]
		x_init[1] = self.start_pos[1]
		x_init[2] = 0.0
		x_init[3] = 0.1
		x_init[4] = self.euler_from_quat(0, 0, 0.0343821, 0.999409)[2]
		self.states[:,0]   = x_init
		self.dstates[:, 0] = self.model.derivative_eqs(None, x_init, [0, 0])
		initial_pose.header.frame_id         = 'ego_racecar/base_link'
		initial_pose.pose.pose.position.x    = x_init[0]
		initial_pose.pose.pose.position.y    = x_init[1]
		initial_pose.pose.pose.position.z    = 0.0
		initial_pose.pose.pose.orientation.x = 0.0
		initial_pose.pose.pose.orientation.y = 0.0
		initial_pose.pose.pose.orientation.z = 0.0343821
		initial_pose.pose.pose.orientation.w = 0.999409
		self.pos_pub.publish(initial_pose)
		print('starting at ({:.3f},{:.3f})'.format(x_init[0], x_init[1]))

	def euler_from_quat(self, x, y, z, w):
		"""
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
		"""
		t0     = 2.0 * (w * x + y * z)
		t1     = 1.0 - 2.0 * (x * x + y * y)
		roll   = math.atan2(t0, t1)
	 
		t2     = 2.0 * (w * y - z * x)
		t2     = 1.0 if t2 > 1.0 else t2
		t2     = -1.0 if t2 < -1.0 else t2
		pitch  = math.asin(t2)
	 
		t3     = 2.0 * (w * z + x * y)
		t4     = 1.0 - 2.0 * (y * y + z * z)
		yaw    = math.atan2(t3, t4)
		return roll, pitch, yaw

	def scan_callback(self, msg:LaserScan):
		ang_range = 0.09 # 5 deg
		range_del = (ang_range / (msg.angle_max-msg.angle_min) * len(msg.ranges)) / 2
		mid_idx   = int(len(msg.ranges) / 2)
		front_lef = int(mid_idx-range_del)
		front_rig = int(mid_idx+range_del)
		distF_arr = np.asarray(msg.ranges[front_lef:front_rig])
		distR_arr = np.asarray(msg.ranges[front_lef-150:mid_idx-100])
		distL_arr = np.asarray(msg.ranges[mid_idx+100:front_rig+150])
		self.distF_avg = np.mean(distF_arr)
		self.distL_avg = np.mean(distL_arr)
		self.distR_avg = np.mean(distR_arr)
		dist_msg = Float32()
		dist_msg.data = float(self.distF_avg)
		self.dist_pub.publish(dist_msg)
		# print("Distnace -- Left: %.2f, Mid: %.2f, Right: %.2f"%(self.distL_avg, self.distF_avg, self.distR_avg) )

	def pose_callback(self, msg:Odometry):
		start = tm.time()
		n = 0
		while n < 800000:
			n += 1
		# Construct track waypoints at the beginning 
		if not self.race_track:
			self.construct_path()
			self.vehicle_reset()
		linear_x      = msg.twist.twist.linear.x
		linear_y      = msg.twist.twist.linear.y
		linear_z      = msg.twist.twist.linear.z
		speed         = LA.norm(np.array([linear_x, linear_y, linear_z]),2)
		self.currentX = msg.pose.pose.position.x
		self.currentY = msg.pose.pose.position.y
		self.currentθ = self.euler_from_quat(msg.pose.pose.orientation.x,
											 msg.pose.pose.orientation.y,
											 msg.pose.pose.orientation.z,
											 msg.pose.pose.orientation.w)[2]
		
		# save data after one lap
		end_point = [self.start_pos[0]-1., self.start_pos[1]]
		end_dist  = math.sqrt((self.currentX - end_point[0])**2 + (self.currentY - end_point[1])**2)
		end_thre  = 0.55
		if end_dist <= end_thre:
			self.end_time = tm.time()
			print("Finish line arrived. Laptime: %.3f second"%(self.end_time - self.init_time))
			states  = self.states.transpose()
			dstates = self.dstates.transpose()
			rows    = states[~np.all(states == 0, axis=1)].shape[0]
			inputs  = self.inputs.transpose()
			idx_d   = [1,2,3]
			states  = np.delete(states, idx_d, 0)
			dstates = np.delete(dstates, idx_d, 0)
			inputs  = np.delete(inputs, idx_d, 0)
			time    = np.insert(np.trim_zeros(self.time_f),0, 0.0)
			time    = time[len(idx_d):] - time[len(idx_d)]
			file_ns = 'f1tenth-EKIN-{}-{}.npz'
			path    = '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/mpc/raceline/' if 'raceline' in self.track_name \
						else '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/mpc/centerline/'
			if self.original:
				path = '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/mpc/original/'
			if self.save_data:
				np.savez(
						path + file_ns.format(self.CTYPE, self.track_name),
						states=states[:rows-len(idx_d), :],
						inputs=inputs[:rows-len(idx_d), :],
						posXY=self.posXY,
						refXY=self.refXY,
						predXY=self.predXY, 
						)
				print("Saving data to " + file_ns.format(self.CTYPE, self.track_name))
			# Kill the node
			self.vel_cmd.drive.speed = 0.0
			self.vel_pub.publish(self.vel_cmd)
			exit()

		"""	
		Update current states with measured data from sim
		x is a 7x1 state vector: [x, y, 𝛿, v, phi, ω, β]^T
		u is a 2x1 input vector: [acc, Δ𝛿]^T
		"""
		if math.sqrt((self.currentX - self.start_pos[0])**2 + (self.currentY - self.start_pos[1])**2) <= 1.55:
			self.vel_cmd.drive.speed = 3.5
			self.vel_cmd.drive.steering_angle = 0.0
			self.vel_pub.publish(self.vel_cmd)
		self.step                += 1
		self.states[0, self.step] = self.currentX
		self.states[1, self.step] = self.currentY
		self.states[3, self.step] = speed
		self.states[4, self.step] = self.currentθ
		x0    = self.states[:, self.step]
		uprev = self.inputs[:, self.step-1]
		# Call MPC planner
		planner = MPCPlanner(x0, self.path, self.track_list, N=self.Horizon)
		v_yaw_ref, dist_prj, xref, proj_idx = planner.projectXref()

		# Call MPC solver
		start_opt = tm.time()
		umpc, fval, xmpc = self.mpc_sol.solve(x0=x0, xref=xref[:2,:], uprev=uprev, v_yaw_ref=v_yaw_ref, dist_prj=dist_prj)
		end_opt   = tm.time()
		# print("Optm comupting time: {:.2f}, cost: {:.5f}".format(end_opt-start_opt, fval))

		self.inputs[0, self.step] = umpc[0,0] # Acceleration
		self.states[2, self.step] = umpc[1,0]
		self.inputs[1, self.step] = self.states[2, self.step] - self.states[2, self.step-1] # Steering angle
		# Update ω and β with numerical integration (Heun's method)
		end         = tm.time()
		h           = end-start
		self.time  += h
		self.states[-2:, self.step] = dynamics.Dynamic().odeintRK6(x0, h, self.inputs[:, self.step]) 
		# Call GP model for prediction
		ω_output = self.gp_predict(self.ωmodel, self.xωscaler, self.yωscaler)
		β_output = self.gp_predict(self.βmodel, self.xβscaler, self.yβscaler)
		
		# Filtering NaN in data
		self.states[:, self.step] = np.nan_to_num(self.states[:, self.step])
		self.inputs[:, self.step] = np.nan_to_num(self.inputs[:, self.step])
		self.time_f[self.step] = self.time
		dxdt_next = self.model.derivative_eqs(None, self.states[:, self.step], self.inputs[:,self.step])
		self.dstates[:, self.step] = dxdt_next
		# Update driving commands
		acc          = dxdt_next[3]
		steering_vel = dxdt_next[2]
		steering_ang = x0[2] + steering_vel
		heading      = x0[4] + dxdt_next[4]
		
		# Only use for racecar in simulator, prevent oscillation (high-pass filter)
		k_s      = 0.025
		k_v      = 2.5
		steer_th = 0.155
		# Avoid hitting the wall
		if self.distF_avg < 3.5:
			acc -= (self.distF_avg) * steer_th
			steering_ang = math.copysign(abs(steering_ang)+k_s, steering_ang) # if 'center' in self.race_type  else math.copysign(abs(steering_ang)+steer_th, steering_ang)
		steering_ang = steering_ang if abs(steering_ang) > k_s else 0.0

		# Publish moving commands
		velocity    = x0[3] + acc
		self.vel_cmd.drive.speed          = float(velocity) 
		self.vel_cmd.drive.steering_angle = steering_ang 
		self.vel_cmd.drive.acceleration   = acc
		self.vel_cmd.drive.steering_angle_velocity = steering_vel
		self.vel_cmd.header = msg.header
		self.vel_pub.publish(self.vel_cmd)
		# print("X: {:.4f}, Y: {:.4f}, Velocity: {:.4f}, Acceleration: {:.4f}, Steering: {:.4f}".format(self.states[0, self.step], self.states[1, self.step], velocity, acc, steering_ang))
		
		# update states 
		self.states[3, self.step]  = velocity
		self.states[4, self.step]  = heading
		self.states[5, self.step] += ω_output
		
		self.states[6, self.step] += β_output

		# Record states
		check_point =1 # 55 ~4 seconds
		if self.step % check_point == 0 or self.step == 1:
			if self.step == 1:
				self.posXY[:, self.rec_idx]  = self.start_pos
			else:
				self.posXY[:, self.rec_idx]  = self.states[:2, self.step]
			print('CHECKPOINT %d reached.'%self.rec_idx)
			self.refXY[self.rec_idx, :]  = xref[:2]
			self.predXY[self.rec_idx, :] = xmpc[:2]
			self.rec_idx += 1

		# MPC prediction visualization
		predict_path = MarkerArray()
		for i in range(xmpc.shape[1]):
			marker                    = Marker()
			marker.header.frame_id    = 'map'
			marker.ns                 = 'Predict-{}'.format(i)
			marker.type               = 2
			marker.action             = 0
			marker.id                 = i
			marker.pose.position.x    = xmpc[0][i]
			marker.pose.position.y    = xmpc[1][i]
			marker.pose.position.z    = 0.0
			marker.scale.x            = 0.15
			marker.scale.y            = 0.15
			marker.scale.z            = 0.15
			marker.color.r            = 0.3
			marker.color.b            = 0.0
			marker.color.g            = 0.1
			marker.color.a            = 1.0
			predict_path.markers.append(marker)
		self.mpcpdt_pub.publish(predict_path)
		
def main(args=None):
	rclpy.init(args=args)
	node = EKMPCControllerNode()
	try:
		rclpy.spin(node)
	except Exception as e:
		print("Error: %s" %e)
	rclpy.shutdown()

if __name__ == '__main__':
	main()