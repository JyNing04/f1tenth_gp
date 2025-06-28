#!/usr/bin/env python3
import math
import rclpy
import rclpy.subscription
import rclpy.publisher
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from f1tenth_msgs.msg import Waypoints
from geometry_msgs.msg import PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
import os
import csv
import numpy as np
from numpy import linalg as LA
from f1tenth_ros2.scripts import pure_pursuit
from f1tenth_ros2.models import dynamics
import time as tm
 
"""
- Determine the current location of the vehicle. 
- Find the path point closest to the vehicle. 
- Find the goal point 
- Transform the goal point to vehicle coordinates. 
- Calculate the curvature  and request the  vehicle to set the steering to that curvature. 
- Update the vehicles position.
"""
class PPControllerNode(Node):
    """
    The class that handles pure pursuit.
    """
    def __init__(self):
        super().__init__('pp_controller')
        qos_profile_in  = rclpy.qos.qos_profile_system_default
        qos_profile_in.depth = 1
        qos_profile_out = qos_profile_in
        map_index       = 0
        map_list        = ['Sepang', 'Shanghai', 'YasMarina']
        map_name        = map_list[map_index]
        raceline_type   = 'raceline_ED' # raceline_ED or centerline
        self.start_pos  = [0.58955, 0.112556] 
        self.original   = False
        speed_profile   = False if self.original else True # True or False
        self.test_mode  = False
        path            = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/tracks') if not speed_profile \
            else os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/tracks_velocities')
        self.track_name = '{}_{}_test'.format(map_name, raceline_type) if self.test_mode else '{}_{}'.format(map_name, raceline_type)
        self.file_name  = os.path.join(path, self.track_name + '.csv') if not self.test_mode else os.path.join(path+'/test', self.track_name + '.csv') # waypoints file
        self.CTYPE      = 'PP'
        self.currentX   = 0.0 # current position of car
        self.currentY   = 0.0
        self.currentθ   = 0.0 # heading of the car
        self.lookahead  = 0.55 # Lookahead distance (m)
        self.lg         = 1.0 # Lookahead gain
        self.k_a        = 0.25
        self.frame_id   = 'map'
        self.idx        = 0
        self.race_track = []
        self.track_size = 0
        self.waypts     = Waypoints()
        self.vel_cmd    = AckermannDriveStamped()
        self.max_steer  = 1.0
        self.distF_avg  = 1.5
        self.odom_in: rclpy.subscription.Subscription = self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_callback, qos_profile_in)
        self.scan_in: rclpy.subscription.Subscription = self.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile_in)
        # self.pts_pub: rclpy.publisher.Publisher       = self.create_publisher(Waypoints, 'nearest_point', qos_profile_out)
        self.vel_pub: rclpy.publisher.Publisher       = self.create_publisher(AckermannDriveStamped, 'drive', qos_profile_out)
        self.pos_pub: rclpy.publisher.Publisher       = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', qos_profile_out)
        # Initialize vehicle states, inputs, dx/dt, forces
        n_states      = 7
        n_inputs      = 2
        n_steps       = 72 * 60 * 3
        self.states   = np.zeros([n_states, n_steps+1])
        self.dstates  = np.zeros([n_states, n_steps+1])
        self.inputs   = np.zeros([n_inputs, n_steps])
        self.Ffy      = np.zeros(n_steps+1)
        self.Frx      = np.zeros(n_steps+1)
        self.Fry      = np.zeros(n_steps+1)
        self.x_init   = np.zeros(n_states)
        self.step     = 0
        self.time     = 0.
        self.time_f   = np.zeros(n_steps+1)
        self.save_data= False
        

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
        initial_pose = PoseWithCovarianceStamped()
        x_init    = np.zeros(self.states.shape[0])
        x_init[0] = self.start_pos[0]
        x_init[1] = self.start_pos[1]
        x_init[2] = 0.0
        x_init[3] = 0.1
        x_init[4] = self.euler_from_quat(0, 0, 0.0343821, 0.999409)[2]
        self.states[:,0]   = x_init
        self.dstates[:, 0] = dynamics.Dynamic().derivative_eqs(None, x_init, [0, 0])
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
        # print("Distnace -- Left: %.2f, Mid: %.2f, Right: %.2f"%(self.distL_avg, self.distF_avg, self.distR_avg) )
    
    def pose_callback(self, msg:Odometry):
        start = tm.time()
        n = 0
        while n < 800000:
            n += 1
        
        # print("iteration: {}, message receiving frequancy: {:.3f}".format(self.step, 1/(end-start)))
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
        end_thre  = 0.4
        # print(end_dist)
        if end_dist <= end_thre:
            print("Saving data to f1tenth-DYN-{}-{}.npz".format(self.CTYPE, self.track_name))
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
            file_ns = 'f1tenth-DYN-{}-{}.npz'
            path    = '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/raceline/' if 'raceline' in self.track_name \
                        else '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/centerline/'
            if self.original:
                path = '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/original/'
            if self.test_mode:
                path += 'test/'
            if self.save_data:
                np.savez(
                        path + file_ns.format(self.CTYPE, self.track_name),
                        time=time,
                        states=states[:rows-len(idx_d), :],
                        dstates=dstates[:rows-len(idx_d), :],
                        inputs=inputs[:rows-len(idx_d), :],
                        )
            # Kill the node
            self.vel_cmd.drive.speed = 0.0
            self.vel_pub.publish(self.vel_cmd)
            exit()

        """	
        Update current states with measured data from sim
	    x is a 7x1 state vector: [x, y, 𝛿, v, phi, ω, β]^T
		u is a 2x1 input vector: [acc, Δ𝛿]^T
		"""
        self.step                += 1
        self.states[0, self.step] = self.currentX
        self.states[1, self.step] = self.currentY
        self.states[3, self.step] = speed
        self.states[4, self.step] = self.currentθ
        # print(self.currentθ)
        # u_prev = self.inputs[:, self.step-1]
        x0     = self.states[:, self.step]
        inputs = pure_pursuit.PurePursuit().pure_pursuit(x0, self.race_track)
        self.inputs[0, self.step] = inputs[0]  # Acceleration
        self.states[2, self.step] = inputs[1]  # Steering angle
        self.inputs[1, self.step] = self.states[2, self.step] - self.states[2, self.step-1] # Steering velocity
        x_next    = self.states[:, self.step]
        # Update ω and β with numerical integration (Heun's method)
        end         = tm.time()
        h           = end-start
        self.time  += h
        # ω_β_next    = dynamics.Dynamic().odeintHeun2(x_next, h, inputs) 
        ω_β_next    = dynamics.Dynamic().odeintRK6(x_next, h, inputs) 
        self.states[-2:, self.step] = ω_β_next
        # Filtering NaN in data
        self.states[:, self.step] = np.nan_to_num(self.states[:, self.step])
        self.inputs[:, self.step] = np.nan_to_num(self.inputs[:, self.step])
        self.time_f[self.step] = self.time
        dxdt_next = dynamics.Dynamic().derivative_eqs(None, self.states[:, self.step], self.inputs[:,self.step])
        self.dstates[:, self.step] = dxdt_next
        # Update driving commands
        acc          = dxdt_next[3]
        steering_ang = x_next[2]
        velocity     = x_next[3] + acc
        heading      = x_next[4]
        # Avoid hitting the wall
        # if self.distF_avg < 2.5:
        #     velocity     *= min(self.distF_avg, 0.8)
            # steering_ang *= 1.5
        # publish drive commands
        self.vel_cmd.drive.speed          = velocity 
        # if velocity >= 12.:
        #     self.vel_cmd.drive.steering_angle = steering_ang * 0.01
        # else:
        self.vel_cmd.drive.steering_angle = steering_ang 
        self.vel_cmd.drive.acceleration   = acc
        self.vel_cmd.header = msg.header
        print("X: {:.4f}, Y: {:.4f}, Velocity: {:.4f}, Acceleration: {:.4f}, Steering: {:.4f}".format(self.states[0, self.step], self.states[1, self.step], velocity, acc, steering_ang))

        if math.sqrt((self.currentX - self.start_pos[0])**2 + (self.currentY - self.start_pos[1])**2) <= 1.55:
            self.vel_cmd.drive.speed = 3.5
            self.vel_cmd.drive.steering_angle = 0.0
            self.vel_pub.publish(self.vel_cmd)
        # Publish moving commands
        self.vel_pub.publish(self.vel_cmd)
        

def main(args=None):
    rclpy.init(args=args)
    node = PPControllerNode()
    try:
        rclpy.spin(node)
    except Exception as e:
        print("Error: %s" %e)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

