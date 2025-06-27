#!/usr/bin/env python3
import math
import os
import csv
import numpy as np
# from sklearn.decomposition import FactorAnalysis
# from numpy import linalg as LA
from f1tenth_ros2.params import f110

"""
- Determine the current location of the vehicle. 
- Find the path point closest to the vehicle. 
- Find the goal point 
- Transform the goal point to vehicle coordinates. 
- Calculate the curvature  and request the  vehicle to set the steering to that curvature. 
- Update the vehicles position.
"""
class PurePursuit():
    """
    The class that handles pure pursuit.
    """
    def __init__(self):
        vehicle_params  = f110.F110()
        self.lf         = vehicle_params['lf']
        self.lr         = vehicle_params['lr']
        self.lookahead  = 0.45 # Lookahead distance (m)
        self.lg         = 1.0 # Lookahead gain
        self.k_v        = 2.
        self.idx        = 0
        self.race_type  = 'race' # center or race
        self.speed_ref  = 7.5 # Reference speed
        self.k_a        = self.speed_ref + 1.0
        self.max_steer  = 1.0
        self.v_profile  = True # True or False

    def find_nearest_goal(self, race_track, curr_x, curr_y):
        ranges = []
        for idx in range(len(race_track)):
            dist_x    = math.pow(curr_x - race_track[idx][0],2)
            dist_y    = math.pow(curr_y - race_track[idx][1],2)
            eucl_dist = math.sqrt(dist_x + dist_y)
            ranges.append(eucl_dist)
        return(ranges.index(min(ranges)))

    def pure_pursuit(self, x0, race_track):
        track_size = len(race_track)
        speed      = x0[3]
        lookahead_idx = self.lookahead + self.lg * speed
        min_idx    = int((self.find_nearest_goal(race_track, x0[0], x0[1]) - lookahead_idx) % track_size) # "+ lookahead_dist" for Yas Marina Curcuit
        # Goal point
        goal_x  = race_track[min_idx][0]
        goal_y  = race_track[min_idx][1]
        
        # Lookahead distance L, steering angle
        l_d          = math.sqrt(math.pow(goal_x - x0[0], 2) + math.pow(goal_y - x0[1], 2))
        alpha        = math.atan2(goal_y - x0[1], goal_x - x0[0]) - x0[4] # Look-ahead heading
        steer_output = math.atan2(2.0 * (self.lr + self.lf) * math.sin(alpha), l_d) 
        if abs(steer_output) > self.max_steer:
            steer_output = math.copysign(self.max_steer, steer_output)
       
        # Only use for racecar in simulator, prevent oscillation (high-pass filter)
        k_s          = 0.0035 
        steer_th     = 0.025
        
        # steer_output = steer_output * 1.13 if self.race_type == 'center' else steer_output * self.speed_ref * 0.17
        steering_ang = steer_output if abs(steer_output) > k_s else 0.0
        
        # Publish throttle & steering to car control
        if self.v_profile: # dynamic speed ref
            ref_vel = race_track[min_idx][2]
            
            acc = self.k_v * (ref_vel - speed) # / ref_vel if x0[3] >= 2.0 else 2.0 
            # Reduce speed when entering sharp turns
            if abs(steering_ang) >= steer_th: 
                if speed > self.speed_ref:
                    acc = (self.k_v/2. * (ref_vel-1.5-speed) - self.k_a*abs(steering_ang) - steer_th * (speed+2.5)) / ref_vel
                else: 
                    acc = (self.k_v/2. * (ref_vel-speed) - self.k_a*abs(steering_ang) - steer_th * self.lookahead * (speed+self.k_v)) / self.speed_ref
        else: # fixed speed ref
            acc = self.k_v * (self.speed_ref - speed)
            # Reduce speed when entering sharp turns
            if abs(steering_ang) >= steer_th: 
                acc = (self.k_v/2.5 * (self.speed_ref - speed) - self.k_a*(abs(steering_ang) + steer_th * speed)) / self.speed_ref
        steering = steering_ang * 1.25 if self.race_type == 'center' else steering_ang * self.speed_ref * 0.26
        return([acc, steering])


