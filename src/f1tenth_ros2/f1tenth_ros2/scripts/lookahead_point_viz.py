#!/usr/bin/env python3
import math
from visualization_msgs.msg import Marker
import rclpy
import rclpy.subscription
import rclpy.publisher
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from f1tenth_msgs.msg import Waypoints
import os
import csv
import numpy as np
from numpy import linalg as LA
 

class LookaheadPointVizNode(Node):
    """
    The class that handles pure pursuit.
    """
    def __init__(self):
        super().__init__('lookahead_point_viz')
        path            = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/tracks') 
        map_index       = 0
        map_list        = ['Sepang', 'Shanghai', 'YasMarina']
        map_name        = map_list[map_index]
        track_name      = map_name + '_raceline_ED' # _centerline or _raceline_ED
        self.file_name  = os.path.join(path, track_name + '.csv') # waypoints file
        self.currentX   = 0.0 # current position of car
        self.currentY   = 0.0
        self.currentθ   = 0.0 # heading of the car
        self.lookahead  = 0.45 # Lookahead distance (m)
        self.lg         = 1.0 # Lookahead gain
        self.frame_id   = 'map'
        self.idx        = 0
        self.race_track = []
        self.track_size = 0
        self.goal       = Marker()
        self.dur        = Duration()
        self.dur.sec    = 1
        self.odom_in: rclpy.subscription.Subscription = self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_callback, 10)
        self.rvpts_pub: rclpy.publisher.Publisher = self.create_publisher(Marker, 'nearest_point_rviz', 1)
        
    def construct_path(self): 
        with open(self.file_name, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter =',')
            next(csv_reader)
            for waypoint in csv_reader:
                self.race_track.append(waypoint)
            # Force elements in racetrack to be float
            self.race_track = list(np.array(self.race_track, dtype='float'))
            # self.race_track = [[float(y) for y in x] for x in self.race_track]
            self.track_size = len(self.race_track)

    def find_nearest_goal(self, curr_x, curr_y):
        ranges = []
        for idx in range(len(self.race_track)):
            dist_x    = math.pow(curr_x - self.race_track[idx][0],2)
            dist_y    = math.pow(curr_y - self.race_track[idx][1],2)
            eucl_dist = math.sqrt(dist_x + dist_y)
            ranges.append(eucl_dist)
        return(ranges.index(min(ranges)))

    def pose_callback(self, msg:Odometry):
        
        linear_x = msg.twist.twist.linear.x
        linear_y = msg.twist.twist.linear.y
        linear_z = msg.twist.twist.linear.z
        speed    = LA.norm(np.array([linear_x, linear_y, linear_z]),2)
        self.currentX = msg.pose.pose.position.x
        self.currentY = msg.pose.pose.position.y
        if not self.race_track:
            self.construct_path()
        else:
            lookahead_dist = self.lookahead + self.lg * speed
            min_idx = int((self.find_nearest_goal(self.currentX, self.currentY) - lookahead_dist) % self.track_size) # "+ lookahead_dist" for Yas Marina Curcuit
            # Visualize the lookahead waypoint
            self.goal.header.frame_id    = self.frame_id
            self.goal.ns                 = 'Visualization'
            self.goal.type               = 2
            self.goal.action             = 0
            self.goal.id                 = self.idx
            self.goal.pose.position.x    = self.race_track[min_idx][0]
            self.goal.pose.position.y    = self.race_track[min_idx][1]
            self.goal.pose.position.z    = 0.0
            self.goal.scale.x            = 0.35
            self.goal.scale.y            = 0.35
            self.goal.scale.z            = 0.35
            self.goal.color.r            = 1.0
            self.goal.color.b            = 0.0
            self.goal.color.g            = 0.3
            self.goal.color.a            = 1.0
            self.goal.lifetime           = self.dur
            self.rvpts_pub.publish(self.goal)


def main(args=None):
    rclpy.init(args=args)
    node = LookaheadPointVizNode()
    try:
        rclpy.spin(node)
    except Exception as e:
        print("Error: %s" %e)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

