#!/usr/bin/env python3
import math
from visualization_msgs.msg import Marker, MarkerArray
import rclpy
import rclpy.subscription
import rclpy.publisher
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from f1tenth_ros2.mpc.planner import MPCPlanner
import os
import csv
import numpy as np
from numpy import linalg as LA
 

class ReferencePathVizNode(Node):
    """
    The class that handles pure pursuit.
    """
    def __init__(self):
        super().__init__('reference_path_viz')
        self.path       = os.path.expanduser('~/dev_ws/src/f1tenth_ros2/f1tenth_ros2/data/tracks') 
        map_index       = 0
        map_list        = ['Sepang', 'Shanghai', 'YasMarina']
        map_name        = map_list[map_index]
        raceline_type   = '_raceline_ED' # _centerline or _raceline_ED
        self.track_name = map_name + raceline_type
        self.track_list = [map_name, raceline_type]
        self.file_name  = os.path.join(self.path, self.track_name + '.csv') # waypoints file
        self.currentX   = 0.0 # current position of car
        self.currentY   = 0.0
        self.currentθ   = 0.0 # heading of the car
        self.lookahead  = 0.45 # Lookahead distance (m)
        self.lg         = 1.0 # Lookahead gain
        self.frame_id   = 'map'
        self.marker_id  = 0
        self.N          = 10
        # self.idx        = 0
        self.race_track = []
        self.track_size = 0
        self.dur        = Duration()
        self.dur.sec    = 1
        self.odom_in: rclpy.subscription.Subscription = self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_callback, 10)
        self.rvpts_pub: rclpy.publisher.Publisher = self.create_publisher(MarkerArray, 'reference_path_viz', 1)
        
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
            idx = idx % self.track_size
            dist_x    = math.pow(curr_x - self.race_track[idx][0],2)
            dist_y    = math.pow(curr_y - self.race_track[idx][1],2)
            eucl_dist = math.sqrt(dist_x + dist_y)
            ranges.append(eucl_dist)
        return(ranges.index(min(ranges)))

    def pose_callback(self, msg:Odometry):
        ref_path   = MarkerArray()
        linear_x = msg.twist.twist.linear.x
        linear_y = msg.twist.twist.linear.y
        linear_z = msg.twist.twist.linear.z
        speed    = LA.norm(np.array([linear_x, linear_y, linear_z]),2)
        self.currentX = msg.pose.pose.position.x
        self.currentY = msg.pose.pose.position.y
        x0       = [self.currentX, self.currentY, 0.1, speed, 0.0]
        planner  = MPCPlanner(x0, self.path, self.track_list, self.N)
        _, _, xref, _ = planner.projectXref()
        if not self.race_track:
            self.construct_path()
        else:
            # lookahead_dist = self.lookahead + self.lg * speed
            # min_idx = int((self.find_nearest_goal(self.currentX, self.currentY) - lookahead_dist) % self.track_size) # "+ lookahead_dist" for Yas Marina Curcuit
            # Visualize the lookahead waypoint
            for i in range(xref.shape[1]-1):
                i = i % self.track_size
                marker                    = Marker()
                marker.header.frame_id    = self.frame_id
                marker.ns                 = 'Goal-{}'.format(i)
                marker.type               = 2
                marker.action             = 0
                marker.id                 = i
                marker.pose.position.x    = xref[0][i]
                marker.pose.position.y    = xref[1][i]
                marker.pose.position.z    = 0.0
                marker.scale.x            = 0.15
                marker.scale.y            = 0.15
                marker.scale.z            = 0.15
                marker.color.r            = 1.0
                marker.color.b            = 0.0
                marker.color.g            = 0.3
                marker.color.a            = 1.0
                marker.lifetime           = self.dur
                ref_path.markers.append(marker)
            self.rvpts_pub.publish(ref_path)


def main(args=None):
    rclpy.init(args=args)
    node = ReferencePathVizNode()
    try:
        rclpy.spin(node)
    except Exception as e:
        print("Error: %s" %e)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

