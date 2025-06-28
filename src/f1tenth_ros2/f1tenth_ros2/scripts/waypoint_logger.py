import math
import rclpy
import atexit
import numpy as np
from rclpy.node import Node
import rclpy.publisher
import rclpy.subscription
from numpy import linalg as LA
# from sensor_msgs.msg import Joy
from std_msgs.msg import Header
from f1tenth_msgs.msg import Waypoints
# from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
# from sensor_msgs.msg import LaserScan
import os.path 
from geometry_msgs.msg import Point

class WaypointLoggerNode(Node):
    def __init__(self):
        super().__init__("waypoint_logger")
        path        = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/Tracks')
        track_name  = 'Shanghai_map'
        file_name   = os.path.join(path, track_name + '.csv')
        self.file   = open(file_name, 'w')
        self.odom_in: rclpy.subscription.Subscription = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)
        self.waypts_pub: rclpy.publisher.Publisher = self.create_publisher(Waypoints, 'waypoints', 1)
        atexit.register(self.file_shutdown)

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

    def odom_callback(self, data:Odometry):
        n = 0
        while n < 500000:
            n += 1
        # Heading of the car
        waypts = Waypoints()
        euler = self.euler_from_quat(data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        # Speed of the car
        linear_x = data.twist.twist.linear.x
        linear_y = data.twist.twist.linear.y
        linear_z = data.twist.twist.linear.z
        speed    = LA.norm(np.array([linear_x, linear_y, linear_z]),2)
        # Logging waypoints to csv file
        self.file.write('%f, %f, %f, %f\n' % (data.pose.pose.position.x, data.pose.pose.position.y, euler[2], speed))
        # Publishing waypoints to topic
        waypts.position.x   = data.pose.pose.position.x
        waypts.position.y   = data.pose.pose.position.y
        waypts.position.z   = data.pose.pose.position.z
        waypts.yaw = float(euler[2])
        waypts.vel = float(speed)
        self.waypts_pub.publish(waypts)

    def file_shutdown(self):
        self.file.close()
        print("File closed...")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointLoggerNode()
    rclpy.spin(node)
    rclpy.shutdown()
                
if __name__ == '__main__':
    main()


    
    
