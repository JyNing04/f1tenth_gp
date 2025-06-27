#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
import rclpy.publisher
import rclpy.subscription
from sensor_msgs.msg import Joy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from inputs import devices, UnpluggedError

class TeleopJoyNode(Node):
    def __init__(self):
        super().__init__("teleop_joy")
        self.joy_sub: rclpy.subscription.Subscription = self.create_subscription(Joy, "joy", self.joy_callback, 10)
        self.move_pub: rclpy.publisher.Publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.reset_pose: rclpy.publisher.Publisher = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 10)
        self.linear_scale  = 5.21
        self.angular_scale = 1.1
        self.auto_mode     = False
        self.count         = 0

    def joy_callback(self, data:Joy):
        cmd_vel       = Twist()
        stamp         = data.header.stamp
        ini_pose      = PoseWithCovarianceStamped()
        Axes_ls       = data.axes
        Button_ls     = data.buttons
        # Test joystick Auto/Manual mode
        if Button_ls[2] == 1:
            self.count += 1
        if self.count % 2 == 0:
            self.auto_mode = True # Autonomous driving mode
        elif self.count % 2 != 0: 
            self.auto_mode = False # Manual driving mode, using joystick
        # Teleop mode (joystick)    
        if self.auto_mode == False:
            # Reset position
            if Button_ls[0] == 1 and Button_ls[3] == 1:
                ini_pose.header.stamp       = stamp
                ini_pose.header.frame_id    = "ego_racecar/base_link"
                ini_pose.pose.pose.position.x    = 0.27
                ini_pose.pose.pose.position.y    = 0.01
                ini_pose.pose.pose.orientation.w = 1.0
                ini_pose.pose.covariance[0]      = 0.25
                ini_pose.pose.covariance[7]      = 0.25
                ini_pose.pose.covariance[35]     = 0.0685
                self.reset_pose.publish(ini_pose)
            # Linear velocity
            cmd_vel.linear.x = self.linear_scale * Axes_ls[1] * -1.0
            # Angular velocity
            cmd_vel.angular.z = self.angular_scale * Axes_ls[3] * 1.0
            # print(cmd_vel.angular.z)
            self.move_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = TeleopJoyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

