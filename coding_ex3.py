# Student name: Antony Silvetti-Schmitt 

import math
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped, TransformStamped
from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
import matplotlib.pyplot as plt
import time
from tf2_msgs.msg import TFMessage
from copy import copy
from visualization_msgs.msg import Marker

# Further info:
# On markers: http://wiki.ros.org/rviz/DisplayTypes/Marker
# Laser Scan message: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html

class CodingExercise3(Node):
   

    
    def __init__(self):
        super().__init__('CodingExercise3')

        self.ranges = [] # lidar measurements
        
        self.point_list = [] # A list of points to draw lines
        self.line = Marker()
        self.line_marker_init(self.line)


        # Ros subscribers and publishers
        self.subscription_ekf = self.create_subscription(Odometry, 'terrasentia/ekf', self.callback_ekf, 10)
        self.subscription_scan = self.create_subscription(LaserScan, 'terrasentia/scan', self.callback_scan, 10)
        self.pub_lines = self.create_publisher(Marker, 'lines', 10)
        self.timer_draw_line_example = self.create_timer(0.1, self.draw_line_example_callback)

    
    def callback_ekf(self, msg):
        # You will need this function to read the translation and rotation of the robot with respect to the odometry frame
        pass
   
    def callback_scan(self, msg):
        self.ranges = list(msg.ranges) # Lidar measurements
        print("some-ranges:", self.ranges[0:5])
        print("Number of ranges:", len(self.ranges))

    def draw_line_example_callback(self):
        # Here is just a simple example on how to draw a line on rviz using line markers. Feel free to use any other method
        p0 = Point()
        p0.x = 0.0
        p0.y = 0.0
        p0.z = 0.0

        p1 = Point()
        p1.x = 1.0
        p1.y = 1.0
        p1.z = 1.0

        self.point_list.append(copy(p0)) 
        self.point_list.append(copy(p1)) # You can append more pairs of points
        self.line.points = self.point_list

        self.pub_lines.publish(self.line) # It will draw a line between each pair of points

    def line_marker_init(self, line):
        line.header.frame_id="/odom"
        line.header.stamp=self.get_clock().now().to_msg()

        line.ns = "markers"
        line.id = 0

        line.type=Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0

        line.scale.x = 0.05
        line.scale.y= 0.05
        
        line.color.r = 1.0
        line.color.a = 1.0
        #line.lifetime = 0


def main(args=None):
    rclpy.init(args=args)

    cod3_node = CodingExercise3()
    
    rclpy.spin(cod3_node)

    cod3_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
