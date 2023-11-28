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

from tf2_ros import TransformBroadcaster

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

        # self.tf_broadcaster = TransformBroadcaster(self) # To broadcast static transform between map and odom (rate limited)
        # self.timer_transform = self.create_timer(0.005, self.transform_callback)

    
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

    def transform_callback(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        # Uncomment next lines and complete the code -> Don't think there is any translation

        # the translation needed to reach robot coordinate frame from global 
        t.transform.translation.x = 0.0 # ...
        t.transform.translation.y = 0.0 # ...
        t.transform.translation.z = 0.0 # ...

        # Think it wants quaternion

        # the quat used to rotate to the robot coordinate frame from global
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    def line_fitting_helper(self, points):
        endptsxs = np.array([points[0,0], points[-1,0]])
        endptys = np.array([points[0,1], points[-1,1]])

        m, intercept = np.polyfit(endptsxs, endptys, 1) # get parameters of line fitting these two points
        # print(f"slope: {m} and intercept is: {intercept}")
        # THIS WAS LST SQUARES, SWITCHING TO END PT FITTING
        # ones = np.ones([xpts.shape[0]])
        # A = np.hstack(xpts, ypts)
        # b = ones
        # m, intercept = np.inv(A.T@A)@A.T@b # normal equations

        a = -m
        b = 1
        c = -intercept

        return a,b,c


    def split_helper(self, splitList,set):
        THRESH = 0.4 # can play with this, think 1m is good
        # find point with max distance from said line
        maxdist = (-1, -1) # distance, index
        # check stop condition -> this means that we can't have a line w/ less than three points, can't have singular points as boundaries
        if (len(set) < 3): # can't split anymore
            return splitList
        else:
            a,b,c = self.line_fitting_helper(set)
        
        # find max distance to fit line
        for j, point in enumerate (set):
            dist = abs(a*point[0] + b*point[1] + c) / np.sqrt(a**2 + b**2)
            if (dist > maxdist[0]):
                maxdist = (dist, j)
        
        # print("MAX DIST IS: ", maxdist)
        if (maxdist[0] > THRESH): # split condition
            # check distance to each side of split point so I know which side to include it with
            distRight = 0
            distLeft = 0
            j = maxdist[1]
            # print(f"Splitting about point:  {j} which is {set[j]}")

            
            self.split_helper(splitList, set[0:j+1]) # don't change the split list b/c we are greater than threshold
            self.split_helper(splitList, set[j:set.shape[0]])

        else:
            # print("Appending!")
            splitList.append(set)
            # print("After appending split list is: ", splitList)
            # return splitList

    def merge_helper(self, splitList):
        # print("Entering Merge")
        # now if adjacenet items in list are close and colinear (to some threshold I set), then we will combine -> also a recursive problem
        i = 0
        while (i < len(splitList) - 1):
            # print("i top is: ", i)
            line1 = splitList[i]
            line2 = splitList[i+1]

            # take the endpoint of line 1, startpoint of line 2
            line1EndPt = line1[-1]
            line2StrtPt = line2[0]

            # distance between points
            dist = np.linalg.norm(line1EndPt - line2StrtPt)

            condition1 = (dist < 20) # tunable
            
            # now get ratio between slopes
            a,b,c = self.line_fitting_helper(line1)
            slopeLine1 = -a/b
            a,b,c = self.line_fitting_helper(line2)
            slopeLine2 = -a/b

            # print(slopeLine1)
            # print(slopeLine2)

            ratio = slopeLine1/slopeLine2

            # will do this based on ratio where 1 is perfect colinearlity
            # print("ratio is ", ratio)
            condition2 = (ratio > 0.8 and ratio < 1.2)

            # this is because we have to code around that face that -.2 and 0.2 will be at most 0.4 apart in slope, but our ratio won't recognize it
            # I'm sure I could go to polar coordinates and do this in a prettier way with atan2, but this works for now
            if not condition2:
                if (abs(slopeLine1) < 0.2  and abs (slopeLine1) < 0.2):
                    condition2 = True
            
            # merge, reset list index
            if (condition1 and condition2):
                print(f"Merging indexes  {i} and {i+1}")
                mergedLine =  np.append(line1, line2, axis=0)
                del splitList[i:i+2]
                splitList.insert(i, mergedLine)
                # print(splitList)
                # print(len(splitList))
                i = i-2 # b/c it will index below and we want to be 0 and we have to reconsider whole list
            
            i += 1


    def split_and_merge(self, rhos, thetas):
        points = np.hstack([rhos*np.cos(thetas), rhos*np.sin(thetas)]) # just turning all to cartesian b/c I'm more familiar
        startingSet = points 
        # Split step
        splitList = []
        self.split_helper(splitList, startingSet)
        self.merge_helper(splitList)

        # At the end I have to decide if I want to go through all points in a line and split if they are too far apart . . . 
        return splitList



def main(args=None):
    rclpy.init(args=args)

    cod3_node = CodingExercise3()
    
    rclpy.spin(cod3_node)

    cod3_node.destroy_node()
    rclpy.shutdown()




if __name__ == '__main__':
    main()
