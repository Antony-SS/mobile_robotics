# version 0.0
# Jose Cuaran


import math
import numpy as np
import rclpy
from rclpy.node import Node
#from rclpy.clock import Clock

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from mobile_robotics.utils import quaternion_from_euler, lonlat2xyz #edit according to your package's name

class OdometryNode(Node):
    # Initialize some variables
    
    gyro_yaw = 0.0

    # These are linear velocities, so I don't need to do r*theta to find

    blspeed = 0.0 # back left wheel speed
    flspeed = 0.0 # front left wheel speed
    brspeed = 0.0 
    frspeed = 0.0


    x = 0.0 # x robot's position
    y = 0.0 # y robot's position
    theta = np.pi/2 # heading angle, setting as pi/2 to start to line up with GPS measurements
    l_wheels = 0.3 # Distance between right and left wheels

    last_time = 0.0
    current_time = 0.0

    def __init__(self):
        super().__init__('minimal_subscriber')
        
        # Declare subscribers to all the topics in the rosbag file, like in the example below. Add the corresponding callback functions.
        # your code here
        self.subscription_Gyro_yaw = self.create_subscription(Float32, 'Gyro_yaw', self.callback_Gy, 10)
        self.subscription_Gyro_roll = self.create_subscription(Float32, 'Gyro_roll', self.callback_Gr, 10)
        self.subscription_Gyro_pitch = self.create_subscription(Float32, 'Gyro_pitch', self.callback_Gp, 10)
        self.subsription_Accel_y = self.create_subscription(Float32, 'Accely', self.callback_Ay, 10)
        self.subscription_Accel_x = self.create_subscription(Float32, 'Accelx', self.callback_Ax, 10)
        self.subscription_Accel_z = self.create_subscription(Float32, 'Accelz', self.callback_Az, 10)
        self.subscription_Bls = self.create_subscription(Float32, 'Blspeed', self.callback_Blspeed, 10)
        self.subscription_Brs = self.create_subscription(Float32, 'Brspeed', self.callback_Brspeed, 10)
        self.subscription_Fls = self.create_subscription(Float32, 'Flspeed', self.callback_Flspeed, 10)
        self.subscription_Frs = self.create_subscription(Float32, 'Frspeed', self.callback_Frspeed, 10)
        self.subscription_lat = self.create_subscription(Float32, 'latitude', self.callback_lat, 10)
        self.subscription_long = self.create_subscription(Float32, 'longitude', self.callback_long, 10)

        self.last_time = self.get_clock().now().nanoseconds/1e9
        
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10) #keep in mind how to declare publishers for next assignments
        self.timer = self.create_timer(0.1, self.timer_callback_odom) #It creates a timer to periodically publish the odometry.
        
        self.tf_broadcaster = TransformBroadcaster(self) # To broadcast the transformation between coordinate frames.

        self.file_object_results  = open("results_part1.txt", "w+")
        self.timer2 = self.create_timer(0.1, self.callback_write_txt_file) #Another timer to record some results in a .txt file
        


    def callback_Gy(self, msg):
        self.gyro_yaw = msg.data

    def callback_Gr(self, msg):
        pass

    def callback_Gp(self, msg):
        pass
    
    def callback_Ay(self, msg):
        pass

    def callback_Ax(self, msg):
        pass

    def callback_Az(self, msg):
        pass
    
    def callback_Blspeed(self, msg):
        self.blspeed = msg.data

    def callback_Brspeed(self, msg):
        self.brspeed = msg.data

    def callback_Flspeed(self, msg):
        self.flspeed = msg.data

    def callback_Frspeed(self, msg):
        self.frspeed = msg.data
    
    def callback_lat(self, msg):
        pass

    def callback_long(self, msg):
        pass
    
    def conjugate_quaternion(self, q):
        # Returns the conjugate of a given quaternion
        return [-q[0], -q[1], -q[2], q[3]]

    def quat_rotation(self,q,v):
        # returns the result of applying the L(v) = qvq* transformation.  Uses matrix form to make it simpler.
        # q should be a python list of length 4
        q0,q1,q2,q3 = q

        Q = np.array([[2*q0**2 - 1 + 2*q1**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
                      [2*q1*q2 + 2*q0*q3, 2*q0**2 - 1 + 2*q2**2, 2*q2*q3 - 2*q0*q1],
                      [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 2*q0**2 - 1 + 2*q3**2]])
        
        return Q@v

    def timer_callback_odom(self):
        '''
        Compute the linear and angular velocity of the robot using the differential-drive robot kinematics
        Perform Euler integration to find the position x and y of the robot
        '''

        self.current_time = self.get_clock().now().nanoseconds/1e9
        dt = self.current_time - self.last_time # DeltaT
        
        vl = (self.blspeed + self.flspeed)/2.0  # Average Left-wheels speed
        vr = (self.brspeed + self.frspeed)/2.0  # Average right-wheels speed
        
        v = (vl + vr) / 2.0 # ... Linear velocity of the robot -> can only move in the direction the robot is pointing (dubins car assumption)
        # w = (vl - vr) / (self.l_wheels) # ... Angular velocity of the robot = angular velocity or right - angular velocity of left.  
        w = self.gyro_yaw
        # Now time to do some Euler integration!

        # ASSUMPTION: This x & y is in the global (inertial) frame . . .
        # V vector is pointing in the heading direction, take sin and cos to get updates to x & y
        self.x = self.x + dt*v*np.cos(self.theta)  # Position is a function of linear velocity and time, difference in these velocities will contribute to ang. vel.
        self.y = self.y + dt*v*np.sin(self.theta)

        # ...Heading angle -> Feel like I could calculate this in two ways: 
        # 1. by integrating gyro data (gyrd acceleration) to theta
        # 2. tracking with angular velocity of robot (comes from encoders on the wheel), integrating that -> will do this for now
        self.theta = (self.theta + (dt*w))   # have to keep this in range of pi to -pi for quaternion transformation

        if (self.theta > np.pi):
            diff = self.theta - np.pi
            self.theta = -np.pi + diff
        elif(self.theta < -np.pi):
            diff = -np.pi - self.theta # should always be positive
            self.theta = np.pi - diff

        position = [self.x, self.y, self.theta]

        quater = quaternion_from_euler(0.0, 0.0, self.theta)

        print("position: ", position)
        print("orientation: ", quater)

        # We need to set an odometry message and publish the transformation between two coordinate frames
        # Further info about odometry message: https://docs.ros2.org/foxy/api/nav_msgs/msg/Odometry.html
        # Further info about tf2: https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html
        # Further info about coordinate frames in ROS: https://www.ros.org/reps/rep-0105.html

        frame_id = 'odom'
        child_frame_id = 'base_link'
        
        
        self.broadcast_tf(position, quater, frame_id, child_frame_id)  # Before creating the odometry message, go to the broadcast_tf function and complete it.
        
        odom = Odometry()
        odom.header.frame_id = frame_id
        odom.header.stamp = self.get_clock().now().to_msg()

        # let me actually calculate the global positon after the rotation now . . .

        # qstar = self.conjugate_quaternion(quater)
        # pose = self.quat_rotation(qstar, position) # passing qstar because I want to rotate coordinate frame

        # print("transformation lead to pose: ", pose)

        # set the pose. Uncomment next lines

        odom.pose.pose.position.x = self.x # ...
        odom.pose.pose.position.y = self.y # ...
        odom.pose.pose.position.z = 0.0 # should always be zero 
        
        # This doesn't make sense to me ... how can we represent it's orientation with a quaternion?  The transformation requires qvq* ...
        odom.pose.pose.orientation.x = quater[0]
        odom.pose.pose.orientation.y = quater[1]
        odom.pose.pose.orientation.z = quater[2]
        odom.pose.pose.orientation.w = quater[3]

        # set the velocities. Uncomment next lines
        odom.child_frame_id = child_frame_id
        odom.twist.twist.linear.x = v*np.cos(self.theta) # v* cos of heading
        odom.twist.twist.linear.y = v*np.sin(self.theta) # v* sin of heading
        odom.twist.twist.linear.z = 0.0 # no vertical velocity
        odom.twist.twist.angular.x = 0.0 # ... ASSUMING 0
        odom.twist.twist.angular.y = 0.0 # ... ASSUMING 0
        odom.twist.twist.angular.z = w # angular velocity for yaw

        self.odom_pub.publish(odom)

        self.last_time = self.current_time
        
    def broadcast_tf(self, pos, quater, frame_id, child_frame_id):
        '''
        It continuously publishes the transformation between two reference frames.
        Complete the translation and the rotation of this transformation
        '''
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id

        # Uncomment next lines and complete the code -> Don't think there is any translation

        # the translation needed to reach robot coordinate frame from global 
        t.transform.translation.x = pos[0] # ...
        t.transform.translation.y = pos[1] # ...
        t.transform.translation.z = 0.0 # ...

        # Think it wants quaternion

        # the quat used to rotate to the robot coordinate frame from global
        t.transform.rotation.x = quater[0]
        t.transform.rotation.y = quater[1]
        t.transform.rotation.z = quater[2]
        t.transform.rotation.w = quater[3]


        # Send the transformation
        self.tf_broadcaster.sendTransform(t)
    
    def callback_write_txt_file(self):
        if (self.x != 0 or self.y != 0 or self.theta != 0):
            self.file_object_results.write(str(self.current_time) + " " + str(self.x)+" "+str(self.y)+" "+str(self.theta)+"\n")

    

def main(args=None):
    rclpy.init(args=args)

    odom_node = OdometryNode()

    rclpy.spin(odom_node)
    odom_node.file_object_results.close()
    odom_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
