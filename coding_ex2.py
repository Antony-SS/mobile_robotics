# Student name: Antony Silvetti-Schmitt

import math
import numpy as np
from numpy import linalg as LA
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Accel
from tf2_ros import TransformBroadcaster

from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
import time
from mobile_robotics.utils import quaternion_from_euler, lonlat2xyz, quat2euler


class ExtendedKalmanFilter(Node):

    
    def __init__(self):
        super().__init__('ExtendedKalmanFilter')
        
        
        #array to save the sensor measurements from the rosbag file
        #measure = [p, q, r, fx, fy, fz, x, y, z, vx, vy, vz] 

        self.measure = np.zeros(12)
        
        #Initialization of the variables used to generate the plots.

        self.PHI = []  
        self.PSI = []
        self.THETA = []
        self.P_R = []
        self.P_R1 = []
        self.P_R2 = []
        self.Pos = []
        self.Vel = []
        self.Quater = []
        self.measure_PosX = []
        self.measure_PosY = []
        self.measure_PosZ = []
        self.P_angular = []
        self.Q_angular = []
        self.R_angular = []
        self.P_raw_angular = []
        self.Q_raw_angular = []
        self.R_raw_angular = []
        self.Bias =[]
        
        self.POS_X = []
        self.POS_Y = []
        
        
        #Initialization of the variables used in the EKF
        
        # initial bias values, these are gyroscope and accelerometer biases
        self.bp= 0.0
        self.bq= 0.0
        self.br= 0.0
        self.bfx = 0.0
        self.bfy = 0.0
        self.bfz = 0.0
        # initial rotation
        self.q2, self.q3, self.q4, self.q1 = quaternion_from_euler(0.0, 0.0, np.pi/2) #[qx,qy,qz,qw]

        #initialize the state vector [x y z vx vy vz          quat    gyro-bias accl-bias]
        self.xhat = np.array([[0, 0, 0, 0, 0, 0, self.q1, self.q2, self.q3, self.q4, self.bp, self.bq, self.br, self.bfx, self.bfy, self.bfz]]).T

        self.rgps=np.array([-0.15, 0 ,0]) #This is the location of the GPS wrt CG, this is very important
        
        #noise params process noise (my gift to you :))
        self.Q = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.5, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
        
        #measurement noise
        # GPS position and velocity -> R is the associated covariance
        self.R = np.diag([10, 10, 10, 2, 2, 2])
        
       
        #Initialize P, the covariance matrix (for process)
        self.P = np.diag([30, 30, 30, 3, 3, 3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.Pdot=self.P*0.0
        
        self.time = []
        self.loop_t = 0

        # You might find these blocks useful when assembling the transition matrices
        self.Z = np.zeros((3,3))
        self.I = np.eye(3,3)
        self.Z34 = np.zeros((3,4))
        self.Z43 = np.zeros((4,3))
        self.Z36 = np.zeros((3,6))

        self.lat = 0
        self.lon = 0
        self.lat0 = 0
        self.lon0 = 0
        self.flag_lat = False
        self.flag_lon = False
        self.dt = 0.0125 # set sample time

        # Ros subscribers and publishers
        self.subscription_imu = self.create_subscription(Imu, 'terrasentia/imu', self.callback_imu, 10)
        self.subscription_gps_lat = self.create_subscription(Float32, 'gps_latitude', self.callback_gps_lat, 10)
        self.subscription_gps_lon = self.create_subscription(Float32, 'gps_longitude', self.callback_gps_lon, 10)
        self.subscription_gps_speed_north = self.create_subscription(Float32, 'gps_speed_north', self.callback_gps_speed_north, 10)
        self.subscription_gps_speend_east = self.create_subscription(Float32, 'gps_speed_east', self.callback_gps_speed_east, 10)
        
        self.timer_ekf = self.create_timer(self.dt, self.ekf_callback)
        self.timer_plot = self.create_timer(1, self.plot_data_callback)

    
    def callback_imu(self,msg):
        
        #measurement vector = [p, q, r, fx, fy, fz, x, y, z, vx, vy, vz]
        # In practice, the IMU measurements should be filtered. In this coding exercise, we are just going to clip
        
        # the values of velocity and acceleration to keep them in physically possible intervals.
        self.measure[0] = np.clip(msg.angular_velocity.x,-5,5) #(-5,5)
        self.measure[1] = np.clip(msg.angular_velocity.y,-5,5) #..(-5,5)
        self.measure[2] = np.clip(msg.angular_velocity.z,-5,5) #..(-5,5)
        self.measure[3] = np.clip(msg.linear_acceleration.x, -6,6) #..(-6,6)
        self.measure[4] = np.clip(msg.linear_acceleration.y, -6,6) #..(-6,6)
        self.measure[5] = np.clip(msg.linear_acceleration.z, -16,-4) #..(-16,-4) -> this is just (-6,6) -10 for gravity
 
    def callback_gps_lat(self, msg):
        self.lat = msg.data
        if (self.flag_lat == False): #just a trick to recover the initial value of latitude
            self.lat0 = msg.data
            self.flag_lat = True
        
        if (self.flag_lat and self.flag_lon): 
            x, y = lonlat2xyz(self.lat, self.lon, self.lat0, self.lon0) # convert latitude and longitude to x and y coordinates
            self.measure[6] = x
            self.measure[7] = y
            self.measure[8] = 0.0 

    
    def callback_gps_lon(self, msg):
        self.lon = msg.data
        if (self.flag_lon == False): #just a trick to recover the initial value of longitude
            self.lon0 = msg.data
            self.flag_lon = True    
    
    def callback_gps_speed_east(self, msg): 
        self.measure[9] = msg.data # .data is right here.  Tested it.
        self.measure[11] = 0.0 # vz

    def callback_gps_speed_north(self, msg):
        self.measure[10] = msg.data # vy

   
    def ekf_callback(self):
        #print("iteration:  ",self.loop_t)
        if (self.flag_lat and self.flag_lon):  #Trick  to sincronize rosbag with EKF
            self.ekf_function()
        else:
            print("Play the rosbag file...")

    
    
    def ekf_function(self):
        
        # Adjusting angular velocities and acceleration with the corresponding bias

        self.p = (self.measure[0]-self.xhat[10,0])
        self.q = (self.measure[1]-self.xhat[11,0])
        self.r = self.measure[2]-self.xhat[12,0]
        self.fx = (self.measure[3]-self.xhat[13,0])
        self.fy = (self.measure[4]-self.xhat[14,0])
        self.fz = self.measure[5]-self.xhat[15,0] # corrected the given code here, not sure why sign was flipped
        
        # Get the current quaternion values from the state vector
        # Remember again the state vector [x y z vx vy vz q1 q2 q3 q4 bp bq br bx by bz]
        self.quat = np.array([[self.xhat[6,0], self.xhat[7,0], self.xhat[8,0], self.xhat[9,0]]]).T
    
        self.q1 = self.quat[0,0]
        self.q2 = self.quat[1,0]
        self.q3 = self.quat[2,0]
        self.q4 = self.quat[3,0]
                
        # Rotation matrix: body to inertial frame
        self.R_bi = np.array([[pow(self.q1,2)+pow(self.q2,2)-pow(self.q3,2)-pow(self.q4,2), 2*(self.q2*self.q3-self.q1*self.q4), 2*(self.q2*self.q4+self.q1*self.q3)],
                          [2*(self.q2*self.q3+self.q1*self.q4), pow(self.q1,2)-pow(self.q2,2)+pow(self.q3,2)-pow(self.q4,2), 2*(self.q3*self.q4-self.q1*self.q2)],
                          [2*(self.q2*self.q4-self.q1*self.q3), 2*(self.q3*self.q4+self.q1*self.q2), pow(self.q1,2)-pow(self.q2,2)-pow(self.q3,2)+pow(self.q4,2)]])
        
            
        #Prediction step
        #First write out all the dots for all the states, e.g. pxdot, pydot, q1dot etc

        pxdot = self.xhat[3,0]
        pydot = self.xhat[4,0]
        pzdot = self.xhat[5,0] # Pretty sure this is the right assumption -> can i just set to 0 ...

        # Putting accelerations into inertial frame
        accels = np.array([self.fx, self.fy, self.fz]) # biases already subtracted
        vdots = self.R_bi@(accels)

        vxdot = vdots[0]
        vydot = vdots[1]
        vzdot = vdots[2]

        print("vzdot is: ", vzdot)

        # print("vxdot: ", vxdot)
        # print("vydot: ", vydot)
        # print("vzdot: ", vzdot)

        omegaw = self.omega(self.p, self.q,self.r)
        qdot = -(0.5)*omegaw@np.array([self.q1,self.q2, self.q3, self.q4]) 

        q1dot = qdot[0] # This is the angle of rotation
        q2dot = qdot[1] # x part of axis of rotation
        q3dot = qdot[2] # y part of axis of rotation
        q4dot = qdot[3] # z part of axis of rotation
        bpdot = 0
        bqdot = 0
        brdot = 0
        bxdot = 0
        bydot = 0
        bzdot = 0
        


        # .. your code here
        
        #Now integrate Euler Integration for Process Updates and Covariance Updates
        # Euler works fine
        # Remember again the state vector [x y z vx vy vz q1 q2 q3 q4 bp bq br bx by bz]
        self.xhat[0,0] = self.xhat[0,0] + self.dt*pxdot
        self.xhat[1,0] = self.xhat[1,0] + self.dt*pydot # ..
        self.xhat[2,0] = self.xhat[2,0] + self.dt*pzdot # .. double check this
        self.xhat[3,0] = self.xhat[3,0] + self.dt*vxdot # vx = vx + deltaT*measuredAccelx (rotated into inertial frame)
        self.xhat[4,0] = self.xhat[4,0] + self.dt*vydot # vy = vy + deltaT*measuredAccely (rotated into inertial frame)
        self.xhat[5,0] = self.xhat[5,0] + self.dt*(vzdot + 9.8066) # .. Do not forget Gravity (9.801 m/s2) -> 9.8066 is more exact
        self.xhat[6,0] = self.xhat[6,0] + self.dt*q1dot # ..
        self.xhat[7,0] = self.xhat[7,0] + self.dt*q2dot # ..
        self.xhat[8,0] = self.xhat[8,0] + self.dt*q3dot # ..
        self.xhat[9,0] = self.xhat[9,0] + self.dt*q4dot # ..

        print("x ekf: ", self.xhat[0,0])
        print("y ekf: ", self.xhat[1,0])
        print("z ekf: ", self.xhat[2,0])
        
        # Extract and normalize the quat    
        self.quat = np.array([[self.xhat[6,0], self.xhat[7,0], self.xhat[8,0], self.xhat[9,0]]]).T
        # .. Normailize quat
        magnitude = np.sqrt(self.quat[0,0]**2 + self.quat[1,0]**2 + self.quat[2,0]**2 + self.quat[3,0]**2)
        self.quat = self.quat/magnitude # code here. Uncomment this line
        
        #re-assign quat
        self.xhat[6,0] = self.quat[0,0] # Need to double index b/c that is how it is defined from some reason
        self.xhat[7,0] = self.quat[1,0]
        self.xhat[8,0] = self.quat[2,0]
        self.xhat[9,0] = self.quat[3,0]

        self.q1 = self.quat[0,0]
        self.q2 = self.quat[1,0]
        self.q3 = self.quat[2,0]
        self.q4 = self.quat[3,0]
        
                
        # Now write out all the partials to compute the transition matrix Phi
        #delV/delQ
        # Triple check this
        Fvq = self.dfvq(self.quat[0,0], self.quat[1,0], self.quat[2,0], self.quat[3,0], self.fx, self.fy , self.fz)
        
        #delV/del_abias -> using the given rotation matrix
        
        Fvb = -self.R_bi
        
        #delQ/delQ
        Fqq = -0.5*omegaw# using the omega from above
     
        #delQ/del_gyrobias
        Fqb = self.fqb(self.quat[0,0], self.quat[1,0], self.quat[2,0], self.quat[3,0]) # ..


        # Now assemble the Transition matrix A
        A = self.createA(fvq=Fvq, fqq= Fqq, fqb=Fqb, fvb= Fvb) 
        
        #Propagate the error covariance matrix, I suggest using the continuous integration since Q, R are not discretized 
        #Pdot = A@P+P@A.transpose() + Q
        #P = P +Pdot*dt

        # -> this is the continuous time case
        Pdot = A@self.P+self.P@A.T + self.Q # maybe update self.pdot here, don't see a need to though
        self.P = self.P + self.dt*Pdot #
        
        #Correction step
        #Get measurements 3 positions and 3 velocities from GPS
        self.z = np.array([[self.measure[6], self.measure[7], self.measure[8], self.measure[9], self.measure[10], self.measure[11]]]).T #x y z vx vy vz


        # PRETTY SURE I HAVE TO CORRECT TO INERTIAL FRAME? See rgps stuff in slides
        #Write out the measurement matrix linearization to get H
        
        # del v/del q
        
        # not sure if P and Q here should be with biases subtracted or not
        Hvq = self.hvq(self.rgps[0], q1 = self.quat[0,0], q2 = self.quat[1,0], q3 = self.quat[2,0], q4 = self.quat[3,0], Q = self.q, R = self.r) # angular rates, not other shit
        
        #del P/del q
        Hxq = self.hxq(self.rgps[0], q1 = self.quat[0,0], q2 = self.quat[1,0], q3 = self.quat[2,0], q4 = self.quat[3,0])
        
        # Assemble H
        H = self.createH(hvq=Hvq, hxq=Hxq) # ..

        #Compute Kalman gain
        
        L =  self.P@H.T@np.linalg.inv(H@self.P@H.T+self.R)
        
        #Perform xhat correction    xhat = xhat + L@(z-H@xhat)
        self.xhat = self.xhat + L@(self.z-H@self.xhat) # .. uncomment
        
        #propagate error covariance approximation P = (np.eye(16,16)-L@H)@P
        
        self.P = (np.eye(16,16) - L@H)@self.P # ..

        #Now let us do some book-keeping 
        # Get some Euler angles
        
        phi, theta, psi = quat2euler(self.quat.T)

        self.PHI.append(phi*180/math.pi)
        self.THETA.append(theta*180/math.pi)
        self.PSI.append(psi*180/math.pi)
    
          
        # Saving data for the plots. Uncomment the 4 lines below once you have finished the ekf function

        DP = np.diag(self.P)
        self.P_R.append(DP[0:3])
        self.P_R1.append(DP[3:6])
        self.P_R2.append(DP[6:10])
        self.Pos.append(self.xhat[0:3].T[0])
        self.POS_X.append(self.xhat[0,0])
        self.POS_Y.append(self.xhat[1,0])
        self.Vel.append(self.xhat[3:6].T[0])
        self.Quater.append(self.xhat[6:10].T[0])
        self.Bias.append(self.xhat[10:16].T[0])
        B = self.measure[6:9].T
        self.measure_PosX.append(B[0])
        self.measure_PosY.append(B[1])
        self.measure_PosZ.append(B[2])

        self.P_angular.append(self.p)
        self.Q_angular.append(self.q)
        self.R_angular.append(self.r)

        self.loop_t += 1
        self.time.append(self.loop_t*self.dt)

    def omega(self, p,q,r):
        return np.array([
            [0,p,q,r],
            [-p,0,-r,q],
            [-q,r,0,-p],
            [-r,-q,p,0]
        ])
    
    def dfvq(self, q1, q2, q3, q4,ax, ay, az):
        return np.array([
            [2*(q1*ax+q4*ay-q3*az),2*(q2*ax+q3*ay+q4*az), 2*(-q3*ax+q2*ay+q1*az), 2*(-q4*ax-q1*ay+q2*az)],
            [2*(q4*ax + q1*ay-q2*az), 2*(q3*ax - q2*ay - q1*az), 2*(q2*ax+q3*ay+q4*az), 2*(q1*ax-q4*ay+q3*az)],
            [2*(-q3*ax+q2*ay+q1*az), 2*(q4*ax + q1*ay - q2*az), 2*(-q1*ax + q4*ay - q3*az), 2*(q2*ax + q3*ay + q4*az)]
        ])

    def Rot(self, q1,q2,q3,q4):
        return np.array([
            [(q1**2+q2**2-q3**2-q4**2), 2*(q2*q3+q1*q4), 2*(q2*q4-q1*q3)],
            [2*(q1*q3-q1*q4), (q1**2 - q2**2 + q3**2 - q4**2), 2*(q3*q4+q1*q2)],
            [2*(q2*q4+q1*q3), 2*(q3*q4 - q1*q2), (q1**2 - q2**2 - q3**2 + q4**2)]
        ])

    def fqb(self, q1,q2,q3,q4):
        return 0.5*np.array([
            [q2,q3,q4],
            [-q1,q4,-q3],
            [-q4,-q1,q2],
            [q3,-q2,-q1]
        ])
    
    
    def createA(self, fvq, fqq, fqb, fvb):
        # cols are actually more than cols, they are just the sets of column I am breaking A into
        col1 = np.vstack([self.Z, self.Z, self.Z43, self.Z, self.Z])
        col2 = np.vstack([self.I, self.Z, self.Z43, self.Z, self.Z])
        col3 = np.vstack([self.Z34, fvq, fqq, self.Z34, self.Z34])
        col4 = np.vstack([self.Z, self.Z, fqb, self.Z, self.Z])
        col5 = np.vstack([self.Z, fvb, self.Z43, self.Z, self.Z])

        A = np.hstack([col1, col2, col3, col4, col5])
        # print("col1 ",col1)
        # print("col2 ",col2)
        # print("col3 ",col3)
        # print("col4 ",col4)
        # print("col5 ",col5)

        # print ("A is: ", A)
        return A

    def hxq(self, rgps1, q1, q2, q3, q4):
        return np.array([
            [-rgps1*2*q1, -rgps1*2*q2, rgps1*2*q3, rgps1*2*q4],
            [-rgps1*2*q4, -rgps1*2*q3, -rgps1*2*q2, -rgps1*2*q1],
            [rgps1*2*q3, -rgps1*2*q4, rgps1*2*q1, -rgps1*2*q2]
        ])
    
    def hvq(self, rgps1, q1, q2, q3, q4, Q, R):
    
        return np.array([
            [rgps1*2*q3*Q + rgps1*2*q4*R,   rgps1*2*q4*Q - rgps1*2*q3*R,   rgps1*2*q1*Q - rgps1*2*q2*R,   rgps1*2*q2*Q + rgps1*2*q1*R],
            [-rgps1*2*q2*Q - rgps1*2*q1*R,   rgps1*2*q2*R - rgps1*2*q1*Q,   rgps1*2*q4*Q - rgps1*2*q3*R,   rgps1*2*q3*Q + rgps1*2*q4*R],
            [rgps1*2*q1*Q - rgps1*2*q2*R,   -rgps1*2*q2*Q - rgps1*2*q1*R,   -rgps1*2*q3*Q - rgps1*2*q4*R,   rgps1*2*q4*Q - rgps1*2*q3*R]
        ])

    def createH(self, hvq, hxq):
        col1 = np.vstack([self.I, self.Z])
        col2 = np.vstack([self.Z, self.I])
        col3 = np.vstack([hxq, hvq])
        col4 = np.vstack([self.Z36, self.Z36])
        
        H = np.hstack([col1, col2, col3, col4])
        return H


    def plot_data_callback(self):

        plt.figure(1)
        plt.clf()
        plt.plot(self.time,self.PHI,'b', self.time, self.THETA, 'g', self.time,self.PSI, 'r')
        plt.legend(['phi','theta','psi'])
        plt.title('Phi, Theta, Psi [deg]')

        plt.figure(2)
        plt.clf()
        plt.plot(self.measure_PosX, self.measure_PosY, self.POS_X, self.POS_Y)
        plt.title('xy trajectory')
        plt.legend(['GPS','EKF'])

        plt.figure(3)
        plt.clf()
        plt.plot(self.time,self.P_R)
        plt.title('Covariance of Position')
        plt.legend(['px','py','pz'])
        # plt.figure(4)
        # plt.clf()
        # plt.plot(self.time,self.P_R1)
        # plt.legend(['pxdot','pydot','pzdot'])
        # plt.title('Covariance of Velocities')
        # plt.figure(5)
        # plt.clf()
        # plt.plot(self.time,self.P_R2)
        # plt.title('Covariance of Quaternions')
        # plt.figure(6)
        # plt.clf()
        # plt.plot(self.time,self.Pos,self.time,self.measure_PosX,'r:', self.time,self.measure_PosY,'r:', self.time,self.measure_PosZ,'r:')
        # plt.legend(['X_ekf', 'Y_ekf', 'Z_ekf','Xgps','Ygps','Z_0'])
        # plt.title('Position')
        # plt.figure(7)
        # plt.clf()
        # plt.plot(self.time,self.Vel)
        # plt.title('vel x y z')
        # plt.figure(8)
        # plt.clf()
        # plt.plot(self.time,self.Quater)
        # plt.title('Quat')
        # plt.figure(9)
        # plt.clf()
        # plt.plot(self.time,self.P_angular,self.time,self.Q_angular,self.time,self.R_angular)
        # plt.title('OMEGA with Bias')
        # plt.legend(['p','q','r'])

        plt.figure(10)
        plt.clf()
        plt.plot(self.time,self.Bias)
        plt.title('Gyroscope and accelerometer Bias')
        plt.legend(['bp','bq','br','bfx','bfy','bfz'])
                
        plt.ion()
        plt.show()
        plt.pause(0.0001)
        

def main(args=None):
    rclpy.init(args=args)

    ekf_node = ExtendedKalmanFilter()
    
    rclpy.spin(ekf_node)

   
    ekf_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
