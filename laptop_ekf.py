"""
Copyright (c) 2023 The uos_sess6072_build Authors.
Authors: Miquel Massot, Blair Thornton, Sam Fenton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import numpy as np
import argparse
from datetime import datetime, timezone
import time
from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate
from math_feeg6043 import Vector, Matrix, Identity, Inverse, eigsorted, gaussian, l2m
from model_feeg6043 import ActuatorConfiguration
from math_feeg6043 import Vector
from model_feeg6043 import rigid_body_kinematics
from model_feeg6043 import RangeAngleKinematics
from model_feeg6043 import TrajectoryGenerate
from math_feeg6043 import l2m
from model_feeg6043 import feedback_control
from math_feeg6043 import Inverse, HomogeneousTransformation
import numpy as np
from model_feeg6043 import extended_kalman_filter_predict, extended_kalman_filter_update


class LaptopPilot:
    # def __init__(self, simulation, self.kg, self,kn, self.tau_s):
    def __init__(self, simulation):

        # network for sensed pose
        aruco_params = {
            "port": 50000,  # Port to listen to (DO NOT CHANGE)
            "marker_id": 25,  # Marker ID to listen to (CHANGE THIS to your marker ID)            
        }
        self.robot_ip = "192.168.90.1"
        
        # handles different time reference, network amd aruco parameters for simulator
        self.sim_time_offset = 0 #used to deal with webots timestamps
        self.sim_init = False #used to deal with webots timestamps
        self.simulation = simulation
        if self.simulation:
            self.robot_ip = "127.0.0.1"          
            aruco_params['marker_id'] = 0  #Ovewrites Aruco marker ID to 0 (needed for simulation)
            self.sim_init = True #used to deal with webots timestamps

        print("Connecting to robot with IP", self.robot_ip)
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)

        ############# INITIALISE ATTRIBUTES ##########       

        self.velocity = 0.075  # velocity in m/s
        self.acceleration = 0.1 # acceleration in m/s^2
        self.turning_radius = 0.2 # turning radius in meters
        self.acceptance_radius = 0.1  # acceptance radius in meters

        self.t_prev = 0  # previous time
        self.t = 0 #elapsed time

        # modelling parameters
        wheel_distance = 0.16  #m wheel seperation to centreline
        wheel_diameter = 0.074 #m wheel diameter
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter) #look at your tutorial and see how to use this class
        
        self.initialise_pose = True # False once the pose is initialised  

        # waypoint for octagon
        # self.northings_path = [0.5, 1, 1.5, 1.5, 1, 0.5, 0.25, 0.25]
        # self.eastings_path = [0.25, 0.25, 0.75, 1.25, 1.65, 1.65, 1.25, 0.75]

        self.northings_path = [0.25, 1.6, 1.6, 0.25, 0.25]
        self.eastings_path = [0.3, 0.3, 1.5, 1.5, 0.3]
        self.relative_path = False # False if you want it to be absolute , True it will offset based on 1st point but the same shape   

        self.path = TrajectoryGenerate(self.northings_path, self.eastings_path ) #initialise the path
   
        self.tau_s = 0.3 # s to remove along track error. ideal = 0.3
        self.L = 0.15  # m distance to remove normal and angular error , ideal = 0.1
        self.v_max = 0.2  # fastest the robot can go
        self.w_max = np.deg2rad(30)  # fastest the robot can turn

        self.k_s = 1 / self.tau_s  # ks
        self.k_n = 0.1  # kn
        self.k_g = 0.1 # kg

        self.initialise_control = True  # False once control gains are initialized, dont change

        #EKF filters 
        self.N = 0
        self.E = 1
        self.G = 2
        self.DOTX = 3
        self.DOTG = 4

        self.state = Vector(5)

        # initial state, start from 0 
        self.init_state = Vector(5)

        # initial , noise, 
        self.init_covariance = Identity(5) 
        self.R = Identity(5) 

        #self define uncertainty
        self.NE_std = [0.1, 0.1]  # northings and eastings standard deviation
        self.G_std = [np.deg2rad(1)]  # yaw standard deviation
        self.DOTX_std = [0.01]  # velocity standard deviation
        self.DOTG_std = [np.deg2rad(0)]  # angular rate standard deviation

        # measurement noise
        self.R_N = [0.1]
        self.R_E = [0.1]
        self.R_G = [np.deg2rad(0.5)]
        self.R_DOTX = [0]
        self.R_DOTG = [np.deg2rad(0.5)]

        self.z = Vector(5) #measurement vector
        self.Q = Identity(5) #measurement noise
        self.H = Matrix(5,5)

        # model pose
        self.est_pose_northings_m = 0
        self.est_pose_eastings_m = 0
        self.est_pose_yaw_rad = 0

        # measured pose
        self.measured_pose_timestamp_s = None
        self.measured_pose_northings_m = None
        self.measured_pose_eastings_m = None
        self.measured_pose_yaw_rad = None

        # wheel speed commands
        self.cmd_wheelrate_right = None
        self.cmd_wheelrate_left = None 

        # encoder/actual wheel speeds
        self.measured_wheelrate_right = None
        self.measured_wheelrate_left = None   

        # lidar
        self.lidar_timestamp_s = None
        self.lidar_data = None

        lidar_xb = 0.03 # location of lidar centre in b-frame primary axis, m
        lidar_yb = 0.03 # location of lidar centre in b-frame secondary axis, m 
        self.lidar = RangeAngleKinematics(lidar_xb,lidar_yb) 
        ###############################################################        

        self.datalog = DataLogger(log_dir="logs")

        # Wheels speeds in rad/s are encoded as a Vector3 with timestamp, 
        # with x for the right wheel and y for the left wheel.        
        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.robot_ip
        )

        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds",Vector3Stamped, self.true_wheel_speeds_callback,ip=self.robot_ip,
        )
        self.lidar_sub = Subscriber(
            "/lidar", LaserScan, self.lidar_callback, ip=self.robot_ip
        )
        self.groundtruth_sub = Subscriber(
            "/groundtruth", Pose, self.groundtruth_callback, ip=self.robot_ip
        )
                    
    def true_wheel_speeds_callback(self, msg):
        print("Received sensed wheel speeds: R=", msg.vector.x,", L=", msg.vector.y)
        # update wheel rates
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        # This is a callback function that is called whenever a message is received        
        print("Received lidar message", msg.header.seq)
        
        if self.sim_init == True:
            self.sim_time_offset = datetime.now(timezone.utc).timestamp() - msg.header.stamp
            self.sim_init = False     

        msg.header.stamp += self.sim_time_offset

        self.lidar_timestamp_s = msg.header.stamp  # we want the lidar measurement timestamp here

        # b to e frame
        p_eb = Vector(3)
        p_eb[0] = self.measured_pose_northings_m  # robot pose northings (see Task 3)
        p_eb[1] = self.measured_pose_eastings_m  # robot pose eastings (see Task 3)
        p_eb[2] = self.measured_pose_yaw_rad  # robot pose yaw (see Task 3)

        # m to e frame
        self.lidar_data = np.zeros((len(msg.ranges), 2))  # specify length of the lidar data

        z_lm = Vector(2)
        # for each map measurement
        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]

            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)  # see tutorial

            self.lidar_data[i, 0] = t_em[0]
            self.lidar_data[i, 1] = t_em[1]

        # this filters out any NaN values
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]
        self.datalog.log(msg, topic_name="/lidar")

    def true_wheel_speeds_callback(self, msg):
        # print("Received sensed wheel speeds: R=", msg.vector.x, ", L=", msg.vector.y)

        # update wheel rates
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def groundtruth_callback(self, msg):
        """This callback receives the odometry ground truth from the simulator."""
        self.datalog.log(msg, topic_name="/groundtruth")
    
    def pose_parse(self, msg, aruco = False):
        # parser converts pose data to a standard format for logging
        time_stamp = msg[0]
        if aruco == True:
            if self.sim_init == True:
                self.sim_time_offset = datetime.now(timezone.utc).timestamp()-msg[0]
                self.sim_init = False                                         
                
            # self.sim_time_offset is 0 if not a simulation. Deals with webots dealing in elapse timeself.sim_time_offset
            # print(
            #     "Received position update from",
            #     datetime.now(timezone.utc).timestamp() - msg[0] - self.sim_time_offset,
            #     "seconds ago",
            # )
            time_stamp = msg[0] + self.sim_time_offset                

        pose_msg = PoseStamped() 
        pose_msg.header = Header()
        pose_msg.header.stamp = time_stamp
        pose_msg.pose.position.x = msg[1]
        pose_msg.pose.position.y = msg[2]
        pose_msg.pose.position.z = 0

        quat = Quaternion()        
        if self.simulation == False and aruco == True: quat.from_euler(0, 0, np.deg2rad(msg[6]))
        else: quat.from_euler(0, 0, msg[6])
        pose_msg.pose.orientation = quat        
        return pose_msg

    def generate_trajectory(self):
        # pick waypoints as current pose relative or absolute northings and eastings
        if self.relative_path == True:

            for i in range(len(self.northings_path)):
                self.northings_path[i] += self.measured_pose_northings_m #offset by current northings
                self.eastings_path[i] += self.measured_pose_eastings_m  #offset by current eastings

            # convert path to matrix and create a trajectory class instance
            # C = l2m([self.northings_path, self.eastings_path])       
            self.path = TrajectoryGenerate(self.northings_path,self.eastings_path)     
            
            # set trajectory variables (velocity, acceleration and turning arc radius)
            self.path.path_to_trajectory(self.velocity, self.acceleration) #velocity and acceleration
            self.path.turning_arcs(self.turning_radius) #turning radius
            self.path.wp_id = 0 #initialises the next waypoint

            # self.path.t_complete = np.nan # will log when the trajectory was complete 
            print('Trajectory wp timestamps\n',self.path.Tp_arc,'s')

    def motion_model(self, state, u, dt): #input state gamma in rad
        # k -1 state
        N_k_1 = state[self.N]
        E_k_1 = state[self.E]
        G_k_1 = state[self.G]
        DOTX_k_1 = state[self.DOTX]
        DOTG_k_1 = state[self.DOTG]

        p_robot = Vector(3)
        p_robot[0] = N_k_1    # northings
        p_robot[1] = E_k_1    # eastings
        p_robot[2] = G_k_1    # yaw rad
         
        p_robot = rigid_body_kinematics(p_robot, u, dt) #est robot position aftter twist, t+1 , INPUT probot-rad, , output rad/s
        # p_robot[2] = p_robot[2] % (2 * np.pi)

        state = np.vstack((p_robot, u))

        # updated state after rigid body kinematics
        N_k = state[self.N]
        E_k = state[self.E]
        G_k = state[self.G]
        DOTX_k = state[self.DOTX]
        DOTG_k = state[self.DOTG]

        # Compute its jacobian
        F = Identity(5)    
        
        # jacobian matrix Fk when the angular rate w is zero
        if abs(DOTG_k) < 1E-2:  # caters for zero angular rate, but uses a threshold to avoid numerical instability
            F[self.N, self.G] = -DOTX_k * dt * np.sin(G_k_1)
            F[self.N, self.DOTX] = dt * np.cos(G_k_1)
            F[self.E, self.G] = DOTX_k * dt * np.cos(G_k_1)
            F[self.E, self.DOTX] = dt * np.sin(G_k_1)
            F[self.G, self.DOTG] = dt        
            
        else:  # original jacobian matrix Fk
            F[self.N, self.G] = (DOTX_k / DOTG_k) * (np.cos(G_k) - np.cos(G_k_1))
            F[self.N, self.DOTX] = (1 / DOTG_k) * (np.sin(G_k) - np.sin(G_k_1))
            F[self.N, self.DOTG] = (DOTX_k / (DOTG_k**2)) * (np.sin(G_k_1) - np.sin(G_k)) + (DOTX_k * dt / DOTG_k) * np.cos(G_k)
            F[self.E, self.G] = (DOTX_k / DOTG_k) * (np.sin(G_k) - np.sin(G_k_1))
            F[self.E, self.DOTX] = (1 / DOTG_k) * (np.cos(G_k_1) - np.cos(G_k))
            F[self.E, self.DOTG] = (DOTX_k / (DOTG_k**2)) * (np.cos(G_k) - np.cos(G_k_1)) + (DOTX_k * dt / DOTG_k) * np.sin(G_k)
            F[self.G, self.DOTG] = dt

        return state, F  #gamma in angle 
    
    def h_update(self, x):
        self.z[self.N] = x[self.N] 
        self.z[self.E] = x[self.E]
        self.z[self.G] = x[self.G]
        self.H[self.N,self.N] = 1
        self.H[self.E,self.E] = 1
        self.H[self.G,self.G] = 1
        return self.z, self.H
    
    # def h_g_update(self, x):

    #     self.z[self.G] = x[self.G]
    #     self.H[self.G,self.G] = 1
    #     return self.z, self.H
    
    # def h_ne_update(self, x):
    #     self.z[self.N] = x[self.N] 
    #     self.z[self.E] = x[self.E]
    #     self.H[self.N,self.N] = 1
    #     self.H[self.E,self.E] = 1
    #     return self.z, self.H

    def run(self, time_to_run=-1):
        self.start_time = datetime.now(timezone.utc).timestamp()
        
        try:
            r = Rate(10.0)
            while True:
                current_time = datetime.now(timezone.utc).timestamp()
                if time_to_run > 0 and current_time - self.start_time > time_to_run:
                    print("Time is up, stopping…")
                    break
                self.infinite_loop()
                r.sleep()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception: ", e)
        finally:
            self.lidar_sub.stop()
            self.groundtruth_sub.stop()
            self.true_wheel_speed_sub.stop()


    def infinite_loop(self):
        """Main control loop

        Your code should go here.
        """
                            
        aruco_pose = self.aruco_driver.read()    

        if aruco_pose is not None:
            
            # <code that parses aruco and logs the topic>
            msg = self.pose_parse(aruco_pose, aruco = True)

            # reads sensed pose for local use
            self.measured_pose_timestamp_s = msg.header.stamp
            self.measured_pose_northings_m = msg.pose.position.x
            self.measured_pose_eastings_m = msg.pose.position.y
            _, _, self.measured_pose_yaw_rad = msg.pose.orientation.to_euler()        
            self.measured_pose_yaw_rad = self.measured_pose_yaw_rad % (np.pi*2) # manage angle wrapping

            self.datalog.log(msg, topic_name="/aruco")

            ###### wait for the first sensor info to initialize the pose ######
            if self.initialise_pose == True: #this part is for initialising 

                self.est_pose_northings_m = self.measured_pose_northings_m
                self.est_pose_eastings_m = self.measured_pose_eastings_m
                self.est_pose_yaw_rad = self.measured_pose_yaw_rad

                # get current time and determine timestep
                self.t_prev = datetime.now(timezone.utc).timestamp() #initialise the time
                self.t = 0 #elapsed time
                time.sleep(0.1) #wait for approx a timestep before proceeding
                
                self.generate_trajectory()

                #setup for EKF
                self.init_state[self.N,0] = self.est_pose_northings_m
                self.init_state[self.E,0] = self.est_pose_eastings_m
                self.init_state[self.G,0] = self.est_pose_yaw_rad  #rad/s
                self.init_state[self.DOTX] = 0
                self.init_state[self.DOTG] = 0

                self.init_covariance[self.N,self.N] = self.NE_std[0]**2
                self.init_covariance[self.E, self.E] = self.NE_std[1]**2
                self.init_covariance[self.G, self.G] = self.G_std[0]**2
                self.init_covariance[self.DOTX, self.DOTX] = self.DOTX_std[0]**2
                self.init_covariance[self.DOTG, self.DOTG] = self.DOTG_std[0]**2

                self.R[self.N, self.N] = self.R_N[0]**2
                self.R[self.E, self.E] = self.R_E[0]**2
                self.R[self.G, self.G] = self.R_G[0]**2
                self.R[self.DOTX, self.DOTX] = self.R_DOTX[0]**2
                self.R[self.DOTG, self.DOTG] = self.R_DOTG[0]**2

                self.state = self.init_state
                self.covariance = self.init_covariance

                self.initialise_pose = False 

        if self.initialise_pose != True:  

            #determine the time step
            t_now = datetime.now(timezone.utc).timestamp()   
            dt = t_now - self.t_prev #timestep from last estimate
            self.t += dt #add to the elapsed time
            self.t_prev = t_now #update the previous timestep for the next loop

            # convert true wheel speeds into twist
            q = Vector(2) #twist, m/s, rad/s
            q[0] = self.measured_wheelrate_right  # wheel rate rad/s (measured)
            q[1] = self.measured_wheelrate_left  # wheel rate rad/s (measured)
            
            u = self.ddrive.fwd_kinematics(q) # m/s, rads/s

            # This is overriding your EKF state with raw measurements
            self.state[self.N] = self.measured_pose_northings_m
            self.state[self.E] = self.measured_pose_eastings_m
            self.state[self.G] = self.measured_pose_yaw_rad
            self.state[self.DOTG] = u[0] #rad/s
            self.state[self.DOTX] = u[1] #m/s

            print('before state = ',self.state)
            # Return the predicted state and the covariance, 
            self.state, self.covariance = extended_kalman_filter_predict(self.state, self.covariance, u, self.motion_model, self.R, dt)
            self.h = self.h_update

            self.z[self.N] = self.measured_pose_northings_m  
            self.z[self.E] = self.measured_pose_eastings_m
            self.z[self.G] = self.measured_pose_yaw_rad

            self.Q[self.N, self.N] = self.NE_std[0]**2  
            self.Q[self.E, self.E] = self.NE_std[1]**2  
            self.Q[self.G, self.G] = self.G_std[0]**2

            self.state, self.covariance = extended_kalman_filter_update(self.state, self.covariance, self.z, self.h, self.Q, wrap_index=self.G)      

            print('after state = ',self.state)
            p_robot = Vector(3)
            p_robot[0,0] = self.state[self.N, 0]
            p_robot[1,0] = self.state[self.E, 0]
            p_robot[2,0] = self.state[self.G, 0]
            p_robot[2,0] = p_robot[2] % (2 * np.pi)
            
            # self.est_pose_northings_m = p_robot[0,0]
            # self.est_pose_eastings_m = p_robot[1,0]
            # self.est_pose_yaw_rad = p_robot[2,0]

            #################### Trajectory sample #################################    
            # feedforward control: check wp progress and sample reference trajectory
            self.path.path_to_trajectory(self.velocity, self.acceleration)
            self.path.turning_arcs(self.turning_radius)
            self.path.wp_progress(self.t, p_robot, self.acceptance_radius)

            p_ref, u_ref = self.path.p_u_sample(self.t)  # sample the path at the current elapsed time

            # feedback control: get pose change to desired trajectory from body
            dp = p_ref - p_robot  # compute difference between reference and estimated pose in the e-frame
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi # handle angle wrapping for yaw
            H_eb = HomogeneousTransformation(p_robot[0:2], p_robot[2])
            ds = Inverse(H_eb.H_R) @ dp    
            # compute control gains for the initial condition (where the robot is stationary)
            
            self.k_s = 1 / self.tau_s  # ks

            if self.initialise_control == True:
                self.k_n = 2 * u_ref[0] / self.L**2  # kn
                self.k_g = u_ref[0] / self.L  # kg
                self.initialise_control = False  # maths changes a bit after the first iteration
            
            # update the controls
            du = feedback_control(ds, self.k_s, self.k_n, self.k_g)

            u = u_ref + du 

            # update control gains for the next timestep
            self.k_n = 2 * u[0] / self.L**2  # kn
            self.k_g = u[0] / self.L  # kg
            # print('kn kg = ',self.k_n, self.k_g)

            # ensure within performance limitation
            if u[0] > self.w_max: 
                u[0] = self.w_max
                print(f'w_max reached {u[0]}')

            if u[0] < -self.w_max: 
                u[0] = -self.w_max
                print(f'-w_max reached {u[0]}')

            if u[1] > self.v_max: 
                u[1] = self.v_max
                print(f'v_max reached {u[1]}')

            if u[1] < -self.v_max: 
                u[1] = -self.v_max
                print(f'-v_max reached {u[1]}')
            
            self.est_pose_northings_m = p_ref[0,0]
            self.est_pose_eastings_m = p_ref[1,0]
            self.est_pose_yaw_rad = p_ref[2,0]

            # self.est_pose_northings_m = self.state[self.N, 0]
            # self.est_pose_eastings_m = self.state[self.E, 0]
            # self.est_pose_yaw_rad = self.state[self.G, 0] % (2 * np.pi)

            # # logs the data             
            msg = self.pose_parse([datetime.now(timezone.utc).timestamp(),self.est_pose_northings_m,self.est_pose_eastings_m,0,0,0,self.est_pose_yaw_rad])
            self.datalog.log(msg, topic_name="/aruco")
            self.datalog.log(msg, topic_name="/est_pose")

            # p_robot = rigid_body_kinematics(p_robot,u,dt)
            # p_robot[2] = p_robot[2] % (2*np.pi)
            # actuator commands
            q = self.ddrive.inv_kinematics(u)

            # # update laptop.py
            # p_robot = rigid_body_kinematics(p_robot, u, dt) #est robot position aftter twist
            # p_robot[2] = p_robot[2] % (2 * np.pi)            
            # self.est_pose_northings_m = p_robot[0,0]
            # self.est_pose_eastings_m = p_robot[1,0]
            # self.est_pose_yaw_rad = p_robot[2,0]
            
            # print('q = ', q)
            
            wheel_speed_msg = Vector3Stamped()
            wheel_speed_msg.vector.x = q[0, 0]  # Right wheelspeed rad/s
            wheel_speed_msg.vector.y = q[1, 0]  # Left wheelspeed rad/s
        
            self.cmd_wheelrate_right = wheel_speed_msg.vector.x
            self.cmd_wheelrate_left = wheel_speed_msg.vector.y
     
            self.wheel_speed_pub.publish(wheel_speed_msg)
            self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--time",
        type=float,
        default=-1,
        help="Time to run an experiment for. If negative, run forever.",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False",
    )

    args = parser.parse_args()

    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)
