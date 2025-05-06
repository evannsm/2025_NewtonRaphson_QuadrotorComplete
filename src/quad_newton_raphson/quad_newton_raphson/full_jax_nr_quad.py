import os
def is_conda_env_activated():
   """Checks if a conda environment is activated."""
   return 'CONDA_DEFAULT_ENV' in os.environ

def get_conda_env():
   """Gets the currently activated conda environment name."""
   return os.environ.get('CONDA_DEFAULT_ENV', None)

if not is_conda_env_activated():
   # print("Please set up and activate the conda environment.")
   # exit(1)
   raise EnvironmentError("Please set up and activate the conda environment.")

elif get_conda_env() != 'wardiNN':
   # print("Conda is activated but not the 'wardiNN' environment. Please activate the 'wardiNN' conda environment.")
   # exit(1)
   raise EnvironmentError("I can see conda is activated but not the 'wardiNN' environment. Please activate the 'wardiNN' conda environment.")

else:
   print("I can see that conda environment 'wardiNN' is activated!!!!")
   print("Ok you're all set :)")
   import sys
   sys.path.append('/home/factslabegmc/miniconda3/envs/wardiNN')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint, VehicleCommand, VehicleStatus, VehicleOdometry, TrajectorySetpoint, RcChannels
from std_msgs.msg import Float64MultiArray

import sympy as smp
import math as m
import scipy.integrate as sp_int
import scipy.linalg as sp_linalg
import numpy as np
import jax.numpy as jnp
from numpy.linalg import norm


from .full_jax_nr_quad_utilities import NR_tracker_original, NR_tracker_optim, predict_output, get_inv_jac_pred_u, NR_tracker_linpred, NR_tracker_flat

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

import time
import ctypes
from transforms3d.euler import quat2euler

from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain
from pyJoules.energy_meter import EnergyContext
from scipy.spatial.transform import Rotation as R

import sys
import traceback
from .Logger import Logger

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""
    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')
        self.mocap_k = -1
        self.full_rotations = 0
        self.made_it = 0

        # test = [-0.7993715405464172, -0.5975781679153442, 0.03904560208320618, 0.04879523441195488]
        # self.roll, self.pitch, yaw = self.xeuler_from_quaternion(*test)
        # print(f"{self.roll = }")
        # print(f"{self.pitch = }")
        # print(f"{yaw = }")
        # exit(0)
###############################################################################################################################################
        self.flat_normal = True
        # Figure out if in simulation or hardware mode to set important variables to the appropriate values
        self.sim = bool(int(input("Are you using the simulator? Write 1 for Sim and 0 for Hardware: ")))
        print(f"{'SIMULATION' if self.sim else 'HARDWARE'}")
        self.double_speed = bool(int(input("Double Speed Trajectories? Press 1 for Yes and 0 for No: ")))

        self.ctrl_loop_time_log = []
        self.x_log, self.y_log, self.z_log, self.yaw_log = [], [], [], []
        self.throttle_log, self.roll_log, self.pitch_log, self.yaw_rate_log = [], [], [], []
        self.ref_x_log, self.ref_y_log, self.ref_z_log, self.ref_yaw_log = [], [], [], []
        self.nr_timel_array = []
        self.pred_timel_array = []
        self.ctrl_callback_timel_log = []
        self.vnorm_log = []
        self.vnorm = 0

        self.mode_channel = 5
        self.pyjoules_on = 0# int(input("Use PyJoules? 1 for Yes 0 for No: ")) #False
        # if self.pyjoules_on:
        #     self.csv_handler = CSVHandler('nr_aggressive.log')
###############################################################################################################################################

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


        # Create Publishers
        # Publishers for Setting to Offboard Mode and Arming/Diasarming/Landing/etc
        self.offboard_control_mode_publisher = self.create_publisher( #publishes offboard control heartbeat
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher( #publishes vehicle commands (arm, offboard, disarm, etc)
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        # Publishers for Sending Setpoints in Offboard Mode: 1) Body Rates and Thrust, 2) Position and Yaw 
        self.rates_setpoint_publisher = self.create_publisher( #publishes body rates and thrust setpoint
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher( #publishes trajectory setpoint
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        
        # Publisher for Logging States, Inputs, and Reference Trajectories for Data Analysis
        self.state_input_ref_log_publisher_ = self.create_publisher( #publishes log of states and input
            Float64MultiArray, '/state_input_ref_log', 10)
        self.state_input_ref_log_msg = Float64MultiArray() #creates message for log of states and input
        self.deeplearningdata_msg = Float64MultiArray() #creates message for log for deep learning project
        

        # Create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription( #subscribes to odometry data (position, velocity, attitude)
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription( #subscribes to vehicle status (arm, offboard, disarm, etc)
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
    
        self.offboard_mode_rc_switch_on = True if self.sim else False #Offboard mode starts on if in Sim, turn off and wait for RC if in hardware
        self.rc_channels_subscriber = self.create_subscription( #subscribes to rc_channels topic for software "killswitch" to make sure we'd like position vs offboard vs land mode
            RcChannels, '/fmu/out/rc_channels', self.rc_channel_callback, qos_profile
        )

###############################################################################################################################################

        # Initialize variables:
        self.cushion_time = 10.0
        self.flight_time = 90.0
        self.time_before_land = self.flight_time + 2*(self.cushion_time)
        print(f"time_before_land: {self.time_before_land}")
        self.offboard_setpoint_counter = 0 #helps us count 10 cycles of sending offboard heartbeat before switching to offboard mode and arming
        self.vehicle_status = VehicleStatus() #vehicle status variable to make sure we're in offboard mode before sending setpoints

        self.T0 = time.time() # initial time of program
        self.time_from_start = time.time() - self.T0 # time from start of program initialized and updated later to keep track of current time in program
        self.first_iteration = True #boolean to help us initialize the first iteration of the program        
###############################################################################################################################################
        if self.sim:
            print("Using simulator throttle from force conversion function")
            self.MASS = 1.5 #set simulation mass from iris model sdf for linearized model calculations
            # The following 3 variables are used to convert between force and throttle commands for the iris drone defined in PX4 stack for gazebo simulation
            self.MOTOR_CONSTANT = 0.00000584 #iris gazebo simulation motor constant
            self.MOTOR_VELOCITY_ARMED = 100 #iris gazebo motor velocity when armed
            self.MOTOR_INPUT_SCALING = 1000.0 #iris gazebo simulation motor input scaling
        elif not self.sim:
            print("Using hardware throttle from force conversion function and certain trajectories will not be available")
            self.MASS = 1.75 #alternate: 1.69kg and have grav_mass = 2.10

        self.GRAVITY = 9.806 #gravity
        self.T_LOOKAHEAD = .8 #lookahead time for prediction and reference tracking in NR controller
        self.LOOKAHEAD_STEP = 0.1 #step lookahead for prediction and reference tracking in NR controller
        self.INTEGRATION_STEP = 0.01 #integration step for NR controller

###############################################################################################################################################

        # '1' for jax_reg_wardi, '2' for jax_optim_wardi, '3' for jax_pred_jac_only, '4' for linear predictor normal,
        # '5' for linear predictor with jax, '6' for flat wardi normal, '7' for flat wardi with jax
        self.ctrl_type = int(input("Enter the control type: 1 for jax_reg_wardi, 2 for jax_optim_wardi, 3 for jax_pred_jac_only, 4 for linear predictor normal, 5 for linear predictor with jax, 6 for flat wardi normal, 7 for flat wardi with jax: "))

        first_thrust = self.MASS * self.GRAVITY # Initialize first input for hover at origin
        STATE = jnp.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T
        INPUT = jnp.array([[self.MASS * self.GRAVITY, 0., 0., 0.]]).T
        ref = np.array([[0, 0, -0.8, 0]]).T
        if self.ctrl_type ==1 :
            print("Using jax_reg_wardi")
            self.u0 = np.array([[self.get_throttle_command_from_force(first_thrust), 0., 0., 0.]]).T    
        elif self.ctrl_type == 2:
            print("Using jax_optim_wardi")
            self.u0 = np.array([[self.get_throttle_command_from_force(first_thrust), 0., 0., 0.]]).T
        elif self.ctrl_type == 3:
            print("Using jax_pred_jac_only")
            self.u0 = np.array([[self.get_throttle_command_from_force(first_thrust), 0., 0., 0.]]).T
        elif self.ctrl_type == 4:
            print("Using linear predictor")
            self.C = self.observer_matrix() #Calculate Observer Matrix Needed After Predictions of all States to Get Only the States We Need in Output
            self.eAT, self.int_eATB, self.jac_inv = self.linearized_model() #Calculate Linearized Model Matrices      
            """
            Jac =array([[ 0.        ,  0.        , -0.83677867,  0.        ],
                        [ 0.        ,  0.83677867,  0.        ,  0.        ],
                        [ 0.20846906,  0.        ,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ,  0.8       ]])

            """
            """self.jac_inv=array( [[ 0.        ,  0.        ,  4.796875  ,  0.        ],
                                    [ 0.        ,  1.19505915,  0.        ,  0.        ],
                                    [-1.19505915, -0.        , -0.        , -0.        ],
                                    [ 0.        ,  0.        ,  0.        ,  1.25      ]])
            """
            self.u0 = np.array([[self.get_throttle_command_from_force(first_thrust), 0., 0., 0.]]).T
        elif self.ctrl_type == 5:
            print("Using linear predictor with jax")
            self.C = self.observer_matrix() #Calculate Observer Matrix Needed After Predictions of all States to Get Only the States We Need in Output
            self.eAT, self.int_eATB, self.jac_inv = self.linearized_model() #Calculate Linearized Model Matrices      
            self.u0 = np.array([[self.get_throttle_command_from_force(first_thrust), 0., 0., 0.]]).T
        elif self.ctrl_type == 6:
            print("Using flat wardi normal")
            self.last_yaw_ref = 0.
            self.u0 = np.array([[self.get_throttle_command_from_force(first_thrust), 0., 0., 0.]]).T
        elif self.ctrl_type == 7:
            print("Using flat wardi with jax")
            self.last_yaw_ref = 0.
            self.u0 = np.array([[self.get_throttle_command_from_force(first_thrust), 0., 0., 0.]]).T
        print(f"here - self.u0: {self.u0}\n")
        # print(f'hiii')
        # exit(0)
        self.use_quat_yaw = True

###############################################################################################################################################
    
        # self.main_traj = self.hover_ref_func
        # self.hover_type = 1
        # self.main_traj = self.circle_horz_ref_func
        # self.main_traj = self.circle_horz_spin_ref_func
        # self.main_traj = self.circle_vert_ref_func
        # self.main_traj = self.fig8_horz_ref_func
        # self.main_traj = self.fig8_vert_ref_func_short
        # self.main_traj = self.fig8_vert_ref_func_tall
        # self.main_traj = self.helix
        self.main_traj = self.helix_spin
        # self.main_traj = self.triangle
        # self.main_traj = self.sawtooth


        # Reverse lookup dictionary (function reference → name string)
        self.trajectory_dictionary = {
            self.circle_horz_ref_func: "circle_horz_ref_func",
            self.circle_horz_spin_ref_func: "circle_horz_spin_ref_func",
            self.circle_vert_ref_func: "circle_vert_ref_func",
            self.fig8_horz_ref_func: "fig8_horz_ref_func",
            self.fig8_vert_ref_func_short: "fig8_vert_ref_func_short",
            self.fig8_vert_ref_func_tall: "fig8_vert_ref_func_tall",
            self.helix: "helix",
            self.helix_spin: "helix_spin",
            self.triangle: "triangle",
            self.sawtooth: "sawtooth",
            self.hover_ref_func: "hover"
        }

        print(f"{self.main_traj = }")
        self.main_traj_name = self.trajectory_dictionary[self.main_traj]
        print(f"Selected trajectory: {self.main_traj_name}")  # Should print: "Selected trajectory: circle_horz_ref_func"


        self.metadata = np.array(['Sim' if self.sim else 'Hardware',
                                  'jax_reg_wardi' if self.ctrl_type == 1 else 'jax_optim_wardi' if self.ctrl_type == 2 else 'jax_pred_jac_only' if self.ctrl_type == 3 else 'linear predictor' if self.ctrl_type == 4 else 'linear predictor with jax' if self.ctrl_type == 5 else 'flat wardi normal' if self.ctrl_type == 6 else 'flat wardi with jax' if self.ctrl_type == 7 else 'Unknown',
                                  '2x Speed' if self.double_speed else '1x Speed',
                                  self.main_traj_name
                                  ])
        # exit(1)
###############################################################################################################################################

        #Create Function @ {1/self.offboard_timer_period}Hz (in my case should be 10Hz/0.1 period) to Publish Offboard Control Heartbeat Signal
        self.offboard_timer_period = 0.1
        self.timer = self.create_timer(self.offboard_timer_period, self.offboard_mode_timer_callback)

        # Create Function at {1/self.newton_raphson_timer_period}Hz (in my case should be 100Hz/0.01 period) to Send NR Control Input
        self.newton_raphson_timer_period = self.INTEGRATION_STEP
        self.timer = self.create_timer(self.newton_raphson_timer_period, self.newton_raphson_timer_callback)

    # The following 4 functions all call publish_vehicle_command to arm/disarm/land/ and switch to offboard mode
    # The 5th function publishes the vehicle command
    # The 6th function checks if we're in offboard mode
    # The 7th function handles the safety RC control switches for hardware
    def arm(self): #1. Sends arm command to vehicle via publish_vehicle_command function
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self): #2. Sends disarm command to vehicle via publish_vehicle_command function
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self): #3. Sends offboard command to vehicle via publish_vehicle_command function
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self): #4. Sends land command to vehicle via publish_vehicle_command function
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_vehicle_command(self, command, **params) -> None: #5. Called by the above 4 functions to send parameter/mode commands to the vehicle
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def vehicle_status_callback(self, vehicle_status): #6. This function helps us check if we're in offboard mode before we start sending setpoints
        """Callback function for vehicle_status topic subscriber."""
        # print('vehicle status callback')
        self.vehicle_status = vehicle_status

    def rc_channel_callback(self, rc_channels):
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        print('rc channel callback')
        # self.mode_channel = 5
        flight_mode = rc_channels.channels[self.mode_channel-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on = True if flight_mode >= 0.75 else False

    # The following 2 functions are used to publish offboard control heartbeat signals
    def publish_offboard_control_heartbeat_signal2(self): #1)Offboard Signal2 for Returning to Origin with Position Control
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_offboard_control_heartbeat_signal1(self): #2)Offboard Signal1 for Newton-Rapshon Body Rate Control
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)


# ~~ The remaining functions are all intimately related to the Newton-Rapshon Control Algorithm ~~
    # The following 2 functions are used to convert between force and throttle commands
    def get_throttle_command_from_force(self, collective_thrust): #Converts force to throttle command
        """ Convert the positive collective thrust force to a positive throttle command. """
        print(f"Conv2Throttle: collective_thrust: {collective_thrust}")
        if self.sim:
            motor_speed = m.sqrt(collective_thrust / (4.0 * self.MOTOR_CONSTANT))
            throttle_command = (motor_speed - self.MOTOR_VELOCITY_ARMED) / self.MOTOR_INPUT_SCALING
            print(f"conv2throttle: thrust: {throttle_command = }")
            return throttle_command
        
        if not self.sim:
            a = 0.00705385408507030
            b = 0.0807474474438391
            c = 0.0252575818743285

            # equation form is a*x + b*sqrt(x) + c = y
            throttle_command = a*collective_thrust + b*m.sqrt(collective_thrust) + c
            print(f"conv2throttle: thrust: {throttle_command = }")
            return throttle_command

    def get_force_from_throttle_command(self, throttle_command): #Converts throttle command to force
        """ Convert the positive throttle command to a positive collective thrust force. """
        print(f"Conv2Force: throttle_command: {throttle_command}")
        if self.sim:
            motor_speed = (throttle_command * self.MOTOR_INPUT_SCALING) + self.MOTOR_VELOCITY_ARMED
            collective_thrust = 4.0 * self.MOTOR_CONSTANT * motor_speed ** 2
            print(f"conv2force: force: {collective_thrust = }")
            return collective_thrust
        
        if not self.sim:
            a = 19.2463167420814
            b = 41.8467162352942
            c = -7.19353022443441

            # equation form is a*x^2 + b*x + c = y
            collective_thrust = a*throttle_command**2 + b*throttle_command + c
            print(f"conv2force: force: {collective_thrust = }")
            return collective_thrust
    
    def normalize_angle(self, angle):
        """ Normalize the angle to the range [-pi, pi]. """
        return m.atan2(m.sin(angle), m.cos(angle))
    
    def xeuler_from_quaternion(self, w, x, y, z):
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
            """
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll_x = m.atan2(t0, t1)
        
            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = m.asin(t2)
        
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw_z = m.atan2(t3, t4)
        
            return roll_x, pitch_y, yaw_z # in radians

    def adjust_yaw(self, yaw):
        mocap_psi = yaw
        self.mocap_k += 1
        psi = None
        
        if self.mocap_k == 0:
            self.prev_mocap_psi = mocap_psi
            psi = mocap_psi

        elif self.mocap_k > 0:
            # mocap angles are from -pi to pi, whereas the angle state variable in the MPC is an absolute angle (i.e. no modulus)
            # I correct for this discrepancy here
            if self.prev_mocap_psi > np.pi*0.9 and mocap_psi < -np.pi*0.9:
                # Crossed 180 deg, CCW
                self.full_rotations += 1
            elif self.prev_mocap_psi < -np.pi*0.9 and mocap_psi > np.pi*0.9:
                # Crossed 180 deg, CW
                self.full_rotations -= 1

            psi = mocap_psi + 2*np.pi * self.full_rotations
            self.prev_mocap_psi = mocap_psi
        
        return psi

    def vehicle_odometry_callback(self, msg): # Odometry Callback Function Yields Position, Velocity, and Attitude Data
        """Callback function for vehicle_odometry topic subscriber."""
        # print('vehicle odometry callback')

        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = msg.position[2]

        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]

        quat = msg.q.tolist()
        quat_xyzw = quat[1:] + [quat[0]]         # Convert to (x, y, z, w) format
        self.quat = quat_xyzw
        print(f"{self.quat = }")
        self.ROT = R.from_quat(quat_xyzw).as_matrix()
        print(f"{self.ROT = }")
        # exit(0)
        # self.roll, self.pitch, yaw = quat2euler(msg.q)
        self.roll, self.pitch, yaw = self.xeuler_from_quaternion(*msg.q)
        # self.roll, self.pitch, yaw = self.euler_from_quaternion(msg.q)
        self.yaw = self.adjust_yaw(yaw)

        # print(f"{(self.roll, self.pitch, self.yaw) = }")
        # exit(0)

        self.p = msg.angular_velocity[0]
        self.q = msg.angular_velocity[1]
        self.r = msg.angular_velocity[2]

        self.state_vector = np.array([[self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw]]).T 
        self.nr_state = np.array([[self.x, self.y, self.z, self.yaw]]).T

        if self.first_iteration:
            self.old_x = self.x
            self.old_y = self.y
            self.old_z = self.z
            self.old_roll = self.roll
            self.old_pitch = self.pitch
            self.old_yaw = self.yaw

            self.old_vx = self.vx
            self.old_vy = self.vy
            self.old_vz = self.vz
            self.old_p = self.p
            self.old_q = self.q
            self.old_r = self.r

            self.old_ax = 0
            self.old_ay = 0
            self.old_az = 0
            self.old_aroll = 0
            self.old_apitch = 0
            self.old_ayaw = 0

            self.old_jx = 0
            self.old_jy = 0
            self.old_jz = 0
            self.old_jroll = 0
            self.old_jpitch = 0
            self.old_jyaw = 0

            if self.flat_normal:
                self.x_df = np.array([[self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz, 0., 0., 0., 0., 0.]]).T
            else:
                self.x_df = np.array([[self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz, 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T
                # self.x_df = np.array([[self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T
            print(f"First iteration: set initial flat state as {self.x_df = }")
            # print(f"{self.x_df.shape = }")
            # exit(0)
        # print(f"State Vector: {self.state_vector}")
        # print(f"NR State: {self.nr_state}")

    def publish_rates_setpoint(self, thrust: float, roll: float, pitch: float, yaw: float): #Publishes Body Rate and Thrust Setpoints
        """Publish the trajectory setpoint."""
        msg = VehicleRatesSetpoint()
        msg.roll = float(roll)
        msg.pitch = float(pitch)
        msg.yaw = float(yaw)
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = -1* float(thrust)

        # print(f"{msg.thrust_body = }")
        # print(f"{msg.roll = }")
        # print(f"{msg.pitch = }")
        # print(f"{msg.yaw = }")
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.rates_setpoint_publisher.publish(msg)
        
        # print("in publish rates setpoint")
        # self.get_logger().info(f"Publishing rates setpoints [r,p,y]: {[roll, pitch, yaw]}")
        print(f"Publishing rates setpoints [thrust, r,p,y]: {[thrust, roll, pitch, yaw]}")

    def publish_position_setpoint(self, x: float, y: float, z: float): #Publishes Position and Yaw Setpoints
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 0.0  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

# ~~ The following 2 functions are the main functions that run at 10Hz and 100Hz ~~
    def offboard_mode_timer_callback(self) -> None: # ~~Runs at 10Hz and Sets Vehicle to Offboard Mode  ~~
        """Offboard Callback Function for The 10Hz Timer."""
        # print("In offboard timer callback")

        if self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard deciding whether to send heartbeat signal, engage offboard, and arm
            if self.time_from_start <= self.time_before_land:
                self.publish_offboard_control_heartbeat_signal1()
            elif self.time_from_start > self.time_before_land:
                self.publish_offboard_control_heartbeat_signal2()


            if self.offboard_setpoint_counter == 10:
                self.engage_offboard_mode()
                self.arm()
            if self.offboard_setpoint_counter < 11:
                self.offboard_setpoint_counter += 1

        else:
            print(f"Offboard Callback: RC Flight Mode Channel {self.mode_channel} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")
            self.offboard_setpoint_counter = 0

    def newton_raphson_timer_callback(self) -> None: # ~~This is the main function that runs at 100Hz and Administrates Calls to Every Other Function ~~
        """Newton-Raphson Callback Function for The 100Hz Timer."""
        print("NR_Callback")
        print(f"{self.offboard_mode_rc_switch_on = }")
        if self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard deciding whether to send heartbeat signal, engage offboard, and arm
            # self.time_from_start = time.time()-self.T0 #update curent time from start of program for reference trajectories and for switching between NR and landing mode
            
            print(f"--------------------------------------")
            print(f"{self.vehicle_status.nav_state = }")
            print(f"{VehicleStatus.NAVIGATION_STATE_OFFBOARD = }")
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                print("IN OFFBOARD MODE")
                print(f"NR_callback- timefromstart: {self.time_from_start}")

                if self.time_from_start <= self.time_before_land: # wardi controller for first {self.time_before_land} seconds
                    print(f"Entering NR Control Loop for next: {self.time_before_land-self.time_from_start} seconds")
                    self.newton_raphson_control()

                elif self.time_from_start > self.time_before_land: #then land at origin and disarm
                    print("BACK TO SPAWN")
                    self.publish_position_setpoint(0.0, 0.0, -0.3)
                    print(f"self.x: {self.x}, self.y: {self.y}, self.z: {self.z}")
                    if abs(self.x) < 0.1 and abs(self.y) < 0.1 and abs(self.z) <= 0.50:
                        print("Switching to Land Mode")
                        self.land()

            if self.time_from_start > self.time_before_land:
                if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                        print("IN LAND MODE")
                        if abs(self.z) <= .24:
                            print("\nDisarming and Exiting Program")
                            self.disarm()
                            print("\nSaving all data!")
                            # if self.pyjoules_on:
                            #     self.csv_handler.save_data()
                            exit(0)
            print(f"--------------------------------------")
            print("\n\n")
        else:
            print(f"NR Callback: RC Flight Mode Channel {self.mode_channel} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~ From here down are the functions that actually calculate the control input ~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    def newton_raphson_control(self): # Runs Newton-Rapshon Control Algorithm Structure
        t0 = time.time()
        """Newton-Raphson Control Algorithm."""
        print(f"NR_State: {self.nr_state}")

        if self.first_iteration:
            print("First Iteration")
            self.T0 = time.time()
            self.first_iteration = False

        self.time_from_start = time.time()-self.T0

        # Change the previous input from throttle to force for NR calculations that require the previous input
        old_throttle = self.u0[0][0]
        old_force = self.get_force_from_throttle_command(old_throttle)
        last_input_using_force = np.vstack([old_force, self.u0[1:]])

#~~~~~~~~~~~~~~~ Calculate reference trajectory ~~~~~~~~~~~~~~~

        if self.time_from_start <= self.cushion_time:
            reffunc = self.hover_ref_func(1)
        elif self.cushion_time < self.time_from_start < self.cushion_time + self.flight_time:
            reffunc = self.main_traj(self.hover_type) if self.main_traj == self.hover_ref_func else self.main_traj()
        elif self.cushion_time + self.flight_time <= self.time_from_start <= self.time_before_land:
            reffunc = self.hover_ref_func(1)
        else:
            reffunc = self.hover_ref_func(1)
        print(f"reffunc: {reffunc}")

        # Calculate the Newton-Rapshon control input and transform the force into a throttle command for publishing to the vehicle
        new_u = self.get_new_NR_input(last_input_using_force, reffunc)
        new_force = new_u[0][0]        
        new_throttle = float(self.get_throttle_command_from_force(new_force))
        new_roll_rate = float(new_u[1][0])  # Convert jax.numpy array to float
        new_pitch_rate = float(new_u[2][0])  # Convert jax.numpy array to float
        new_yaw_rate = float(new_u[3][0])    # Convert jax.numpy array to float

        # Build the final input vector to save as self.u0 and publish to the vehicle via publish_rates_setpoint:
        final = [new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate] # final input vector
        current_input_save = np.array(final).reshape(-1, 1) # reshaped to column vector for saving as class variable
        self.u0 = current_input_save # saved as class variable for next iteration of NR control

        print(f"{final = }")
        self.publish_rates_setpoint(final[0], final[1], final[2], final[3])
        
        # Log the states, inputs, and reference trajectories for data analysis
        controller_callback_time = time.time() - t0
        state_input_ref_log_info = [float(self.x), float(self.y), float(self.z), float(self.yaw), float(final[0]), float(final[1]), float(final[2]), float(final[3]), float(reffunc[0][0]), float(reffunc[1][0]), float(reffunc[2][0]), float(reffunc[3][0]), self.time_from_start, controller_callback_time, self.vnorm]
        self.update_logged_data(state_input_ref_log_info)


# ~~ The following functions handle the log update and data retrieval for analysis ~~
    def update_logged_data(self, data):
        print("Updating Logged Data")
        self.x_log.append(data[0])
        self.y_log.append(data[1])
        self.z_log.append(data[2])
        self.yaw_log.append(data[3])
        self.throttle_log.append(data[4])
        self.roll_log.append(data[5])
        self.pitch_log.append(data[6])
        self.yaw_rate_log.append(data[7])
        self.ref_x_log.append(data[8])
        self.ref_y_log.append(data[9])
        self.ref_z_log.append(data[10])
        self.ref_yaw_log.append(data[11])
        self.ctrl_loop_time_log.append(data[12])
        self.ctrl_callback_timel_log.append(data[13])
        self.vnorm_log.append(data[14])

    def get_x_log(self): return np.array(self.x_log).reshape(-1, 1)
    def get_y_log(self): return np.array(self.y_log).reshape(-1, 1)
    def get_z_log(self): return np.array(self.z_log).reshape(-1, 1)
    def get_yaw_log(self): return np.array(self.yaw_log).reshape(-1, 1)
    def get_throttle_log(self): return np.array(self.throttle_log).reshape(-1, 1)
    def get_roll_log(self): return np.array(self.roll_log).reshape(-1, 1)
    def get_pitch_log(self): return np.array(self.pitch_log).reshape(-1, 1)
    def get_yaw_rate_log(self): return np.array(self.yaw_rate_log).reshape(-1, 1)
    def get_ref_x_log(self): return np.array(self.ref_x_log).reshape(-1, 1)
    def get_ref_y_log(self): return np.array(self.ref_y_log).reshape(-1, 1)
    def get_ref_z_log(self): return np.array(self.ref_z_log).reshape(-1, 1)
    def get_ref_yaw_log(self): return np.array(self.ref_yaw_log).reshape(-1, 1)
    def get_ctrl_loop_time_log(self): return np.array(self.ctrl_loop_time_log).reshape(-1, 1)
    def get_pred_timel_log(self): return np.array(self.pred_timel_array).reshape(-1, 1)
    def get_nr_timel_log(self): return np.array(self.nr_timel_array).reshape(-1, 1)
    def get_ctrl_callback_timel_log(self): return np.array(self.ctrl_callback_timel_log).reshape(-1, 1)
    def get_vnorm_log(self): return np.array(self.vnorm_log).reshape(-1, 1)
    def get_metadata(self): return self.metadata.reshape(-1, 1)



# ~~ The following functions do the actual calculations for the Newton-Rapshon Control Algorithm ~~
    def get_new_NR_input(self, last_input, reffunc): #Gets Newton-Rapshon Control Input "new_u" with or without pyjoules by calling get_new_NR_input_execution_function
        """ Calls the Newton-Rapshon Control Algorithm Execution Function with or without pyjoules """ 
        print("######################################################################")
        if self.pyjoules_on:
            with EnergyContext(handler=self.csv_handler, domains=[RaplPackageDomain(0), RaplCoreDomain(0)]):
                new_u = self.get_new_NR_input_execution_function(last_input, reffunc)
        else:
            new_u = self.get_new_NR_input_execution_function(last_input, reffunc)
        return new_u

    def quaternion_from_yaw(self, yaw):
        """ Convert yaw angle to a quaternion. """
        half_yaw = yaw / 2.0
        return np.array([np.cos(half_yaw), 0, 0, np.sin(half_yaw)])

    def quaternion_conjugate(self, q):
        """ Return the conjugate of a quaternion. """
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quaternion_multiply(self, q1, q2):
        """ Multiply two quaternions. """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def yaw_error_from_quaternion(self, q):
        """ Extract the yaw error (in radians) from a quaternion. """
        return 2 * np.arctan2(q[3], q[0])
    
    def quaternion_shortest_path(self, q):
        """ Calculate the shortest path quaternion. """
        # if q[0] < 0:
        #     return -q
        return np.sign(q[0]) * q
    
    def quaternion_normalize(self, q):
        """ Normalize a quaternion. """
        return q / np.linalg.norm(q)

    def shortest_path_yaw_quaternion(self, current_yaw, desired_yaw):
        """ Calculate the shortest path between two yaw angles using quaternions. """
        q_current = self.quaternion_normalize(self.quaternion_from_yaw(current_yaw)) #unit quaternion from current yaw angle
        q_desired = self.quaternion_normalize(self.quaternion_from_yaw(desired_yaw)) #unit quaternion from desired yaw angle
        
        q_error = self.quaternion_multiply(q_desired, self.quaternion_conjugate(q_current)) #error quaternion
        q_error_normalized = self.quaternion_normalize(q_error) #normalize error quaternion
        q_error_shortest = self.quaternion_shortest_path(q_error_normalized) #shortest path quaternion
        return self.yaw_error_from_quaternion(q_error_shortest) #return yaw error from shortest path quaternion

    def shortest_path_yaw(self, current_yaw, desired_yaw): #Calculates shortest path between two yaw angles
        """ Calculate the shortest path to the desired yaw angle. """
        current_yaw = self.normalize_angle(current_yaw)
        desired_yaw = self.normalize_angle(desired_yaw)
        
        delta_yaw = self.normalize_angle(desired_yaw - current_yaw)
        
        return delta_yaw
    
    def get_tracking_error(self, reffunc, pred):
        # print(f"reffunc: {reffunc}")
        print(f"pred: {pred}")
        err = reffunc - pred
        # # current_yaw = pred[3][0] # current yaw angle
        # # desired_yaw = reffunc[3][0] # desired yaw angle
        if self.use_quat_yaw:
            # print("Using quaternion for yaw error!!!!!")
            err[3][0] = self.shortest_path_yaw_quaternion(pred[3][0], reffunc[3][0])
        else:
            err[3][0] = self.shortest_path_yaw(pred[3][0], reffunc[3][0])
        return err
    

    # def execute_cbf(self, current, phi, max, min, gamma):
    #     v = 0.0 # influence value initialized to 0 as default for if no CBF is needed        
    #     if current >= 0:
    #         zeta = gamma * (max - current) - phi
    #         if zeta < 0:
    #             v = zeta
    #     if current < 0:
    #         zeta = gamma * (min - current) - phi
    #         if zeta > 0:
    #             v = zeta
    #     return v


    def execute_cbf(self, current, phi, max_value, min_value, gamma):
        zeta_max = gamma * (max_value - current) - phi
        zeta_min = gamma * (min_value - current) - phi

        v = np.where(current >= 0, np.minimum(0, zeta_max), np.maximum(0, zeta_min))
        return v

    
    def integral_cbf(self, last_input, phi):
        print("INTEGRAL CBF IMPLEMENTATION")
        """ CBF IMPLEMENTATION THAT DIRECTS udot TO KEEP INPUT u WITHIN SAFE REGION """
        # Set up CBF parameters:
        # Get current thrust (force) and rates
        curr_thrust = last_input[0][0]
        curr_roll_rate = last_input[1][0]
        curr_pitch_rate = last_input[2][0]
        curr_yaw_rate = last_input[3][0]

        # Get current newton-raphson udot value we just calculated that we want to direct towards safe region(NR = (dg/du)^-1 * (yref - ypred)) (before alpha tuning)
        phi_thrust = phi[0][0]
        phi_roll_rate = phi[1][0]
        phi_pitch_rate = phi[2][0]
        phi_yaw_rate = phi[3][0]

        # CBF FOR THRUST
        thrust_gamma = 1.0 # CBF parameter
        thrust_max = 27 # max thrust (force) value to limit thrust to
        thrust_min = 0.5 # min thrust (force) value to limit thrust to
        v_thrust = self.execute_cbf(curr_thrust, phi_thrust, thrust_max, thrust_min, thrust_gamma)

        # SET UP CBF FOR RATES
        rates_max_abs = 0.8 # max absolute value of roll, pitch, and yaw rates to limit rates to
        rates_max = rates_max_abs 
        rates_min = -rates_max_abs
        gamma_rates = 1.0 # CBF parameter
        v_roll = self.execute_cbf(curr_roll_rate, phi_roll_rate, rates_max, rates_min, gamma_rates) #CBF FOR ROLL
        v_pitch = self.execute_cbf(curr_pitch_rate, phi_pitch_rate, rates_max, rates_min, gamma_rates) #CBF FOR PITCH
        v_yaw = self.execute_cbf(curr_yaw_rate, phi_yaw_rate, rates_max, rates_min, gamma_rates) #CBF FOR YAW

        v = np.array([[v_thrust, v_roll, v_pitch, v_yaw]]).T
        print(f"{v = }")
        return v
    

    def get_new_NR_input_execution_function(self, last_input, ref): # Executes Newton-Rapshon Control Algorithm -- gets called by get_new_NR_input either with or without pyjoules
        """ Calculates the Newton-Raphson Control Input. """
        nrt0 = time.time()
        STATE = np.array([self.state_vector[0][0], self.state_vector[1][0], self.state_vector[2][0], self.state_vector[3][0], self.state_vector[4][0], self.state_vector[5][0], self.state_vector[6][0], self.state_vector[7][0], self.state_vector[8][0]]).reshape(9,1)
        INPUT = np.array([last_input[0][0], last_input[1][0], last_input[2][0], last_input[3][0]]).reshape(4,1)
        print(f"{STATE = }")
        print(f"{INPUT = }")
        if self.ctrl_type == 1 :
            print("Using Original Wardi All Jax") #0-order hold prediction
            u, v = NR_tracker_original(STATE, INPUT, ref, self.T_LOOKAHEAD, self.LOOKAHEAD_STEP, self.INTEGRATION_STEP, self.MASS) #(currstate, currinput, ref, T_lookahead, integration_step, sim_step, mass):
            nr_time_elapsed = time.time() - nrt0
            print(f"NR_time_elapsed: {nr_time_elapsed}, Good For {1/nr_time_elapsed} Hz")
            print(f"v: {v}")
            print(f"u: {u}")

            self.pred_timel_array.append(nr_time_elapsed)
            self.nr_timel_array.append(nr_time_elapsed)
            return u
        elif self.ctrl_type == 2:
            print("Using Optim Wardi All Jax") #0-order hold prediction
            u = NR_tracker_optim(STATE, INPUT, ref, self.T_LOOKAHEAD, self.LOOKAHEAD_STEP, self.INTEGRATION_STEP, self.MASS)
            nr_time_elapsed = time.time() - nrt0
            print(f"NR_time_elapsed: {nr_time_elapsed}, Good For {1/nr_time_elapsed} Hz")
            print(f"u: {u}")

            self.pred_timel_array.append(nr_time_elapsed)
            self.nr_timel_array.append(nr_time_elapsed)
            return u 
        elif self.ctrl_type == 3:
            print("Using Jax for Pred and Jacobian Only for Original Wardi") #0-order hold prediction
            pred = np.array(predict_output(STATE, INPUT, self.T_LOOKAHEAD, self.INTEGRATION_STEP, self.MASS)).reshape(-1,1)
            dgdu = np.array(get_inv_jac_pred_u(STATE, INPUT, self.T_LOOKAHEAD, self.INTEGRATION_STEP, self.MASS))
            self.pred_timel_array.append(time.time() - nrt0)

            error = self.get_tracking_error(ref, pred) # calculates tracking error
            NR = dgdu @ error # calculates newton-raphson control input without speed-up parameter
            v = self.integral_cbf(last_input, NR)
            udot = NR + v # udot = { inv(dg/du) * (yref - ypred) } + v = NR + v = newton-raphson control input + CBF adjustment
            change_u = udot * self.newton_raphson_timer_period #crude integration of udot to get u (maybe just use 0.02 as period)
            alpha = np.array([[20,30,30,30]]).T # Speed-up parameter (maybe play with uniform alpha values rather than ones that change for each input)
            u = last_input + alpha * change_u # u_new = u_old + alpha * change_u
    
            nr_time_elapsed = time.time()-nrt0
            self.nr_timel_array.append(nr_time_elapsed)
            print(f"NR_time_elapsed: {nr_time_elapsed}, Good For {1/nr_time_elapsed} Hz")
            print(f"u: {u}")
            return u
        elif self.ctrl_type == 4:
            print("Using Linearized Predictor for Wardi")
            gravity = np.array([[self.MASS * self.GRAVITY, 0, 0, 0]]).T #gravity vector that counteracts input vector: [-mg, 0, 0, 0]
            # STATE = np.array([self.state_vector[0][0], self.state_vector[1][0], self.state_vector[2][0], self.state_vector[3][0], self.state_vector[4][0], self.state_vector[5][0], self.state_vector[6][0], self.state_vector[7][0], self.state_vector[8][0]]).reshape(9,1)
            # INPUT = np.array([last_input[0][0], last_input[1][0], last_input[2][0], last_input[3][0]]).reshape(4,1)
            pred = np.array(self.C @ (self.eAT@STATE + self.int_eATB @ (INPUT - gravity))) # y(t+T) = C * (eAT * x(t) + int_eATB * (u(t) - gravity))
            self.pred_timel_array.append(time.time() - nrt0)
            print(f"{pred = }")

            error = self.get_tracking_error(ref, pred) # calculates tracking error
            NR = self.jac_inv @ error # calculates newton-raphson control input without speed-up parameter
            v = self.integral_cbf(last_input, NR)
            udot = NR + v # udot = { inv(dg/du) * (yref - ypred) } + v = NR + v = newton-raphson control input + CBF adjustment
            change_u = udot * self.newton_raphson_timer_period #crude integration of udot to get u (maybe just use 0.02 as period)
            alpha = np.array([[20,30,30,30]]).T # Speed-up parameter (maybe play with uniform alpha values rather than ones that change for each input)
            u = last_input + alpha * change_u # u_new = u_old + alpha * change_u
    
            nr_time_elapsed = time.time()-nrt0
            self.nr_timel_array.append(nr_time_elapsed)
            print(f"NR_time_elapsed: {nr_time_elapsed}, Good For {1/nr_time_elapsed} Hz")
            print(f"u: {u}")
            return u
        elif self.ctrl_type == 5:
            print(f"Using Linearized Predictor for Wardi in Jax")
            u, v = NR_tracker_linpred(STATE, INPUT, ref, self.INTEGRATION_STEP, self.MASS, self.eAT, self.int_eATB, self.jac_inv) #(currstate, currinput, ref, T_lookahead, integration_step, sim_step, mass):
            nr_time_elapsed = time.time() - nrt0
            print(f"NR_time_elapsed: {nr_time_elapsed}, Good For {1/nr_time_elapsed} Hz")
            print(f"v: {v}")
            print(f"u: {u}")

            self.pred_timel_array.append(nr_time_elapsed)
            self.nr_timel_array.append(nr_time_elapsed)
            return u
        
        elif self.ctrl_type == 6:
            print(f"Using Differentially Flat System for Wardi")
            u, self.x_df, self.vnorm = self.get_NR_flat_synthesis(STATE, INPUT, self.x_df, ref, self.T_LOOKAHEAD, self.INTEGRATION_STEP, self.MASS)
            nr_time_elapsed = time.time() - nrt0
            print(f"NR_time_elapsed: {nr_time_elapsed}, Good For {1/nr_time_elapsed} Hz")
            print(f"u: {u}")

            self.pred_timel_array.append(nr_time_elapsed)
            self.nr_timel_array.append(nr_time_elapsed)
            return u
        elif self.ctrl_type == 7:
            print(f"Using Differentially Flat System for Wardi in Jax")
            # curr_yaw = STATE[-1][0]
            curr_yaw_ref = ref[3][0].item()
            yaw_dot = (curr_yaw_ref - self.last_yaw_ref) / self.newton_raphson_timer_period
            self.last_yaw_ref = curr_yaw_ref
            u, self.x_df = NR_tracker_flat(STATE, INPUT, self.x_df, ref, self.T_LOOKAHEAD, self.INTEGRATION_STEP, self.MASS, self.ROT, yaw_dot)
            # clipped_thrust, p, q, r, x_df = NR_tracker_flat(STATE, INPUT, self.x_df, ref, self.T_LOOKAHEAD, self.INTEGRATION_STEP, self.MASS)
            # print(f"{clipped_thrust = }")
            # print(f"{p = }")
            # print(f"{q = }")
            # print(f"{r = }")
            # print(f"{x_df = }")
            self.vnorm = 0
            nr_time_elapsed = time.time() - nrt0
            print(f"NR_time_elapsed: {nr_time_elapsed}, Good For {1/nr_time_elapsed} Hz")
            print(f"u: {u}")
            # exit(0)
            self.pred_timel_array.append(nr_time_elapsed)
            self.nr_timel_array.append(nr_time_elapsed)
            return u
        
    def get_NR_flat_synthesis(self, STATE, INPUT, x_df, ref, T_lookahead, step, MASS):

        curr_x, curr_y, curr_z, curr_yaw = STATE[0][0], STATE[1][0], STATE[2][0], STATE[-1][0]
        curr_vx, curr_vy, curr_vz = STATE[3][0], STATE[4][0], STATE[5][0]
        curr_yawdot = INPUT[3][0]
        curr_thrust, curr_roll_rate, curr_pitch_rate = INPUT[0][0], INPUT[1][0], INPUT[2][0]

        T_lookahead = self.T_LOOKAHEAD
        step = self.newton_raphson_timer_period

        flat_normal = self.flat_normal




        gamma = 10
        tau = curr_thrust

        # h = -(10/13.5**2) * (tau-13.5)**2 + 10
        # dhdtau = -(20/13.5**2) * (tau-13.5)

        h = -(5/72)*(tau-3)*(tau-27)
        dhdtau = -(5/72)*(2*tau-30)

        # h = -(10/121) * (tau**2 - 32*tau + 135)
        # dhdtau = -(20/121) * (tau - 16)

        # h = -(40/289) * (tau-10)*(tau-27)
        # dhdtau = -(40/289) * (2*tau - 37)

        # h = -(10/81)*tau**2 + (180/81)*tau
        # dhdtau = -(20/81)*tau + (180/81)
        # h = -(8/5)*(tau-10)*(tau-18)
        # dhdtau = -(8/5)*(2*tau-25)

        # h = -(tau-10)*(tau-16)
        # dhdtau = -(2*tau-28)


        a3 = np.array([[0,0,-1]]).T
        grav_vector = self.GRAVITY * a3
        accels = x_df[8:11] + grav_vector
        print(f"{x_df[8:11] = }")
        print(f"{grav_vector = }")
        x = accels
        xT_over_normx = x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else np.zeros((1,3))
        print(f"{x = }")
        print(f"{xT_over_normx = }")
        taudot_coeff = MASS * xT_over_normx
        phi = dhdtau * taudot_coeff

        print(f"{tau = }")
        print(f"{h = }")

        print(f"{dhdtau = }")
        print(f"{taudot_coeff = }")
        print(f"{phi = }")
        print(f"{phi.shape = }")
        # exit(0)


        if flat_normal:
            print(f"HIII")
            z1 = np.array([[curr_x, curr_y, curr_z, curr_yaw]]).T # position 
            z2 = np.array([[curr_vx, curr_vy, curr_vz, curr_yawdot]]).T # velocity
            z3 = x_df[8:12] #acceleration
            # print(f"{z1 = }")
            # print(f"{z2 = }")
            # print(f"{z3 = }")

            dgdu_inv = (2/T_lookahead**2) * np.eye(4)
            alpha = 1*np.array([[20,30,30,30]]).T
            pred = z1 + z2*T_lookahead + (1/2)*z3*T_lookahead**2
            print(f"{pred = }")
            print(f"{ref = }")
            error = self.get_tracking_error(ref, pred)
            print(f"{error = }")
            NR = dgdu_inv @ error
            u_df = alpha * NR
            self.u_df = u_df
            print(f"{u_df = }")


            d = -phi.T @ u_df[0:3] - gamma * h
            print(f"{d = }")

            if d <= 0:
                v = np.zeros((4,1))
            else:
                v = (d / (np.linalg.norm(phi))**2 ) * phi
                v = np.vstack([v, np.zeros((1,1))])

            print(f"{v.shape = }")
            print(f"{v = }")


            A_df = np.block([
                        [np.zeros((4,4)), np.eye(4), np.zeros((4,4))],
                        [np.zeros((4,4)), np.zeros((4,4)), np.eye(4)],
                        [np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4))]
                        ])
            
            B_df = np.block([
                        [np.zeros((4,4))],
                        [np.zeros((4,4))],
                        [np.eye(4)]
                        ])
            x_dot_df = A_df @ x_df + B_df @ (u_df + v)
            x_df = x_df + x_dot_df * step
            # print(f"{self.x_df[4:] = }")
            sigma = x_df[0:4]
            sigmad1 = x_df[4:8]
            sigmad2 = x_df[8:12]
            sigmad3 = u_df

            x_df = np.vstack([sigma, sigmad1, sigmad2])


            print(f"{sigma = }")
            print(f"{sigmad1 = }")
            print(f"{sigmad2 = }")
            print(f"{sigmad3 = }")




        else:
            z1 = np.array([[curr_x, curr_y, curr_z, curr_yaw]]).T # position 
            z2 = np.array([[curr_vx, curr_vy, curr_vz, curr_yawdot]]).T # velocity
            z3 = x_df[8:12] #acceleration
            z4 = x_df[12:] #jerk
            print(f"{z1 = }")
            print(f"{z2 = }")
            print(f"{z3 = }")
            print(f"{z4 = }")

            dgdu_inv = (6/T_lookahead**3) * np.eye(4)
            alpha = np.array([[20,30,30,30]]).T
            pred = z1 + z2*T_lookahead + (1/2)*z3*T_lookahead**2 + (1/6)*z4*T_lookahead**3
            # print(f"{pred = }")
            # print(f"{ref = }")
            error = self.get_tracking_error(ref, pred)
            # print(f"{error = }")
            NR = dgdu_inv @ error
            u_df = alpha * NR
            self.u_df = u_df
            # print(f"{u_df = }")

            A_df = np.block([
                        [np.zeros((4,4)), np.eye(4), np.zeros((4,4)), np.zeros((4,4))],
                        [np.zeros((4,4)), np.zeros((4,4)), np.eye(4), np.zeros((4,4))],
                        [np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.eye(4)],
                        [np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4))]
                        ])

            B_df = np.block([
                        [np.zeros((4,4))],
                        [np.zeros((4,4))],
                        [np.zeros((4,4))],
                        [np.eye(4)]
                        ])  
            
            sd3 = z4[0:3]
            p = -2 * sd3
            hj = -np.linalg.norm(sd3) + 40**2
            gamma2 = 10
            d = 2 * (sd3.T @ u_df[0:3]) - gamma2*hj
            print(f"{sd3 = }")
            print(f"{p = }")
            print(f"{hj = }")
            print(f"{d = }")
            if d <= 0:
                v = np.zeros((4,1))
                print(f"{v = }")

            else:
                v = (d / (np.linalg.norm(p))**2 ) * p
                print(f"{v = }")
                v = np.vstack([v, np.zeros((1,1))])
                print(f"{v = }")
                # exit(0)

            


            x_dot_df = A_df @ x_df + B_df @ (u_df + v)
            x_df = x_df + x_dot_df * step
            # print(f"{self.x_df[4:] = }")
            sigma = x_df[0:4]
            sigmad1 = x_df[4:8]
            sigmad2 = x_df[8:12]
            sigmad3 = x_df[12:]
            sigmad4 = u_df


            # dgdu_inv = (2/T_lookahead**2) * np.eye(4)
            # alpha = 1*np.array([[20,30,30,30]]).T
            # pred = z1 + z2*T_lookahead + (1/2)*z3*T_lookahead**2
            # print(f"{pred = }")
            # print(f"{ref = }")
            # error = self.get_tracking_error(ref, pred)
            # print(f"{error = }")
            # NR = dgdu_inv @ error
            # u_df = alpha * NR
            # self.u_df = u_df
            # print(f"{u_df = }")


            # d = -phi.T @ u_df[0:3] - gamma * h
            # print(f"{d = }")

            # if d <= 0:
            #     v = np.zeros((4,1))
            # else:
            #     v = (d / (np.linalg.norm(phi))**2 ) * phi
            #     v = np.vstack([v, np.zeros((1,1))])

            # print(f"{v.shape = }")
            # print(f"{v = }")


            # A_df = np.block([
            #             [np.zeros((4,4)), np.eye(4), np.zeros((4,4))],
            #             [np.zeros((4,4)), np.zeros((4,4)), np.eye(4)],
            #             [np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4))]
            #             ])
            
            # B_df = np.block([
            #             [np.zeros((4,4))],
            #             [np.zeros((4,4))],
            #             [np.eye(4)]
            #             ])
            # x_dot_df = A_df @ x_df[0:12] + B_df @ (u_df + v)
            # x_df2 = x_df[0:12] + x_dot_df * step
            # # print(f"{self.x_df[4:] = }")
            # # sigma = x_df[0:4]
            # # sigmad1 = x_df[4:8]
            # # sigmad2 = x_df[8:12]
            # # sigmad3 = u_df

            # sigma = x_df2[0:4]
            # sigmad1 = x_df2[4:8]
            # sigmad2 = x_df2[8:12]
            # sigmad3 = x_df1[12:16]
            # # sigmad4 = u_df

            # x_df = np.vstack([sigma, sigmad1, sigmad2, sigmad3])



        




        a1 = np.array([[1,0,0]]).T
        a2 = np.array([[0,1,0]]).T
        a3 = np.array([[0,0,-1]]).T

        # print(f"{sigmad2[0:3] = }")
        grav_vector = self.GRAVITY * a3
        # print(f"{grav_vector = }")
        accels = sigmad2[0:3] + self.GRAVITY*a3
        print(f"{accels = }")

        thrust = MASS * np.linalg.norm(accels)
        # print(f"{thrust = }")

        # thrust_max = 27.0
        # thrust_min = 0.5
        # clipped_thrust = np.clip(thrust, thrust_min, thrust_max)
        clipped_thrust = thrust
        print(f"{clipped_thrust = }")

        b3 = accels / np.linalg.norm(accels)
        print(f"{b3 = }")
        # e1 = np.array([[1., 0., 0.]]).T
        e1 = np.cos(curr_yaw) * a1 + np.sin(curr_yaw) * a2
        print(f"{e1 = }")
        # print(f"{e1n = }")
        # exit(0)
        val2 = np.cross(b3, e1, axis=0)
        b2 = val2 / np.linalg.norm(val2)
        print(f"{b2 = }")


        b1 = np.cross(b2, b3, axis=0)
        print(f"{b1 = }")

        j = sigmad3[0:3]
        print(f"{j = }")

        print(f"{j.T @ b3 = }")
        print(f"{(j.T @ b3) * b3 = }")
        val3 = (j - (j.T @ b3) * b3)
        print(f"{val3 = }")    
        # val3 = j
        # print(f"{MASS / clipped_thrust = }")
        # print(f"{b2.T @ val3 = }")
        # print(f"{b2.T @ ((MASS / thrust) * val3) = }")
        p = (b2.T @ ((self.MASS/-clipped_thrust) * val3)).item()
        q = (b1.T @ ((self.MASS/-clipped_thrust) * val3)).item()
        print(f"{p = }")
        print(f"{q = }")
        # yaw_err = self.shortest_path_yaw_quaternion(curr_yaw, ref[3][0].item())
        # print(f"{yaw_err = }")
        # r = curr_yawdot + (yaw_err * step)

        # print(f"{ref[3][0].item() = }")
        # print(f"{curr_yaw = }")
        # print(f"{yaw_err = }")
        # print(f"{r = }")
        # exit(0)
        # Define matrices A and B
        A = np.hstack((e1, b2, a3))
        B = self.ROT

        # Define known values in x and y

        curr_yaw_ref = ref[3][0].item()
        # yaw_dot = (curr_yaw_ref - curr_yaw)
        yaw_dot = (curr_yaw_ref - self.last_yaw_ref) / step
        x_known = np.array([yaw_dot])
        self.last_yaw_ref = curr_yaw_ref

        y_known = np.array([p, q])  # y1 = 4, y2 = 2

        print(f"{x_known = }")
        print(f"{y_known = }")
        print(f"{x_known.shape = }")
        print(f"{y_known.shape = }")
        # exit(0)

        # Split A into parts affecting unknowns (A1) and knowns (A2)
        A1 = A[:, :2]  # First two columns (unknown x1, x2)
        A2 = A[:, 2:]  # Last column (known x3)

        # Split B into parts affecting unknowns (B1) and knowns (B2)
        B1 = B[:, 2:]  # Last column (unknown y3)
        B2 = B[:, :2]  # First two columns (known y1, y2)

        # Compute right-hand side
        rhs = (B2 @ y_known) - (A2 @ x_known)  # Move known values to RHS

        # Solve system for [x1, x2, y3]
        M = np.hstack((A1, -B1))  # Construct coefficient matrix
        unknowns = np.linalg.solve(M, rhs)  # Solve for unknowns

        # Extract solutions
        x1, x2, r = unknowns
        r = -r
        print(f"{r = }")

        max_rate = 0.8
        r = np.clip(r, -max_rate, max_rate)
        p = np.clip(p, -max_rate, max_rate)
        q = np.clip(q, -max_rate, max_rate)
        # print(f"{p = }, {q = }, {r = }")
        u = np.array([[clipped_thrust, p, q, r]]).T
        # print(f"{u = }")
        print(f"{e1 = }")
        print(f"{b2 = }")
        print(f"{a3 = }")

        return u, x_df, np.linalg.norm(v)

    def observer_matrix(self): #Calculates Observer Matrix for Prediction of desired outputs from all 9 states
        """ Calculates the observer matrix for prediction of desired outputs from all 9 states. """
        C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return C

    def linearized_model(self): #Calculates Linearized Model Matrices for Prediction (eAT, int_eATB, int_eAT, C)
        """ Calculates the linearized model matrices for prediction. """
        # x(t) = eAT*x(t0) + int_eATB*u(t) with u(t) = u over T = T_lookahead seconds
        # simplifies to x(t) = eAT*x(t) + int_eATB*(u - gravity) in our implementation as seen in getyorai_g_linear_predict
        # y(t) = C*x(t)
        GRAVITY = self.GRAVITY
        MASS = self.MASS
        T_LOOKAHEAD = self.T_LOOKAHEAD
        A = smp.Matrix(
            [
                [0, 0, 0,    1, 0, 0,     0,   0, 0],
                [0, 0, 0,    0, 1, 0,     0,   0, 0],
                [0, 0, 0,    0, 0, 1,     0,   0, 0],

                [0, 0, 0,    0, 0, 0,     0,-1*GRAVITY, 0],
                [0, 0, 0,    0, 0, 0,     GRAVITY,   0, 0],
                [0, 0, 0,    0, 0, 0,     0,   0, 0],

                [0, 0, 0,    0, 0, 0,     0,   0, 0],
                [0, 0, 0,    0, 0, 0,     0,   0, 0],
                [0, 0, 0,    0, 0, 0,     0,   0, 0],

            ]
            )
        eAT = smp.exp(A*T_LOOKAHEAD)
        B = smp.Matrix(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],

                [-1/MASS, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
            )
        A = np.array(A).astype(np.float64)
        int_eAT = np.zeros_like(A)
        rowcol = int_eAT.shape[0]
        for row in range(rowcol):
            for column in range(rowcol):
                f = lambda x: sp_linalg.expm(A*(T_LOOKAHEAD-x))[row,column]
                int_eAT[row,column] = sp_int.quad(f, 0, T_LOOKAHEAD)[0]

        int_eATB = int_eAT @ B
        eAT = np.array(eAT).astype(np.float64)
        int_eATB = np.array(int_eATB).astype(np.float64) 
        jac_inv = np.linalg.inv(self.C @ int_eATB)#Calculate Inverse Jacobian of Linearized Model Matrices
        return eAT, int_eATB, jac_inv




# ~~ The following functions are reference trajectories for tracking ~~
    def hover_ref_func(self, num): #Returns Constant Hover Reference Trajectories At A Few Different Positions ([x,y,z,yaw])
        """ Returns constant hover reference trajectories at a few different positions. """
        hover_dict = {
            1: np.array([[0.0, 0.0, -0.6, 0.]]).T,
            2: np.array([[0.0, 0.8, -0.8, 0.]]).T,
            3: np.array([[0.8, 0.0, -0.8, 0.]]).T,
            4: np.array([[0.8, 0.8, -0.8, 0.]]).T,
            5: np.array([[0.0, 0.0, -10.0, 0.0]]).T,
            6: np.array([[1.0, 1.0, -4.0, 0.0]]).T,
            7: np.array([[3.0, 4.0, -5.0, 0.0]]).T,
            8: np.array([[1.0, 1.0, -3.0, 0.0]]).T,
            9: np.array([[0.,0.,-(0.8*np.sin((2*np.pi/5)*(time.time()))+1.0),0.]]).T,
        }
        if num > len(hover_dict) or num < 1:
            print(f"hover_dict #{num} not found")
            exit(0)

        if not self.sim:
            if num > 4:
                print("hover modes 5+ not available for hardware")
                exit(1)
            
        print(f"hover_dict #{num}")
        return hover_dict.get(num)
   
    def circle_horz_ref_func(self): #Returns Circle Reference Trajectory in Horizontal Plane ([x,y,z,yaw])
        """ Returns circle reference trajectory in horizontal plane. """
        print("circle_horz_ref_func")

        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD
        
        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637

        if self.double_speed:
            PERIOD /= 2.0

        w = 2*m.pi / PERIOD
        r = np.array([[.6*m.cos(w*t), .6*m.sin(w*t), -0.7, 0.0]]).T

        return r
    
    def circle_horz_spin_ref_func(self): #Returns Circle Reference Trajectory in Horizontal Plane while Yawing([x,y,z,yaw])
        """ Returns circle reference trajectory in horizontal plane while yawing. """
        print("circle_horz_spin_ref_func")

        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD

        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637

        if self.double_speed:
            PERIOD /= 2.0

        w = 2*m.pi / PERIOD

        SPIN_PERIOD = 20
        yaw_ref = (t) / (SPIN_PERIOD / (2*m.pi))

        r = np.array([[.6*m.cos(w*t), .6*m.sin(w*t), -0.7, yaw_ref ]]).T

        return r
    
    def circle_vert_ref_func(self): #Returns Circle Reference Trajectory in Vertical Plane ([x,y,z,yaw])
        """ Returns circle reference trajectory in vertical plane. """
        print("circle_vert_ref_func")

        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD
        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637

        if self.double_speed:
            PERIOD /= 2.0

        w = 2*m.pi / PERIOD
        r = np.array([[.35*m.cos(w*t), 0.0, -1*(.35*m.sin(w*t) + .75), 0.0]]).T

        return r
    
    def fig8_horz_ref_func(self): #Returns Figure 8 Reference Trajectory in Horizontal Plane ([x,y,z,yaw])
        """ Returns figure 8 reference trajectory in horizontal plane. """
        print("fig8_horz_ref_func")

        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD

        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637

        if self.double_speed:
            PERIOD /= 2.0

        w = 2*m.pi / PERIOD
        r = np.array([[.35*m.sin(2*w*t), .35*m.sin(w*t), -0.8, 0.0]]).T

        return r
    
    def fig8_vert_ref_func_short(self): #Returns A Short Figure 8 Reference Trajectory in Vertical Plane ([x,y,z,yaw])
        """ Returns a short figure 8 reference trajectory in vertical plane. """
        print(f"fig8_vert_ref_func_short")

        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD

        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637

        if self.double_speed:
            PERIOD /= 2.0

        w = 2*m.pi / PERIOD
        r = np.array([[0.0, .35*m.sin(w*t), -1*(.35*m.sin(2*w*t) + 0.8), 0.0]]).T

        return r
    
    def fig8_vert_ref_func_tall(self): #Returns A Tall Figure 8 Reference Trajectory in Vertical Plane ([x,y,z,yaw])
        """ Returns a tall figure 8 reference trajectory in vertical plane. """
        print(f"fig8_vert_ref_func_tall")

        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD

        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637
        if self.double_speed:
            PERIOD /= 2.0

        w = 2*m.pi / PERIOD
        r = np.array([[0.0, .35*m.sin(2*w*t), -1*(.35*m.sin(w*t)+0.8), 0.0]]).T

        return r
    
    def get_helix_position(self, t, z0=0.8, z_max=2.3, radius=0.6, num_turns=3, cycle_time=30.0):
        """
        Returns (x, y, z) at time t for a helix with its own up/down cycle time.
        The helix:
        - spirals up from z0 to z_max over cycle_time/2
        - spirals down from z_max to z0 over cycle_time/2
        This repeats independently of any total time duration.
        """
        t_cycle = t % cycle_time
        T_half = cycle_time / 2.0

        if t_cycle <= T_half:
            # Going up
            z = z0 + (z_max - z0) * (t_cycle / T_half)
            progress = (z - z0) / (z_max - z0)  # 0 to 1
        else:
            # Going down
            t_down = t_cycle - T_half
            z = z_max - (z_max - z0) * (t_down / T_half)
            progress = (z - z0) / (z_max - z0)  # 1 to 0

        # Angle is based on how far along this cycle you are
        theta = 2 * np.pi * num_turns * progress
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        return x, y, -z


    def helix(self): #Returns Helix Reference Trajectory ([x,y,z,yaw])
        """ Returns helix reference trajectory. """
        print(f"helix")
        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD

        x,y,z = self.get_helix_position(t)
        r = np.array([[x, y, z, 0.0]]).T

        # PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637
        # PERIOD_Z = 13

        # if self.double_speed:
        #     PERIOD /= 2.0
        #     PERIOD_Z /= 2.0
        # w = 2*m.pi / PERIOD
        # w_z = 2*m.pi / PERIOD_Z

        # z0 = 0.8
        # height_variance = 0.3
        # r = np.array([[.6*m.cos(w*t), .6*m.sin(w*t), -1*(z0 + height_variance * m.sin(w_z * t)), 0.0]]).T
        return r

    def helix_spin(self): #Returns Spiral Staircase Reference Trajectories while Spinning ([x,y,z,yaw])
        """ Returns helix reference trajectory while yawing. """
        print(f"helix_spin")
        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD

        # PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637

        # if self.double_speed:
        #     PERIOD /= 2.0

        # w = 2*m.pi / PERIOD

        # PERIOD_Z = 13
        # w_z = 2*m.pi / PERIOD_Z
        # z0 = 0.8
        # height_variance = 0.3

        SPIN_PERIOD = 20
        yaw_ref = (t) / (SPIN_PERIOD / (2*m.pi))

        x,y,z = self.get_helix_position(t)
        r = np.array([[x, y, z, yaw_ref]]).T
        # r = np.array([[.6*m.cos(w*t), .6*m.sin(w*t), -1*(z0 + height_variance * m.sin(w_z * t)), yaw_ref]]).T
        return r
    
    def yawing_only(self): #Returns Yawing Reference Trajectory ([x,y,z,yaw])
        """ Returns yawing reference trajectory. """
        print(f"yawing_only")

        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD

        SPIN_PERIOD = 20
        
        yaw_ref = (t) / (SPIN_PERIOD / (2*m.pi))
        r = np.array([[0., 0., -0.5, yaw_ref]]).T
        return r
        
    def interpolate_sawtooth(self, t, num_repeats):
        # Define the points for the modified sawtooth trajectory
        points = [
            (0, 0), (0, 0.4), (0.4, -0.4), (0.4, 0.4), (0.4, -0.4),
            (0, 0.4), (0, -0.4), (-0.4, 0.4), (-0.4, -0.4), 
            (-0.4, 0.4), (0, -0.4), (0, 0)
        ]

        traj_time = self.flight_time  # Total time for the trajectory
        N = num_repeats  # Number of repetitions
        traj_time /= N  # Adjust the total time based on the number of repetitions

        # Define the segment time
        T_seg = traj_time / (len(points) - 1)  # Adjust segment time based on the number of points
        
        # Calculate the time within the current cycle
        cycle_time = t % ((len(points) - 1) * T_seg)
        
        # Determine which segment we're in
        segment = int(cycle_time // T_seg)
        
        # Time within the current segment
        local_time = cycle_time % T_seg
        
        # Select the start and end points of the current segment
        start_point = points[segment]
        end_point = points[(segment + 1) % len(points)]
        
        # Linear interpolation for the current segment
        x = start_point[0] + (end_point[0] - start_point[0]) * (local_time / T_seg)
        y = start_point[1] + (end_point[1] - start_point[1]) * (local_time / T_seg)
        
        return x, y

    def sawtooth(self, num_repeats=1):
        num_repeats = 2 if self.double_speed else 1
        """ Returns a /|/ sawtooth reference trajectory that repeats num_repeats times within self.flight_time. """
        print(f"sawtooth_pattern with {num_repeats} repeats")
        z_ref = -0.7  # Constant altitude
        yaw_ref = 0.0  # Optional yaw control

        # Calculate the x and y positions based on the current time
        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD

        x_ref, y_ref = self.interpolate_sawtooth(t, num_repeats)

        r = np.array([[x_ref, y_ref, z_ref, yaw_ref]]).T
        return r


    def interpolate_triangle(self, t, num_repeats):
        # Define the triangle points
        side_length = 0.6  # replace with your desired side length
        h = np.sqrt(side_length**2 - (side_length/2)**2)
        points = [(0, h/2), (side_length/2, -h / 2), (-side_length/2, -h / 2)]

        traj_time = self.flight_time  # Total time for the trajectory
        N = num_repeats  # Number of repetitions

        # Calculate the segment time
        T_seg = traj_time / (3 * N)

        # Calculate the time within the current cycle
        cycle_time = t % (3 * T_seg)
        
        # Determine which segment we're in
        segment = int(cycle_time // T_seg)
        
        # Time within the current segment
        local_time = cycle_time % T_seg
        
        # Select the start and end points of the current segment
        start_point = points[segment]
        end_point = points[(segment + 1) % 3]
        
        # Linear interpolation for the current segment
        x = start_point[0] + (end_point[0] - start_point[0]) * (local_time / T_seg)
        y = start_point[1] + (end_point[1] - start_point[1]) * (local_time / T_seg)

        return x, y

    def triangle(self, num_repeats = 1):
        num_repeats = 2 if self.double_speed else 1
        """ Returns interpolated triangular reference trajectory ([x, y, z, yaw]) """
        print(f"triangular_trajectory with {num_repeats} repeats")
        z_ref = -0.7  # Constant altitude
        yaw_ref = 0.0 # Constant yaw

        # Define the first point
        side_length = 0.6
        h = np.sqrt(side_length**2 - (side_length / 2)**2)
        first_point = (0, h / 2)

        # Wait until within 0.1 units of the first point
        if self.made_it == 0:
            if np.sqrt((self.x - first_point[0])**2 + (self.y - first_point[1])**2) > 0.1:
                return np.array([[first_point[0], first_point[1], z_ref, yaw_ref]]).T
            else:
                self.made_it = 1


        # Calculate the x and y positions based on the current time
        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD
        x_ref, y_ref = self.interpolate_triangle(t, num_repeats)

        r = np.array([[x_ref, y_ref, z_ref, yaw_ref]]).T
        return r




# ~~ Entry point of the code -> Initializes the node and spins it. Also handles exceptions and logging ~~
def main(args=None):
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    logger = None

    def shutdown_logging(*args):
        print("\nInterrupt/Error/Termination Detected, Triggering Logging Process and Shutting Down Node...")
        if logger:
            logger.log(offboard_control)
        offboard_control.destroy_node()
        rclpy.shutdown()
    # Register the signal handler for Ctrl+C (SIGINT)
    # signal.signal(signal.SIGINT, shutdown_logging)

    try:
        print(f"\nInitializing ROS 2 node: '{__name__}' for offboard control")
        logger = Logger([sys.argv[1]])  # Create logger with passed filename
        rclpy.spin(offboard_control)    # Spin the ROS 2 node
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected (Ctrl+C), exiting...")
    except Exception as e:
        print(f"\nError in main: {e}")
        traceback.print_exc()
    finally:
        shutdown_logging()
        if offboard_control.pyjoules_on:
            offboard_control.csv_handler.save_data()
        print("\nNode has shut down.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError in __main__: {e}")
        traceback.print_exc()