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
from .nr_quad_JAX_utilities import predict_outputs, predict_states, compute_jacobian, compute_adjusted_invjac
from .nr_quad_JAX_utilities import predict_outputs_1order, predict_states_1order, compute_jacobian_1order, compute_adjusted_invjac_1order
from .nr_quad_JAX_NN_utilities import *

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
###############################################################################################################################################

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

        self.mode_channel = 5
        self.pyjoules_on = int(input("Use PyJoules? 1 for Yes 0 for No: ")) #False
        if self.pyjoules_on:
            self.csv_handler = CSVHandler('nr_aggressive.log')
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
        self.cushion_time = 8.0
        self.flight_time = 20.0
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
        self.INTEGRATION_STEP = 0.1
        # Initialize first input for hover at origin
        self.u0 = np.array([[self.get_throttle_command_from_force(-1*self.MASS * self.GRAVITY), 0, 0, 0]]).T
        print(f"u0: {self.u0}")
        # exit(0)

###############################################################################################################################################
        # self.use_quat_yaw: bool = bool(int(input("Use quaternion for yaw error? Press 1 for Yes and 0 for No: ")))
        self.use_quat_yaw = True
        # print(f"{'Using quaternion for yaw error' if self.use_quat_yaw else 'Using angle for yaw error'}")

        self.pred_type = int(input("JaxNonlin, Neural Network, or C-CompiledNonlin -based Predictor? Write 3 for Jax, 2 for NN, and 0 for Nonlinear: "))
        print(f"Predictor #{self.pred_type}: Using {'Jax' if self.pred_type == 3 else 'NN' if self.pred_type == 2 else 'Linear' if self.pred_type == 1 else 'Nonlinear'} Predictor")

        self.C = self.observer_matrix() #Calculate Observer Matrix Needed After Predictions of all States to Get Only the States We Need in Output

        # self.nonlin0 = False
        if self.pred_type == 0 or self.pred_type == 3: #For Nonlinear & Jax Predictors
            self.nonlin0 = True # use 0-order hold for now
            # self.nonlin0 = not bool(int(input("Press 0 for 0-Order Hold and 1 for 1st-Order Hold: ")))

        if self.pred_type == 0: #Nonlinear Predictor
            # print("Using Nonlinear Predictor")
            class Vector9x1(ctypes.Structure):
                _fields_ = [
                    ('x', ctypes.c_double),
                    ('y', ctypes.c_double),
                    ('z', ctypes.c_double),
                    ('vx', ctypes.c_double),
                    ('vy', ctypes.c_double),
                    ('vz', ctypes.c_double),
                    ('roll', ctypes.c_double),
                    ('pitch', ctypes.c_double),
                    ('yaw', ctypes.c_double),
                ]

            if self.nonlin0:
                print("Using CPP 0-Order Hold Predictor")
                # Load the C shared library
                base_path = os.path.dirname(os.path.abspath(__file__))        # Get the directory where the script is located
                nonlin_path = os.path.join(base_path, 'nonlin_0order.so')
                self.my_library = ctypes.CDLL(nonlin_path)  # Update the library filename
                # Set argument and return types for the function
                self.my_library.performCalculations.argtypes = [
                    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int
                    ] 
                print(f"NONLIN LIB PATH: {nonlin_path}")
                # exit(0)
            else:
                print("Using CPP 1st-Order Hold Predictor")
                self.udot = np.array([[0, 0, 0, 0]], dtype=np.float64).T
                print(f"udot: {self.udot}")
                print(f"udot shape: {self.udot.shape}")
                # Load the C shared library
                base_path = os.path.dirname(os.path.abspath(__file__))        # Get the directory where the script is located
                nonlin_path = os.path.join(base_path, 'nonlin_1storder.so')
                self.my_library = ctypes.CDLL(nonlin_path)  # Update the library filename
                # Set argument and return types for the function
                self.my_library.performCalculations.argtypes = [
                    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
                ]
                print(f"NONLIN LIB PATH: {nonlin_path}")
                # exit(0)
            self.my_library.performCalculations.restype = ctypes.POINTER(Vector9x1)

        if self.pred_type == 1: #Linear Predictor
            # print("Using Linear Predictor")
            self.linearized_model() #Calculate Linearized Model Matrices      
            # print(f"{self.jac_inv=}")   
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
            # exit(0)   

        if self.pred_type == 2: #Jax Neural Network Predictor
            print("Using Jax Feedforward NN")
            self.nonlin0 = False
            x = np.random.randn(9)
            u = np.random.randn(4)
            self.apply_model = apply_model
            self.compute_inv_jac = compute_inv_jac
            self.apply_model(x,u)
            self.compute_inv_jac(x,u)
            # compute_jacobian(x,u)
            print("succeeded")
            # exit(0)

        if self.pred_type == 5: #Neural Network Predictor
            print("Using Feedforward NN")
            # Load NN predictor:
            class FeedForward(nn.Module):
                def __init__(self, input_size = 13, output_size = 4):
                    super().__init__()
                    self.feedfwd_stack = nn.Sequential(
                        nn.Linear(input_size, 64),
                        nn.Linear(64, output_size)
                    )
                def forward(self, x):
                    logits = self.feedfwd_stack(x)
                    return logits
                
            self.NN = FeedForward()
            base_path = os.path.dirname(os.path.abspath(__file__))        # Get the directory where the script is located
            ff_path_hardware = os.path.join(base_path, 'holybro_ff.pt')
            ff_path_sim = os.path.join(base_path, 'sim_ff.pt')
            if self.sim:
                self.NN.load_state_dict(torch.load(ff_path_sim))
            else:
                self.NN.load_state_dict(torch.load(ff_path_hardware))

        # if np.__version__ != 1.24:
        #     if self.pred_type == 3: #Jax Predictor
        #         print("Can't use Jax. Requires numpy version >= 1.24")
        #         exit(1)
        # elif np.__version__ == 1.24:
        #     if self.pred_type == 3: #Jax Predictor
        #         print("Using Jax")
        #         if self.nonlin0:
        #             print("Using Jax 0 Order Hold Predictor")
        #         else:
        #             print("Using Jax 1stOrder Hold Predictor")
        #             self.udot = np.array([[0, 0, 0, 0]], dtype=np.float64).T
        # else:
        #     print("Can't use Jax. requires numpy version >= 1.24")

        self.metadata = np.array(['Sim' if self.sim else 'Hardware',
                                  'Jax' if self.pred_type == 3 else 'NN' if self.pred_type == 2 else 'Linear' if self.pred_type == 1 else 'Nonlinear',
                                  '2x Speed' if self.double_speed else '1x Speed',
                                  '0OrderHold' if self.nonlin0 else '1stOrderHold',
                                  'QuatYawError' if self.use_quat_yaw else 'EulerYawError',
                                  'PyJoules' if self.pyjoules_on else 'NoPyJoules',
                                  ])
###############################################################################################################################################

        #Create Function @ {1/self.offboard_timer_period}Hz (in my case should be 10Hz/0.1 period) to Publish Offboard Control Heartbeat Signal
        self.offboard_timer_period = 0.1
        self.timer = self.create_timer(self.offboard_timer_period, self.offboard_mode_timer_callback)

        # Create Function at {1/self.newton_raphson_timer_period}Hz (in my case should be 100Hz/0.01 period) to Send NR Control Input
        self.newton_raphson_timer_period = 0.01
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
        """ Convert the collective thrust to a throttle command. """
        collective_thrust = -collective_thrust
        # print(f"Conv2Throttle: collective_thrust: {collective_thrust}")
        if self.sim:
            motor_speed = m.sqrt(collective_thrust / (4.0 * self.MOTOR_CONSTANT))
            throttle_command = (motor_speed - self.MOTOR_VELOCITY_ARMED) / self.MOTOR_INPUT_SCALING
            return -throttle_command
        
        if not self.sim:
            a = 0.00705385408507030
            b = 0.0807474474438391
            c = 0.0252575818743285

            # equation form is a*x + b*sqrt(x) + c = y
            throttle_command = a*collective_thrust + b*m.sqrt(collective_thrust) + c
            return -throttle_command

    def get_force_from_throttle_command(self, throttle_command): #Converts throttle command to force
        """ Convert the throttle command to a collective thrust. """
        throttle_command = -throttle_command
        print(f"Conv2Force: throttle_command: {throttle_command}")
        if self.sim:
            motor_speed = (throttle_command * self.MOTOR_INPUT_SCALING) + self.MOTOR_VELOCITY_ARMED
            collective_thrust = 4.0 * self.MOTOR_CONSTANT * motor_speed ** 2
            return -collective_thrust
        
        if not self.sim:
            a = 19.2463167420814
            b = 41.8467162352942
            c = -7.19353022443441

            # equation form is a*x^2 + b*x + c = y
            collective_thrust = a*throttle_command**2 + b*throttle_command + c
            return -collective_thrust
    
    def normalize_angle(self, angle):
        """ Normalize the angle to the range [-pi, pi]. """
        return m.atan2(m.sin(angle), m.cos(angle))

    # def _reorder_input_quaternion(self, quaternion):
    #     """Reorder quaternion to have w term first."""
    #     # x, y, z, w = quaternion
    #     return quaternion

    # def quaternion_matrix(self, quaternion):
    #     """
    #     Return homogeneous rotation matrix from quaternion.

    #     >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    #     >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    #     True

    #     """
    #     TRANSLATION_IDENTITY = [0.0, 0.0, 0.0]
    #     ROTATION_IDENTITY = np.identity(3, dtype=np.float64)
    #     ZOOM_IDENTITY = [1.0, 1.0, 1.0]
    #     SHEAR_IDENTITY = TRANSLATION_IDENTITY
    #     rotation_matrix = transforms3d.quaternions.quat2mat(
    #         self._reorder_input_quaternion(quaternion)
    #     )
    #     return transforms3d.affines.compose(TRANSLATION_IDENTITY,
    #                                         rotation_matrix,
    #                                         ZOOM_IDENTITY)

    # def euler_from_matrix(self, matrix, axes='sxyz'):
    #     """
    #     Return Euler angles from rotation matrix for specified axis sequence.

    #     axes : One of 24 axis sequences as string or encoded tuple

    #     Note that many Euler angle triplets can describe one matrix.

    #     >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    #     >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    #     >>> R1 = euler_matrix(al, be, ga, 'syxz')
    #     >>> numpy.allclose(R0, R1)
    #     True
    #     >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    #     >>> for axes in _AXES2TUPLE.keys():
    #     ...    R0 = euler_matrix(axes=axes, *angles)
    #     ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    #     ...    if not numpy.allclose(R0, R1): print axes, "failed"

    #     """
    #     return transforms3d.euler.mat2euler(matrix, axes=axes)


    # def euler_from_quaternion(self, quaternion, axes='sxyz'):
    #     """
    #     Return Euler angles from quaternion for specified axis sequence.

    #     >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    #     >>> numpy.allclose(angles, [0.123, 0, 0])
    #     True

    #     """
    #     return self.euler_from_matrix(self.quaternion_matrix(quaternion), axes)
    
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

        # print(f"{msg.q = }")
        # self.roll, self.pitch, yaw = quat2euler(msg.q)
        self.roll, self.pitch, yaw = self.xeuler_from_quaternion(*msg.q)
        # self.roll, self.pitch, yaw = self.euler_from_quaternion(msg.q)
        self.yaw = self.adjust_yaw(yaw)

        self.p = msg.angular_velocity[0]
        self.q = msg.angular_velocity[1]
        self.r = msg.angular_velocity[2]

        self.state_vector = np.array([[self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw]]).T 
        self.nr_state = np.array([[self.x, self.y, self.z, self.yaw]]).T
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
        msg.thrust_body[2] = 1* float(thrust)

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
        # print("NR_Callback")
        if self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard deciding whether to send heartbeat signal, engage offboard, and arm
            # self.time_from_start = time.time()-self.T0 #update curent time from start of program for reference trajectories and for switching between NR and landing mode
            
            print(f"--------------------------------------")
            # print(self.vehicle_status.nav_state)
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
            reffunc = self.circle_horz_ref_func()
            # reffunc = self.circle_horz_spin_ref_func()
            # reffunc = self.circle_vert_ref_func()
            # reffunc = self.fig8_horz_ref_func()
            # reffunc = self.fig8_vert_ref_func_short()
            # reffunc = self.fig8_vert_ref_func_tall()
            # reffunc = self.helix()
            # reffunc = self.helix_spin()
            # reffunc = self.triangle()
            # reffunc = self.sawtooth()

        elif self.cushion_time + self.flight_time <= self.time_from_start <= self.time_before_land:
            reffunc = self.hover_ref_func(1)
        else:
            reffunc = self.hover_ref_func(1)


        # reffunc = self.hover_ref_func(1)
        # reffunc = self.sawtooth()
        # reffunc = self.triangle()
        # reffunc = self.yawing_only()

        # reffunc = self.circle_horz_ref_func()
        # reffunc = self.circle_horz_spin_ref_func()
        # reffunc = self.circle_vert_ref_func()
        # reffunc = self.fig8_horz_ref_func()
        # reffunc = self.fig8_vert_ref_func_short()
        # reffunc = self.fig8_vert_ref_func_tall()
        # reffunc = self.helix()
        # reffunc = self.helix_spin()
        # reffunc = self.yawing_only()

        print(f"reffunc: {reffunc}")

        # Calculate the Newton-Rapshon control input and transform the force into a throttle command for publishing to the vehicle
        new_u = self.get_new_NR_input(last_input_using_force, reffunc)
        # print(f"new_u: {new_u}")
        new_force = new_u[0][0]        
        print(f"new_force: {new_force}")

        new_throttle = self.get_throttle_command_from_force(new_force)
        new_throttle = float(new_throttle)  # Already a float, but for clarity
        new_roll_rate = float(new_u[1][0])  # Convert jax.numpy array to float
        new_pitch_rate = float(new_u[2][0])  # Convert jax.numpy array to float
        new_yaw_rate = float(new_u[3][0])    # Convert jax.numpy array to float


        # Build the final input vector to save as self.u0 and publish to the vehicle via publish_rates_setpoint:
        final = [new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate] # final input vector
        current_input_save = np.array(final).reshape(-1, 1) # reshaped to column vector for saving as class variable
        self.u0 = current_input_save # saved as class variable for next iteration of NR control
        # print(f"newInput: \n{current_input_save}")
        # print(f"final: {final}")

        # final = 4*[float(0.0)]
        print(f"{final = }")
        self.publish_rates_setpoint(final[0], final[1], final[2], final[3])
        
        # exit(1)
        # Log the states, inputs, and reference trajectories for data analysis
        controller_callback_time = time.time() - t0
        state_input_ref_log_info = [float(self.x), float(self.y), float(self.z), float(self.yaw), float(final[0]), float(final[1]), float(final[2]), float(final[3]), float(reffunc[0][0]), float(reffunc[1][0]), float(reffunc[2][0]), float(reffunc[3][0]), self.time_from_start, controller_callback_time]
        self.update_logged_data(state_input_ref_log_info)
        # self.state_input_ref_log_msg.data = state_input_ref_log_info
        # self.state_input_ref_log_publisher_.publish(self.state_input_ref_log_msg)
        # exit(0)


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
    
    def get_new_NR_input_execution_function(self,last_input,ref): # Executes Newton-Rapshon Control Algorithm -- gets called by get_new_NR_input either with or without pyjoules
        """ Calculates the Newton-Raphson Control Input. """
        nrt0 = time.time() # time before NR calculations
        currstate = jnp.array([self.state_vector[0][0], self.state_vector[1][0], self.state_vector[2][0], self.state_vector[3][0], self.state_vector[4][0], self.state_vector[5][0], self.state_vector[6][0], self.state_vector[7][0], self.state_vector[8][0]])
        last_input = jnp.array([-last_input[0][0], last_input[1][0], last_input[2][0], last_input[3][0]])

        alpha=np.array([[20,30,30,30]]).T # Speed-up parameter (maybe play with uniform alpha values rather than ones that change for each input)

        pred = predict_output(currstate, last_input, self.T_LOOKAHEAD, self.INTEGRATION_STEP, self.C)
        error = error_func(currstate, last_input, self.T_LOOKAHEAD, self.INTEGRATION_STEP, self.C, ref)
        print(f"{pred = }")
        print(f"{ref = }")
        print(f"{error = }")

        dgdu = get_grad_error_u(currstate, last_input, self.T_LOOKAHEAD, self.INTEGRATION_STEP, self.C, ref)
        hess = get_hess_error_u(currstate, last_input, self.T_LOOKAHEAD, self.INTEGRATION_STEP, self.C, ref)
        d2gdu2_inv = jnp.linalg.inv(hess) if hess.shape[0] > 1 else 1/hess
        print(f"{dgdu = }")
        print(f"{hess = }")
        print(f"{d2gdu2_inv = }")


        

        scaling = d2gdu2_inv @ dgdu
        scaling = scaling.reshape(1,)
        print(f"{scaling = }")
        NR = scaling
        udot = -alpha * NR
        print(f"{udot = }")

        change_u = udot * self.newton_raphson_timer_period
        print(f"{change_u = }")

        print(f"{last_input = }")
        new_u = last_input + change_u
        print(f"{new_u = }")


        nr_time_elapsed = time.time()-nrt0
        self.nr_timel_array.append(nr_time_elapsed)
        print(f"NR_time_elapsed: {nr_time_elapsed}, Good For {1/nr_time_elapsed} Hz")
        self.udot = udot
        return new_u


# ~~ The following functions have the various non-linear models and neural networks for predicting the system output state ~~
    def get_jax_predict(self, last_input): #Predicts System Output State Using Jax
        """ Predicts the system output state using a numerically integrated nonlinear model. """
        if self.nonlin0: 
            print(f"0 order hold jax")
            # t1 = time.time()
            STATE = jnp.array([self.state_vector[0][0], self.state_vector[1][0], self.state_vector[2][0], self.state_vector[3][0], self.state_vector[4][0], self.state_vector[5][0], self.state_vector[6][0], self.state_vector[7][0], self.state_vector[8][0]])
            INPUT = jnp.array([-last_input[0][0], last_input[1][0], last_input[2][0], last_input[3][0]])
            # print(f"STATE: \n{STATE}")
            outputs = predict_outputs(STATE, INPUT, self.T_LOOKAHEAD, self.GRAVITY, self.MASS, self.C, integration_step=0.1)
            # print(f"Outputs: \n{outputs}")
            adjusted_invjac = compute_adjusted_invjac(STATE, INPUT, self.T_LOOKAHEAD, self.GRAVITY, self.MASS, self.C, integration_step=0.1)
            # print(f"adjusted_invjac: \n{adjusted_invjac}")


            outputs = np.array(outputs).reshape(-1, 1)
            # print(f"{outputs = }")
            # print(f"{type(outputs) = }")
            adjusted_invjac = np.array(adjusted_invjac)
            # print(f"{adjusted_invjac = }")
            # print(f"{type(adjusted_invjac) = }")
            self.jac_inv = adjusted_invjac

            # exit(0)
            return outputs
        
        elif not self.nonlin0:
            print(f"1 order hold jax")
            STATE = jnp.array([self.state_vector[0][0], self.state_vector[1][0], self.state_vector[2][0], self.state_vector[3][0], self.state_vector[4][0], self.state_vector[5][0], self.state_vector[6][0], self.state_vector[7][0], self.state_vector[8][0]])
            INPUT = jnp.array([-last_input[0][0], last_input[1][0], last_input[2][0], last_input[3][0]])
            INPUT_DERIVS = jnp.array([self.udot[0][0], self.udot[1][0], self.udot[2][0], self.udot[3][0]])
            outputs = predict_outputs_1order(STATE, INPUT, INPUT_DERIVS, self.T_LOOKAHEAD, self.GRAVITY, self.MASS, self.C, integration_step=0.1)
            adjusted_invjac = compute_adjusted_invjac_1order(STATE, INPUT, INPUT_DERIVS, self.T_LOOKAHEAD, self.GRAVITY, self.MASS, self.C, integration_step=0.1)

            outputs = np.array(outputs).reshape(-1, 1)
            # print(f"{outputs = }")
            # print(f"{type(outputs) = }")
            adjusted_invjac = np.array(adjusted_invjac)
            # print(f"{adjusted_invjac = }")
            # print(f"{type(adjusted_invjac) = }")
            self.jac_inv = adjusted_invjac
            return outputs
        
    def get_nn_predict(self, last_input): #Predicts System Output State Using Neural Network
        position_now = np.array(self.state_vector.T.tolist()[0])
        # position_now[-1] = self.adjust_yaw(position_now[-1])
        curr_thrust = -last_input[0][0]
        curr_rolldot = last_input[1][0]
        curr_pitchdot = last_input[2][0]
        curr_yawdot = last_input[3][0]
        
        input_data = np.array([curr_thrust, curr_rolldot, curr_pitchdot, curr_yawdot])
        
        outputNN = self.apply_model(position_now, input_data)
        inv_jac, cond, cond2 = self.compute_inv_jac(position_now, input_data)

        outputs = np.array(outputNN).reshape(-1, 1)
        print(f"\n")
        print(f"{cond = }")
        print(f"{cond2 = }")
        print(f"\n")

        self.jac_inv = inv_jac
        return outputs




    def get_nn_predict2(self, last_input): #Predicts System Output State Using Neural Network
        """ Predicts the system output state using a feedforward neural network. """
        position_now = self.state_vector.T.tolist()[0]
        curr_thrust = -last_input[0][0]
        curr_rolldot = last_input[1][0]
        curr_pitchdot = last_input[2][0]
        curr_yawdot = last_input[3][0]
        input_data = [curr_thrust, curr_rolldot, curr_pitchdot, curr_yawdot]
        # print(f"position_now: {position_now}")
        # print(f"input_data: {input_data}")

        # Concatenate state vector and input vector
        state_vector = torch.tensor(position_now, dtype=torch.float32)
        input_vector = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)
        dataNN = torch.cat([state_vector, input_vector])
        # print(f"dataNN: {dataNN}")

        # print("Feed Forward NN")
        # t1 = time.time()
        outputNN = self.NN(dataNN)
        # print(f"outputNN: {outputNN}")

        # Compute Jacobian
        jacobian = torch.zeros((4, 4))
        # print(input_vector.grad)
        # input_vector.grad.zero_()

        for i in range(4):
            self.NN.zero_grad()  # Reset gradients to zero
            if outputNN.grad is not None:
                outputNN.grad.zero_()
            outputNN[i].backward(retain_graph=True)  # Compute gradients
            jacobian[i] = input_vector.grad

        # print("Jacobian Matrix:\n", jacobian)

        inv_jac = np.linalg.inv(jacobian)
        # print(f"inv_jac: {inv_jac}")
        inv_jac[:, 2] = -inv_jac[:, 2]
        # print(f"inv_jac: {inv_jac}")

        self.jac_inv = inv_jac

        outputNN = outputNN.detach().numpy()
        outputNN = np.array([[outputNN[0], outputNN[1], outputNN[2], outputNN[3]]]).T
        # print(f"{outputNN=}")
        # print(f"outputNN.shape: {outputNN.shape}")
        # print(f"outputNN: {outputNN}")
        return outputNN
        # exit(0)
    
    def get_nonlin_predict_compiled(self, last_input): #Predicts System Output State Using Numerically Integrated Nonlinear Model
        """ Compiled in C: Predicts the system output state using a numerically integrated nonlinear model. """

        # t1 = time.time()
        g = self.GRAVITY
        m = self.MASS
        curr_thrust = -last_input[0][0]
        curr_rolldot = last_input[1][0]
        curr_pitchdot = last_input[2][0]
        curr_yawdot = last_input[3][0]

        # print(f"{self.state_vector = }")
        # print(f"[curr_thrust, curr_rolldot, curr_pitchdot, curr_yawdot] = [{curr_thrust, curr_rolldot, curr_pitchdot, curr_yawdot}]")


        curr_x = self.x
        curr_y = self.y
        curr_z = self.z
        curr_vx = self.vx
        curr_vy = self.vy
        curr_vz = self.vz
        curr_roll = self.roll
        curr_pitch = self.pitch
        curr_yaw = self.yaw

        T_lookahead = self.T_LOOKAHEAD
        integration_step = 0.1
        integrations = T_lookahead / integration_step
        integrations_int = int(integrations)
        
        # print(f"{self.stateVector = }")
        # print(f"[curr_thrust, curr_rolldot, curr_pitchdot, curr_yawdot] = [{curr_thrust, curr_rolldot, curr_pitchdot, curr_yawdot}]")
        # print(f"[thrust_dot, roll_dd, pitch_dd, yaw_dd] = [{thrust_dot, roll_dd, pitch_dd, yaw_dd}]")

        if self.nonlin0:
            # Call the C function for non-linear prediction of outputs
            output_ptr = self.my_library.performCalculations(
                g, m, curr_thrust, curr_rolldot, curr_pitchdot, curr_yawdot,
                curr_x, curr_y, curr_z, curr_vx, curr_vy, curr_vz, curr_roll, curr_pitch, curr_yaw, integration_step, integrations_int
            )
            # Dereference the output pointer to access the vector
            nonlin_pred = output_ptr.contents
            x = nonlin_pred.x
            y = nonlin_pred.y
            z = nonlin_pred.z
            vx = nonlin_pred.vx
            vy = nonlin_pred.vy
            vz = nonlin_pred.vz
            roll = nonlin_pred.roll
            pitch = nonlin_pred.pitch
            yaw = nonlin_pred.yaw
            pred = np.array([[x, y, z, vx, vy, vz, roll, pitch, yaw]]).T
            outputs = self.C @ pred
            print(f"outputs: {outputs}")
            # Free the allocated memory
            self.my_library.free(output_ptr)

            # print(f"pred_z: {z}")
            # Call the C function 4 more times to get jacobian wrt inputs
            epsilon = 1e-5

            pertub_thrust = curr_thrust + epsilon
            output_ptr = self.my_library.performCalculations(
                g, m, pertub_thrust, curr_rolldot, curr_pitchdot, curr_yawdot,
                curr_x, curr_y, curr_z, curr_vx, curr_vy, curr_vz, curr_roll, curr_pitch, curr_yaw, integration_step, integrations_int
            )
            nonlin_pred = output_ptr.contents
            x_var1 = nonlin_pred.x
            y_var1 = nonlin_pred.y
            z_var1 = nonlin_pred.z
            yaw_var1 = nonlin_pred.yaw
            self.my_library.free(output_ptr)
            dpdu1 = np.array([[(x_var1 - x) / epsilon,
                            (y_var1 - y) / epsilon,
                            (z_var1 - z) / epsilon,
                            (yaw_var1 - yaw) / epsilon]]).T


            perturb_rolldot = curr_rolldot + epsilon
            output_ptr = self.my_library.performCalculations(
                g, m, curr_thrust, perturb_rolldot, curr_pitchdot, curr_yawdot,
                curr_x, curr_y, curr_z, curr_vx, curr_vy, curr_vz, curr_roll, curr_pitch, curr_yaw, integration_step, integrations_int
            )
            nonlin_pred = output_ptr.contents
            x_var2 = nonlin_pred.x
            y_var2 = nonlin_pred.y
            z_var2 = nonlin_pred.z
            yaw_var2 = nonlin_pred.yaw
            self.my_library.free(output_ptr)
            dpdu2 = np.array([[(x_var2 - x) / epsilon,
                            (y_var2 - y) / epsilon,
                            (z_var2 - z) / epsilon,
                            (yaw_var2 - yaw) / epsilon]]).T



            perturb_pitchdot = curr_pitchdot + epsilon
            output_ptr = self.my_library.performCalculations(
                g, m, curr_thrust, curr_rolldot, perturb_pitchdot, curr_yawdot,
                curr_x, curr_y, curr_z, curr_vx, curr_vy, curr_vz, curr_roll, curr_pitch, curr_yaw, integration_step, integrations_int
            )
            nonlin_pred = output_ptr.contents
            x_var3 = nonlin_pred.x
            y_var3 = nonlin_pred.y
            z_var3 = nonlin_pred.z
            yaw_var3 = nonlin_pred.yaw
            self.my_library.free(output_ptr)
            dpdu3 = np.array([[(x_var3 - x) / epsilon,
                            (y_var3 - y) / epsilon,
                            (z_var3 - z) / epsilon,
                            (yaw_var3 - yaw) / epsilon]]).T


            perturb_yawdot = curr_yawdot + epsilon
            output_ptr = self.my_library.performCalculations(
                g, m, curr_thrust, curr_rolldot, curr_pitchdot, perturb_yawdot,
                curr_x, curr_y, curr_z, curr_vx, curr_vy, curr_vz, curr_roll, curr_pitch, curr_yaw, integration_step, integrations_int
            )
            nonlin_pred = output_ptr.contents
            x_var4 = nonlin_pred.x
            y_var4 = nonlin_pred.y
            z_var4 = nonlin_pred.z
            yaw_var4 = nonlin_pred.yaw
            self.my_library.free(output_ptr)
            dpdu4 = np.array([[(x_var4 - x) / epsilon,
                            (y_var4 - y) / epsilon,
                            (z_var4 - z) / epsilon,
                            (yaw_var4 - yaw) / epsilon]]).T



            jac_u = np.hstack([dpdu1, dpdu2, dpdu3, dpdu4])
            # print(f"dpdu3: {dpdu3}")
            # print(f"jac_u: {jac_u}")
            # print(f"jac_u.shape: {jac_u.shape}")
            # make the third column of inverse jacobian negative
            inv_jac = np.linalg.inv(jac_u)
            # print(f"inv_jac: {inv_jac}")
            inv_jac[:, 2] = -inv_jac[:, 2]
            # print(f"inv_jac: {inv_jac}")

            self.jac_inv = inv_jac
            # print(f"jac_inv:  {self.jac_inv}")
            # exit(0)

        else:
            thrust_dot = self.udot[0][0]
            roll_dd = self.udot[1][0]
            pitch_dd = self.udot[2][0]
            yaw_dd = self.udot[3][0]
            # Call the C function for non-linear prediction of outputs
            output_ptr = self.my_library.performCalculations(
                g, m, curr_thrust, curr_rolldot, curr_pitchdot, curr_yawdot,
                curr_x, curr_y, curr_z, curr_vx, curr_vy, curr_vz, curr_roll, curr_pitch, curr_yaw, integration_step, integrations_int,
                thrust_dot, roll_dd, pitch_dd, yaw_dd
            )
            # Dereference the output pointer to access the vector
            nonlin_pred = output_ptr.contents
            x = nonlin_pred.x
            y = nonlin_pred.y
            z = nonlin_pred.z
            vx = nonlin_pred.vx
            vy = nonlin_pred.vy
            vz = nonlin_pred.vz
            roll = nonlin_pred.roll
            pitch = nonlin_pred.pitch
            yaw = nonlin_pred.yaw
            pred = np.array([[x, y, z, vx, vy, vz, roll, pitch, yaw]]).T
            outputs = self.C @ pred
            # print(f"outputs: {outputs}")

            # Free the allocated memory
            self.my_library.free(output_ptr)

            # print(f"pred_z: {z}")
            # Call the C function 4 more times to get jacobian wrt inputs
            epsilon = 1e-5

            pertub_thrust = curr_thrust + epsilon
            # Call the C function for non-linear prediction of outputs
            output_ptr = self.my_library.performCalculations(
                g, m, pertub_thrust, curr_rolldot, curr_pitchdot, curr_yawdot,
                curr_x, curr_y, curr_z, curr_vx, curr_vy, curr_vz, curr_roll, curr_pitch, curr_yaw, integration_step, integrations_int,
                thrust_dot, roll_dd, pitch_dd, yaw_dd
            )
            nonlin_pred = output_ptr.contents
            x_var1 = nonlin_pred.x
            y_var1 = nonlin_pred.y
            z_var1 = nonlin_pred.z
            yaw_var1 = nonlin_pred.yaw
            self.my_library.free(output_ptr)
            dpdu1 = np.array([[(x_var1 - x) / epsilon,
                            (y_var1 - y) / epsilon,
                            (z_var1 - z) / epsilon,
                            (yaw_var1 - yaw) / epsilon]]).T


            perturb_rolldot = curr_rolldot + epsilon
            output_ptr = self.my_library.performCalculations(
                g, m, pertub_thrust, perturb_rolldot, curr_pitchdot, curr_yawdot,
                curr_x, curr_y, curr_z, curr_vx, curr_vy, curr_vz, curr_roll, curr_pitch, curr_yaw, integration_step, integrations_int,
                thrust_dot, roll_dd, pitch_dd, yaw_dd
            )
            nonlin_pred = output_ptr.contents
            x_var2 = nonlin_pred.x
            y_var2 = nonlin_pred.y
            z_var2 = nonlin_pred.z
            yaw_var2 = nonlin_pred.yaw
            self.my_library.free(output_ptr)
            dpdu2 = np.array([[(x_var2 - x) / epsilon,
                            (y_var2 - y) / epsilon,
                            (z_var2 - z) / epsilon,
                            (yaw_var2 - yaw) / epsilon]]).T



            perturb_pitchdot = curr_pitchdot + epsilon
            output_ptr = self.my_library.performCalculations(
                g, m, pertub_thrust, perturb_rolldot, perturb_pitchdot, curr_yawdot,
                curr_x, curr_y, curr_z, curr_vx, curr_vy, curr_vz, curr_roll, curr_pitch, curr_yaw, integration_step, integrations_int,
                thrust_dot, roll_dd, pitch_dd, yaw_dd
            )
            nonlin_pred = output_ptr.contents
            x_var3 = nonlin_pred.x
            y_var3 = nonlin_pred.y
            z_var3 = nonlin_pred.z
            yaw_var3 = nonlin_pred.yaw
            self.my_library.free(output_ptr)
            dpdu3 = np.array([[(x_var3 - x) / epsilon,
                            (y_var3 - y) / epsilon,
                            (z_var3 - z) / epsilon,
                            (yaw_var3 - yaw) / epsilon]]).T


            perturb_yawdot = curr_yawdot + epsilon
            output_ptr = self.my_library.performCalculations(
                g, m, pertub_thrust, perturb_rolldot, perturb_pitchdot, perturb_yawdot,
                curr_x, curr_y, curr_z, curr_vx, curr_vy, curr_vz, curr_roll, curr_pitch, curr_yaw, integration_step, integrations_int,
                thrust_dot, roll_dd, pitch_dd, yaw_dd
            )
            nonlin_pred = output_ptr.contents
            x_var4 = nonlin_pred.x
            y_var4 = nonlin_pred.y
            z_var4 = nonlin_pred.z
            yaw_var4 = nonlin_pred.yaw
            self.my_library.free(output_ptr)
            dpdu4 = np.array([[(x_var4 - x) / epsilon,
                            (y_var4 - y) / epsilon,
                            (z_var4 - z) / epsilon,
                            (yaw_var4 - yaw) / epsilon]]).T



            jac_u = np.hstack([dpdu1, dpdu2, dpdu3, dpdu4])
            # print(f"dpdu3: {dpdu3}")
            # print(f"jac_u: {jac_u}")
            # print(f"jac_u.shape: {jac_u.shape}")
            # make the third column of inverse jacobian negative
            inv_jac = np.linalg.inv(jac_u)
            # print(f"inv_jac: {inv_jac}")
            inv_jac[:, 2] = -inv_jac[:, 2]
            # print(f"inv_jac: {inv_jac}")

            self.jac_inv = inv_jac
            # print(f"jac_inv:  {self.jac_inv}")

        return outputs
    
    def get_nonlin_predict_0order_python(self, last_input): # DO NOT USE -- NOT FAST ENOUGH FOR 100Hz. HERE FOR REFERENCE FOR COMPILED 0-ORDER HOLD NONLINEAR PREDICTION
        """ DO NOT USE (SLOW- not compiled in C): Predicts the system output state using a numerically integrated nonlinear model with 0-order hold. """
        
        g = self.GRAVITY
        m = self.MASS
        curr_thrust = -last_input[0][0]
        curr_rolldot = last_input[1][0]
        curr_pitchdot = last_input[2][0]
        curr_yawdot = last_input[3][0]

        curr_x = self.x
        curr_y = self.y
        curr_z = self.z
        curr_vx = self.vx
        curr_vy = self.vy
        curr_vz = self.vz
        curr_roll = self.roll
        curr_pitch = self.pitch
        curr_yaw = self.yaw

        T_lookahead = 0.8
        integration_step = 0.1
        integrations = T_lookahead / integration_step
        integrations = int(integrations)

        cumm_change_x = 0.0
        cumm_change_y = 0.0
        cumm_change_z = 0.0
        cumm_change_vx = 0.0
        cumm_change_vy = 0.0
        cumm_change_vz = 0.0
        cumm_change_roll = 0.0
        cumm_change_pitch = 0.0
        cumm_change_yaw = 0.0

        xdot = curr_vx
        ydot = curr_vy
        zdot = curr_vz

        vxdot = -(curr_thrust/m) * (m.sin(curr_roll)*m.sin(curr_yaw) + m.m.cos(curr_roll)*m.cos(curr_yaw)*m.sin(curr_pitch));
        vydot = -(curr_thrust/m) * (m.cos(curr_roll)*m.sin(curr_yaw)*m.sin(curr_pitch) - m.m.cos(curr_yaw)*m.sin(curr_roll));
        vzdot = g - (curr_thrust/m) * (m.cos(curr_roll)*m.cos(curr_pitch));
    
        rolldot = curr_rolldot
        pitchdot = curr_pitchdot
        yawdot = curr_yawdot

        roll = curr_roll
        pitch = curr_pitch
        yaw = curr_yaw
        change_vx = 0
        change_vy = 0
        change_vz = 0



        for _ in range(integrations):
            change_x = (xdot+cumm_change_vx) * integration_step;
            change_y = (ydot+cumm_change_vy) * integration_step;
            change_z = (zdot+cumm_change_vz) * integration_step;
            change_vx = vxdot * integration_step;
            change_vy = vydot * integration_step;
            change_vz = vzdot * integration_step;
            change_roll = rolldot * integration_step;
            change_pitch = pitchdot * integration_step;
            change_yaw = yawdot * integration_step;


            roll = roll + change_roll;
            pitch = pitch + change_pitch;
            yaw =  yaw + change_yaw;

            sr = m.sin(roll)
            sy = m.sin(yaw)
            sp = m.sin(pitch)

            cr = m.cos(roll)
            cp = m.cos(pitch)
            cy = m.cos(yaw)
            
            
            vxdot = -(curr_thrust/m) * (sr*sy + cr*cy*sp);
            vydot = -(curr_thrust/m) * (cr*sy*sp - cy*sr);
            vzdot = g - (curr_thrust/m) * (cr*cp);


            cumm_change_x = cumm_change_x + change_x;
            cumm_change_y = cumm_change_y + change_y; 
            cumm_change_z = cumm_change_z + change_z; 
            cumm_change_vx = cumm_change_vx + change_vx; 
            cumm_change_vy = cumm_change_vy + change_vy; 
            cumm_change_vz = cumm_change_vz + change_vz; 
            cumm_change_roll = cumm_change_roll + change_roll; 
            cumm_change_pitch = cumm_change_pitch + change_pitch; 
            cumm_change_yaw = cumm_change_yaw + change_yaw;




        x = curr_x + cumm_change_x
        y = curr_y + cumm_change_y
        z = curr_z + cumm_change_z
        
        vx = curr_vx + cumm_change_vx
        vy = curr_vy + cumm_change_vy
        vz = curr_vz + cumm_change_vz
    
        roll = curr_roll + cumm_change_roll;
        pitch = curr_pitch + cumm_change_pitch;
        yaw = curr_yaw + cumm_change_yaw;

        nonlin_pred = np.array([[x, y, z, vx, vy, vz, roll, pitch, yaw]]).T
        outputs = self.C @ nonlin_pred
        return outputs
    


# ~~ The following functions produce the linearized model matrices that we use for prediction and NR input calculation ~~
    def getyorai_g_linear_predict(self, curr_input): #Calculates Linearized System Output Prediction ([x,y,z,yaw])
        """ Predicts the system output state using the closed form equation for an LTI system.  """
        gravity = np.array([[self.MASS * self.GRAVITY, 0, 0, 0]]).T #gravity vector that counteracts input vector: [-mg, 0, 0, 0]
        # unforced = self.eAT @ self.state_vector
        # print(f"{unforced = }")
        # forced = self.int_eATB @ (curr_input + gravity)
        # print(f"{forced = }")
        lin_pred = self.eAT@self.state_vector + self.int_eATB @ (curr_input + gravity) # xdot = eAT*x(t) + int_eATB*(u-gravity)
        yorai_g = self.C @ lin_pred # y(t) = C*x(t) = [x,y,z,yaw]
        return yorai_g

    def getyorai_gJac_linear_predict(self): #Calculates Jacobian of Linearized System Output Prediction for Newton-Raphson Input Calc : udot = a*inv(dg/du)(err)
        """ Calculates the Jacobian of the closed form linearized system output prediction wrt inputs at the operating point of hover above origin with euler angles 0. """
        Jac = np.concatenate((self.int_eATB[0:3, :], self.int_eATB[-1:, :]), axis=0) # dg/du = C * int_eATB
        print(f"{Jac =}")
        return Jac
    
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

        A = smp.Matrix(
            [
                [0, 0, 0,    1, 0, 0,     0,   0, 0],
                [0, 0, 0,    0, 1, 0,     0,   0, 0],
                [0, 0, 0,    0, 0, 1,     0,   0, 0],

                [0, 0, 0,    0, 0, 0,     0,-1*self.GRAVITY, 0],
                [0, 0, 0,    0, 0, 0,     self.GRAVITY,   0, 0],
                [0, 0, 0,    0, 0, 0,     0,   0, 0],

                [0, 0, 0,    0, 0, 0,     0,   0, 0],
                [0, 0, 0,    0, 0, 0,     0,   0, 0],
                [0, 0, 0,    0, 0, 0,     0,   0, 0],

            ]
            )
        
        eAT = smp.exp(A*self.T_LOOKAHEAD)

        B = smp.Matrix(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],

                [1/self.MASS, 0, 0, 0],
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
                f = lambda x: sp_linalg.expm(A*(self.T_LOOKAHEAD-x))[row,column]
                int_eAT[row,column] = sp_int.quad(f, 0, self.T_LOOKAHEAD)[0]


        int_eATB = int_eAT @ B
        eAT = np.array(eAT).astype(np.float64)
        int_eATB = np.array(int_eATB).astype(np.float64)

        # print(f"A: \n {A}")
        # print(f"eAT: \n {eAT}")
        # print(f"int_eAT: \n {int_eAT}")

        # print(f"B: \n {B}")
        # print(f"int_eATB: \n {int_eATB}")
        # print(f"C: \n {C}")
        self.eAT = eAT
        self.int_eATB = int_eATB
        self.int_eAT = int_eAT
        self.jac_inv = np.linalg.inv(self.getyorai_gJac_linear_predict()) #Calculate Inverse Jacobian of Linearized Model Matrices



# ~~ The following functions are reference trajectories for tracking ~~
    def hover_ref_func(self, num): #Returns Constant Hover Reference Trajectories At A Few Different Positions ([x,y,z,yaw])
        """ Returns constant hover reference trajectories at a few different positions. """
        hover_dict = {
            1: np.array([[0.0, 0.0, -0.6, 0.0]]).T,
            2: np.array([[0.0, 0.8, -0.8, 0.0]]).T,
            3: np.array([[0.8, 0.0, -0.8, 0.0]]).T,
            4: np.array([[0.8, 0.8, -0.8, 0.0]]).T,
            5: np.array([[0.0, 0.0, -10.0, 0.0]]).T,
            6: np.array([[1.0, 1.0, -4.0, 0.0]]).T,
            7: np.array([[3.0, 4.0, -5.0, 0.0]]).T,
            8: np.array([[1.0, 1.0, -3.0, 0.0]]).T,
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

        SPIN_PERIOD = 10
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

    def helix(self): #Returns Helix Reference Trajectory ([x,y,z,yaw])
        """ Returns helix reference trajectory. """
        print(f"helix")
        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD

        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637
        PERIOD_Z = 13

        if self.double_speed:
            PERIOD /= 2.0
            PERIOD_Z /= 2.0
        w = 2*m.pi / PERIOD
        w_z = 2*m.pi / PERIOD_Z

        z0 = 0.8
        height_variance = 0.3
        r = np.array([[.6*m.cos(w*t), .6*m.sin(w*t), -1*(z0 + height_variance * m.sin(w_z * t)), 0.0]]).T
        return r

    def helix_spin(self): #Returns Spiral Staircase Reference Trajectories while Spinning ([x,y,z,yaw])
        """ Returns helix reference trajectory while yawing. """
        print(f"helix_spin")
        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD

        PERIOD = 13 # used to have w=.5 which is rougly PERIOD = 4*pi ~= 12.56637

        if self.double_speed:
            PERIOD /= 2.0

        w = 2*m.pi / PERIOD

        PERIOD_Z = 13
        w_z = 2*m.pi / PERIOD_Z
        z0 = 0.8
        height_variance = 0.3

        SPIN_PERIOD = 15
        yaw_ref = (t) / (SPIN_PERIOD / (2*m.pi))

        r = np.array([[.6*m.cos(w*t), .6*m.sin(w*t), -1*(z0 + height_variance * m.sin(w_z * t)), yaw_ref]]).T
        return r
    
    def yawing_only(self): #Returns Yawing Reference Trajectory ([x,y,z,yaw])
        """ Returns yawing reference trajectory. """
        print(f"yawing_only")

        t_traj = self.time_from_start - self.cushion_time
        t = t_traj + self.T_LOOKAHEAD

        SPIN_PERIOD = 15
        
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