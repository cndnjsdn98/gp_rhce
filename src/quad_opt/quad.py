""" Implementation of the Simplified Simulator and its quadrotor dynamics.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from src.utils.utils import parse_xacro_file
import os

def custom_quad_param_loader(quad_name):

    this_path = os.path.dirname(os.path.realpath(__file__))
    params_file = os.path.join(this_path, '..', '..', 'quads', quad_name, quad_name + '.xacro')

    # Get parameters for drone
    attrib = parse_xacro_file(params_file)

    quad = Quadrotor(noisy=False, drag=True, payload=False, motor_noise=False)
    quad.quad_name = quad_name

    quad.mass = (float(attrib['mass']) + float(attrib['mass_rotor']) * 4)
    quad.J = np.array([float(attrib['body_inertia'][0]['ixx']),
                       float(attrib['body_inertia'][0]['iyy']),
                       float(attrib['body_inertia'][0]['izz'])])
    quad.length = float(attrib['arm_length'])

    quad.motor_constant = float(attrib["motor_constant"])
    if 'max_thrust' in attrib.keys() and attrib['max_thrust'] is not None:
        quad.max_thrust = float(attrib['max_thrust'])
    else:
        quad.max_thrust = float(attrib["max_rot_velocity"]) ** 2 * float(attrib["motor_constant"])
    quad.c = float(attrib['moment_constant'])
    quad.rotor_drag_coeff = float(attrib["rotor_drag_coefficient"])

    # Input constraints
    if 'max_input_value' in attrib.keys():
        quad.max_input_value = float(attrib['max_input_value'])  # Motors at full thrust
    if 'min_input_value' in attrib.keys():
        quad.min_input_value = float(attrib['min_input_value'])  # Motors turned off

    # x configuration
    if quad_name != "hummingbird":
        dx = float(attrib['rotor_dx'])
        dy = float(attrib['rotor_dy'])
        dz = float(attrib['rotor_dz'])

        quad.x_f = np.array([dx, -dx, -dx, dx])
        quad.y_f = np.array([-dy, -dy, dy, dy])
        quad.z_l_tau = np.array([-quad.c, quad.c, -quad.c, quad.c])

    # + configuration
    else:
        quad.x_f = np.array([quad.length, 0, -quad.length, 0])
        quad.y_f = np.array([0, quad.length, 0, -quad.length])
        quad.z_l_tau = -np.array([-quad.c, quad.c, -quad.c, quad.c])

    # Compute hover thrust
    quad.hover_thrust = (quad.mass * 9.81) / (quad.max_thrust * 4)

    return quad

class Quadrotor:

    def __init__(self, noisy=False, drag=False, payload=False, motor_noise=False):
        """
        Initialization of the 3D quadrotor class
        :param noisy: Whether noise is used in the simulation
        :type noisy: bool
        :param drag: Whether to simulate drag or not.
        :type drag: bool
        :param payload: Whether to simulate a payload force in the simulation
        :type payload: bool
        :param motor_noise: Whether non-gaussian noise is considered in the motor inputs
        :type motor_noise: bool
        """
        self.quad_name = ""
        
        # Either 'x' or '+'. The xy configuration of the thrusters in the body plane.
        configuration = 'x'

        # Maximum thrust in Newtons of a thruster when rotating at maximum speed.
        self.max_rot_velocity = 838
        self.motor_constant = 1.56252e-06
        self.max_thrust =  6.00318901352 #34.19432 #
        self.rotor_thrust_coeff = 8.54858e-6
        self.rotor_drag_coeff = 8.06428e-05
        self.hover_thrust = 0.2

        # System state space
        self.pos = np.zeros((3,))
        self.vel = np.zeros((3,))
        self.angle = np.array([1., 0., 0., 0.])  # Quaternion format: qw, qx, qy, qz
        self.a_rate = np.zeros((3,))

        # # System state space for GP evaluation.
        # self.gp_pos = None
        # self.gp_vel = None
        # self.gp_angle = None
        # self.gp_a_rate = None

        # Input constraints
        self.max_input_value = 1  # Motors at full thrust
        self.min_input_value = 0  # Motors turned off

        # Quadrotor intrinsic parameters
        self.J = np.array([.03, .03, .06])  # N m s^2 = kg m^2
        self.mass = 1.0  # kg

        # Length of motor to CoG segment
        self.length = 0.47 / 2  # m

        # Positions of thrusters
        if configuration == '+':
            self.x_f = np.array([self.length, 0, -self.length, 0])
            self.y_f = np.array([0, self.length, 0, -self.length])
        elif configuration == 'x':
            h = np.cos(np.pi / 4) * self.length
            self.x_f = np.array([h, -h, -h, h])
            self.y_f = np.array([-h, -h, h, h])

        # For z thrust torque calculation
        self.c = 0.013  # m   (z torque generated by each motor)
        self.z_l_tau = np.array([-self.c, self.c, -self.c, self.c])

        # Gravity vector
        self.g = np.array([[0], [0], [9.81]])  # m s^-2

        # Actuation thrusts
        self.u_noiseless = np.array([0.0, 0.0, 0.0, 0.0])
        self.u = np.array([0.0, 0.0, 0.0, 0.0])  # N

        # Drag coefficients [kg / m]
        self.rotor_drag_xy = 0.3
        self.rotor_drag_z = 0.0  # No rotor drag in the z dimension
        self.rotor_drag = np.array([self.rotor_drag_xy, self.rotor_drag_xy, self.rotor_drag_z])[:, np.newaxis]
        self.aero_drag = 0.08

        self.drag = drag
        self.noisy = noisy
        self.motor_noise = motor_noise

        self.payload_mass = 0.3  # kg
        self.payload_mass = self.payload_mass * payload

