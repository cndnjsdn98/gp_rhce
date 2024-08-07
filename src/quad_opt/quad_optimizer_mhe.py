#!/usr/bin/env python
""" Implementation of the nonlinear optimizer for the data-augmented MHE.

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


import os
import sys
import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
import rospy
from scipy.linalg import block_diag

from src.utils.quad import custom_quad_param_loader
from src.quad_opt.quad_optimizer import QuadOptimizer

class QuadOptimizerMHE(QuadOptimizer):
    def __init__(self, quad, t_mhe=0.5, n_mhe=50, mhe_type="kinematic",
                 q_mhe=None, q0_factor=None, r_mhe=None, 
                 model_name="quad_3d_acados_mhe", solver_options=None,
                 mhe_gpy_ensemble=None, change_mass=0,
                 compile_acados=True):
        """
        :param quad: quadrotor object.
        :type quad: Quadrotor3D
        :param t_mhe: time horizon for MHE optimization.
        :param n_mhe: number of optimization nodes in time horizon for MHE.
        :param mhe_type: define model type to be used for MHE [kinematic, or dynamic].
        :param q_mhe: diagonal of Qe matrix for model mismatch cost of MHE cost function. Must be a numpy array of length 12.
        :param q0_factor: integer to set the arrival cost of MHE as a factor/multiple of q_mhe.
        :param r_mhe: diagonal of Re matrix for measurement mismatch cost of MHE cost function. Must be a numpy array of length ___.
        :param B_x: dictionary of matrices that maps the outputs of the gp regressors to the state space.
        :param model_name: Acados model name.
        :param solver_options: Optional set of extra options dictionary for solvers.
        :param mhe_gpy_ensemble: GPyEnsemble instance to be utilized in MHE
        :param change_mass: Value of varying payload mass 
        # TODO: Change the chage_mass to be boolean
        """
        super().__init__(quad, t_mhe=t_mhe, n_mhe=n_mhe, mhe_type=mhe_type, 
                         mhe_gpy_ensemble=mhe_gpy_ensemble)
        self.mhe_type = mhe_type 
        self.change_mass = change_mass
        self.mhe_with_gpyTorch = mhe_gpy_ensemble is not None

        # Weighted squared error loss function q = (p_xyz, q_wxyz, v_xyz, r_xyz), r = (u1, u2, u3, u4)
        if q_mhe is None:
            # State noise std
            # System Noise
            w_p = np.ones((1,3)) * 0.004
            w_q = np.ones((1,3)) * np.deg2rad(0.0001)
            w_v = np.array([[3, 3, 3]]) * 0.005            # w_v = np.ones((1,3)) * 1
            w_r = np.ones((1,3)) * np.deg2rad(0.5)
            w_d = np.ones((1,3)) * 0.00001 # 0.0000001
            w_a = np.ones((1,3)) * 0.05
            w_m = np.ones((1,1)) * 0.0001
            if mhe_type == "kinematic":
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_a)))
            elif mhe_type == "dynamic":
                if self.mhe_with_gpyTorch:
                    if change_mass != 0:
                        q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_m)))
                    else:
                        q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r)))
                else:
                    if change_mass != 0:
                        q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_d, w_m)))
                    else:
                        q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_d)))
        if r_mhe is None:
            # Measurement noise std
            # Measurement Noise
            v_p = np.ones((1,3)) * 0.002                 # Position (vicon)
            v_r = np.array([[10, 1, 1]]) * 1.218e-06
            # v_r = np.ones((1,3)) * 1.218e-01    # Angular Velocity
            v_a = np.ones((1,3)) * 8.999e-05                # Acceleration
            v_d = np.ones((1,3)) * 0.0001            # Disturbance
            # Inverse covariance
            if mhe_type == "dynamic" and self.mhe_with_gpyTorch:
                r_mhe = 1/np.squeeze(np.hstack((v_p, v_r, v_d))) 
            elif mhe_type == "dynamic" and self.mhe_with_gpyTorch:
                r_mhe = 1/np.squeeze(np.hstack((v_p, v_r)))
            else:
                r_mhe = 1/np.squeeze(np.hstack((v_p, v_r, v_a))) 

        if q0_factor is None:
            q0_factor = 1
        self.x0_bar = None
        print("\n###########################################################################################")
        print("Q_estimate         = ", q_mhe)
        print("Q_arrival_cost     = ", q_mhe * q0_factor)
        print("R_estimate         = ", r_mhe)
        print("###########################################################################################\n")

        # Add one more weight to the rotation (use quaternion norm weighting in acados)
        q_mhe = np.concatenate((q_mhe[:3], np.mean(q_mhe[3:6])[np.newaxis], q_mhe[3:]))

        self.T_mhe = t_mhe  # Time horizon for MHE
        self.N_mhe = n_mhe  # number of nodes within estimation horizon


        if mhe_type == "kinematic":
            # Full state vector (16-dimensional)
            self.x = cs.vertcat(self.x, self.a)
            self.x_dot = cs.vertcat(self.x_dot, self.a_dot)
            self.state_dim = 16
            # Full state noise vector (16-dimensional)
            self.w = cs.vertcat(self.w, self.w_a)
            # Update Full input state vector
            self.u = cs.vertcat()
            # Full measurement state vector
            self.y = cs.vertcat(self.p, self.r, self.a)
        elif mhe_type == "dynamic":
            # Full state vector (13-dimensional)
            self.x = cs.vertcat(self.x, self.d, self.param)
            self.x_dot = cs.vertcat(self.x_dot, self.d_dot, self.param_dot)
            if self.mhe_with_gpyTorch:
                self.state_dim = 16
            else:
                self.state_dim = 13
            # Full state noise vector (13-dimensional)
            self.w = cs.vertcat(self.w, self.w_d)

            f_thrust = self.u * self.quad.max_thrust / (self.quad.mass + self.k_m)
            self.a = cs.vertcat(0.0, 0.0, (f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3])) #a_thrust
            # Full measurement state vector
            self.y = cs.vertcat(self.p, self.r, self.d)

        if self.mhe_with_gpyTorch:
            self.mhe_gpy_ensemble = mhe_gpy_ensemble
            self.mhe_gpy_ensemble.switch_modelDict_to_Batch()
            self.mhe_gpy_ensemble.cpu()
            self.mhe_gpy_x = [6, 7, 8]
            self.B_x = np.zeros((16, 3))
            for i, idx in enumerate([7, 8, 9]):
                self.B_x[idx, i] = 1

            if self.n_param > 0:
                self.B_x = np.append(self.B_x, np.zeros((self.n_param, self.B_x.shape[1])), axis=0)
            self.q_hist = np.zeros((0, 4))
            self.mhe_model_error = None

        # Nominal model equations symbolic function (no GP)
        self.quad_xdot_nominal = self.quad_dynamics(mhe=True, mhe_type=mhe_type)
        # Build full model for MHE. Will have 13 variables. self.dyn_x contains the symbolic variable that
        # should be used to evaluate the dynamics function. It corresponds to self.x if there are no GP's, or
        # self.x_with_gp otherwise      
        acados_models_mhe, nominal_with_gp_mhe = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u, w=self.w)['x_dot'], model_name, mhe=True)
        
        # Convert dynamics variables to functions of the state and input vectors
        for dyn_model_idx in nominal_with_gp_mhe.keys():
            dyn = nominal_with_gp_mhe[dyn_model_idx]
            self.quad_xdot_mhe[dyn_model_idx] = cs.Function('x_dot', [self.x, self.u, self.w], [dyn], ['x', 'u', 'w'], ['x_dot'])        

        # ### Setup and compile Acados OCP solvers ### #
        if compile_acados:
            self.acados_mhe_solver = {}
            for key_model in acados_models_mhe.values():
                ocp_mhe, nyx, nx, nu = self.create_mhe_solver(key_model, q_mhe, q0_factor, r_mhe, solver_options)
                self.nyx = nyx
                self.nx = nx
                self.nu = nu

                # Compile acados OCP solver if necessary
                json_file_mhe = os.path.join(self.acados_models_dir, "mhe", key_model.name + '.json')
                self.acados_mhe_solver = AcadosOcpSolver(ocp_mhe, json_file=json_file_mhe)

    def create_mhe_solver(self, model, q_cost, q0_factor, r_cost, solver_options):
        """
        Creates OCP objects to formulate the MPC optimization
        :param model: Acados model of the system
        :type model: cs.MX 
        :param q_cost: diagonal of Q matrix for model mismatch cost of MHE cost function. Must be a numpy array of length 12.
        :param q0_factor: integer to set the arrival cost of MHE as a factor/multiple of q_mhe.
        :param r_cost: diagonal of R matrix for measurement mismatch cost of MHE cost function. 
        Must be a numpy array of length equal to number of measurements.
        :param solver_options: Optional set of extra options dictionary for solvers.
        """
        # Set Arrival Cost as a factor of q_cost
        q0_cost = q_cost * q0_factor
        if self.n_param > 0:
            q_cost = q_cost[:-self.n_param]

        # Number of states and Inputs of the model
        # make acceleration as error
        x = model.x
        u = model.u
        yx = self.y

        nx = x.size()[0]
        nyx = yx.size()[0]
        nu = u.size()[0]

        # Total number of elements in the cost functions
        ny_0 = nyx + nu + nx # h(x), w and arrival cost
        ny_e = 0
        ny = nyx + nu # h(x) and w
        n_param = model.p.size()[0] if isinstance(model.p, cs.MX) else 0
        
        # Set up Cost Matrices
        Q = np.diag(q_cost)
        R = np.diag(r_cost)
        Q0 = np.diag(q0_cost)

        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, '../common')

        # Create OCP object to formulate the MPC optimization
        ocp_mhe = AcadosOcp()
        ocp_mhe.acados_include_path = acados_source_path + '/include'
        ocp_mhe.acados_lib_path = acados_source_path + '/lib'
        ocp_mhe.model = model
        
        # Set Prediction horizon
        ocp_mhe.solver_options.tf = self.T_mhe
        ocp_mhe.dims.N = self.N_mhe
        
        # Cost of Initial stage    
        ocp_mhe.cost.cost_type_0 = 'NONLINEAR_LS'
        ocp_mhe.cost.W_0 = block_diag(R, Q, Q0)
        ocp_mhe.model.cost_y_expr_0 = cs.vertcat(yx, u, x)
        ocp_mhe.cost.yref_0 = np.zeros((ny_0,))

 
        # Cost of Intermediate stages
        ocp_mhe.cost.cost_type = 'NONLINEAR_LS'         
        ocp_mhe.cost.W = block_diag(R, Q)
        ocp_mhe.model.cost_y_expr = cs.vertcat(yx, u)
        ocp_mhe.cost.yref = np.zeros((ny,))
        

        # set y_ref terminal stage which doesn't exist so 0s
        ocp_mhe.cost.cost_type_e = "LINEAR_LS"
        ocp_mhe.cost.yref_e = np.zeros((ny_e, ))
        ocp_mhe.cost.W_e = np.zeros((0,0))
        
        # Initialize parameters
        ocp_mhe.parameter_values = np.zeros((n_param, ))

        # Quadrotor Mass Estimation limits
        if isinstance(self.k_m, cs.MX):
            ocp_mhe.constraints.lbx_0 = self.param_lbx
            ocp_mhe.constraints.ubx_0 = self.param_ubx
            ocp_mhe.constraints.idxbx_0 = np.array([self.state_dim])

        # Solver options
        ocp_mhe.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp_mhe.solver_options.integrator_type = 'ERK'
        ocp_mhe.solver_options.print_level = 0
        ocp_mhe.solver_options.nlp_solver_type = 'SQP_RTI' if solver_options is None else solver_options["solver_type"]

        # Path to where code will be exported
        ocp_mhe.code_export_directory = os.path.join(self.acados_models_dir, "mhe")

        return ocp_mhe, nyx, nx, nu

def main():
    rospy.init_node("acados_compiler_mhe", anonymous=True)
    ns = rospy.get_namespace()
    print(ns)

    # Load MHE parameters
    n_mhe = rospy.get_param(ns + 'n_mhe', default=50)
    t_mhe = rospy.get_param(ns + 't_mhe', default=0.5)
    mhe_type = rospy.get_param(ns + 'mhe_type', default="kinematic")
    quad_name = rospy.get_param(ns + 'quad_name', default=None)
    assert quad_name != None
    with_gp = rospy.get_param(ns + 'with_gp', default=False)
    change_mass = rospy.get_param(ns + 'change_mass', default=0)
    
    # System Noise
    w_p = np.ones((1,3)) * rospy.get_param(ns + 'w_p', default=0.004)
    w_q = np.ones((1,3)) * rospy.get_param(ns + 'w_q', default=0.01)
    w_v = np.ones((1,3)) * rospy.get_param(ns + 'w_v', default=0.005)
    w_r = np.ones((1,3)) * rospy.get_param(ns + 'w_r', default=0.5)
    w_a = np.ones((1,3)) * rospy.get_param(ns + 'w_a', default=0.05)

    w_d = np.ones((1,3)) * rospy.get_param(ns + 'w_d', default=0.00001)
    w_m = np.ones((1,3)) * rospy.get_param(ns + 'w_m', default=0.0001)

    # Measurement Noise
    v_p = np.ones((1,3)) * rospy.get_param(ns + 'v_p', default=0.002)
    v_r = np.ones((1,3)) * rospy.get_param(ns + 'v_r', default=1e-6)
    v_a = np.ones((1,3)) * rospy.get_param(ns + 'v_a', default=1e-5)
    v_d = np.ones((1,3)) * rospy.get_param(ns + 'v_d', default=0.0001)
    # System Weights
    if mhe_type == "kinematic":
        q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_a)))
    elif mhe_type == "dynamic":
        if not with_gp:
            if change_mass != 0:
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_m)))
            else:
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r)))
        else:
            if change_mass != 0:
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_d, w_m)))
            else:
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_d)))
    q0_factor = 1 # arrival cost factor
    if mhe_type == "dynamic" and with_gp:
        r_mhe = 1/np.squeeze(np.hstack((v_p, v_r, v_d))) 
    elif mhe_type == "dynamic" and not with_gp:
        r_mhe = 1/np.squeeze(np.hstack((v_p, v_r)))
    else:
        r_mhe = 1/np.squeeze(np.hstack((v_p, v_r, v_a))) 

    # Load Quad Instance
    quad = custom_quad_param_loader(quad_name)

    # Compile Acados Model
    quad_opt = QuadOptimizerMHE(quad, t_mhe=t_mhe, n_mhe=n_mhe, mhe_type=mhe_type,
                                q_mhe=q_mhe, q0_factor=q0_factor, r_mhe=r_mhe,
                                model_name=quad_name, mhe_gpy_ensemble=None,
                                change_mass=change_mass)
    return

def init_compile():
    quad_name = "clark"
    quad = custom_quad_param_loader(quad_name)
    quad_opt = QuadOptimizerMHE(quad)

if __name__ == "__main__":
    main()
    # init_compile()