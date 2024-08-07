#!/usr/bin/env python

import os
import rospy
import threading
import numpy as np
from tqdm import tqdm
import casadi as cs
import pickle 
import json

from std_msgs.msg import Bool
from mav_msgs.msg import Actuators
from mavros_msgs.msg import AttitudeTarget
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TwistStamped
from quadrotor_msgs.msg import ControlCommand
from gp_rhce.msg import ReferenceTrajectory

from src.quad_opt.quad_optimizer import QuadOptimizer
from src.utils.utils import v_dot_q, quaternion_inverse, safe_mkdir_recursive
from src.visualization.visualization import trajectory_tracking_results, state_estimation_results
from src.utils.quad import custom_quad_param_loader

class VisualizerWrapper:
    def init(self):
        # Get Params
        self.quad_name = rospy.get_param("gp_mpc/quad_name", default="clark")
        self.env = rospy.get_param("gp_mpc/environment", default="arena")
        
        self.quad = custom_quad_param_loader(self.quad_name)
        self.quad_opt = QuadOptimizer(self.quad)

        self.t_mpc = rospy.get_param("gp_mpc/t_mpc", default=1)
        self.n_mpc = rospy.get_param("gp_mpc/n_mpc", default=10)
        self.t_mhe = rospy.get_param("gp_mhe/t_mhe", default=0.5)
        self.n_mhe = rospy.get_param("gp_mhe/n_mhe", default=50)
        self.use_groundtruth = rospy.get_param("gp_mpc/use_groundtruth", default=True)
        self.mpc_with_gp = rospy.get_param("gp_mpc/with_gp", default=False)
        self.mhe_with_gp = rospy.get_param("gp_mhe/with_gp", default=False)

        ns = rospy.get_namespace()
        self.results_dir = rospy.get_param(ns + "results_dir", default="./../../results/")
        self.mpc_meta = {
            'quad_name': self.quad_name,
            'env': self.env,
            't_mpc': self.t_mpc,
            'n_mpc': self.n_mpc,
            'gt': self.use_groundtruth,
            'with_gp': self.mpc_with_gp,
        }
        self.mhe_meta = {
            'quad_name': self.quad_name,
            'env': self.env,
            't_mhe': self.t_mhe,
            'n_mhe': self.n_mhe,
            'with_gp': self.mhe_with_gp,
        }
        self.mpc_dataset_name = "%s_mpc_%s%s"%(self.env, "gt_" if self.use_groundturth else "", self.quad_name)
        self.mhe_dataset_name = "%s_mhe_%s"%(self.env, self.quad_name)
        self.mpc_dir = os.path.join(self.results_dir, self.mpc_dataset_name)
        self.mhe_dir = os.path.join(self.results_dir, self.mhe_dataset_name)
        # Create Directory
        safe_mkdir_recursive(self.mpc_dir)
        safe_mkdir_recursive(self.mhe_dir)

        self.record = False

        # Subscriber topic names
        # Note: groundtruth measurements will be coming from pose+twist (arena) or odom_gz (gazebo)
        pose_topic = rospy.get_param("/gp_mhe/pose_topic", default = "/mocap/" + self.quad_name + "/pose")
        twist_topic = rospy.get_param("/gp_mhe/twist_topic", default ="/mocap/" + self.quad_name + "/twist")
        odom_gz_topic = rospy.get_param("/gp_mhe/odom_gz_topic", default = "/" + self.quad_name + "/ground_truth/odometry")
        imu_topic = rospy.get_param("/gp_mhe/imu_topic", default = "/mavros/imu/data_raw") 
        motor_thrust_topic = rospy.get_param("/gp_mhe/imu_topic", default = "/" + self.quad_name + "/motor_thrust")
        state_est_topic = rospy.get_param("/gp_mhe/state_est_topic", default = "/" + self.quad_name + "/state_est")
        acceleration_est_topic = rospy.get_param("/gp_mhe/acceleration_est_topic", default = "/" + self.quad_name + "/acceleration_est")
        ref_topic = rospy.get_param("/gp_mpc/ref_topic", default = "/gp_mpc/reference")
        control_topic = rospy.get_param("/gp_mpc/control_topic", default = "/mavros/setpoint_raw/attitude")
        control_gz_topic = rospy.get_param("/gp_mpc/control_gz_topic", default = "/" + self.quad_name + "autopilot/control_command_input")
        record_topic = rospy.get_param("/gp_mpc/record_topic", default = "/" + self.quad_name + "/record")

        # Subscribers
        self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback, queue_size=10)
        self.twist_sub = rospy.Subscriber(twist_topic, TwistStamped, self.twist_callback, queue_size=10)
        self.imu_sub = rospy.Subscriber(imu_topic, Imu, self.imu_callback, queue_size=10)
        self.motor_thrust_sub = rospy.Subscriber(motor_thrust_topic, Actuators, self.motor_thrust_callback, queue_size=10)
        self.odom_gz_sub = rospy.Subscriber(odom_gz_topic, Odometry, self.odom_gz_callback, queue_size=10)
        self.state_est_sub = rospy.Subscriber(state_est_topic, Odometry, self.state_est_callback, queue_size=10)
        self.acceleration_est_sub = rospy.Subscriber(acceleration_est_topic, Imu, self.acceleration_est_callback, queue_size=10)
        self.ref_sub = rospy.Subscriber(ref_topic, ReferenceTrajectory, self.ref_callback, queue_size=10)
        self.control_sub = rospy.Subscriber(control_topic, AttitudeTarget, self.control_callback, queue_size=10)
        self.control_gz_sub = rospy.Subscriber(control_gz_topic, ControlCommand, self.control_gz_callback, queue_size=10)
        self.record_sub = rospy.Subscriber(record_topic, Bool, self.record_callback, queue_size=10)

        # Initialize vectors to store Reference Trajectory
        self.seq_len = None
        self.ref_traj_name = None
        self.ref_v = None
        self.x_ref = None
        self.t_ref = None
        self.u_ref = None
        # Initialize vectors to store Tracking and Estimation Results
        self.t_act = np.zeros((0, 1))
        self.x_act = np.zeros((0, 13))
        self.x_est = np.zeros((0, 13))
        self.y = np.zeros((0, 9))
        self.accel_est = np.zeros((0, 3))
        self.motor_thrusts = np.zeros((0, 4))
        self.w_control = np.zeros((0, 3))
        self.collective_thrusts = np.zeros((0, 1))

        self.mpc_gpy_pred = np.zeros((0, 3))
        self.mhe_gpy_pred = np.zeros((0, 3))

        # System States
        self.p_act = None
        self.q_act = None
        self.v_act = None
        self.w_act = None
        self.p_meas = None
        self.w_meas = None
        self.a_meas = None

    def save_recording_data(self):
        # Remove Exceeding data entry if needed
        if len(self.x_act) > len(self.motor_thrusts):
            self.x_act = self.x_act[:-1]
        
        # Save MPC results
        mpc_error = np.zeros_like(self.x_act)
        x_pred_traj = np.zeros_like(self.x_act)
        rospy.loginfo("Filling in MPC dataset and saving...")
        for i in tqdm(range(0, len(self.x_act), 2)):
            x0 = self.x_act[i]
            xf = self.x_act[i+1]
            u = self.motor_thrusts[i]
            dt = self.t_act[i+1] - self.t_act[i]
            # Dynamic Model Pred
            x_pred = self.quad_opt.forward_prop(x0, u, t_horizon=dt)
            x_pred = x_pred[-1, np.newaxis, :]
            x_pred_traj[i] = x0
            x_pred_traj[i+1] = x_pred

            # MPC Model error
            x_err = xf - x_pred
            mpc_error[i] = x_err
            mpc_error[i+1] = x_err

        # Organize arrays to dictionary
        mpc_dict = {
            "t": self.t_act,
            "x" : self.x_act if self.use_groundtruth else self.x_est,
            "x_act": self.x_act,
            "error": mpc_error,
            "x_pred": x_pred_traj,
            "input_in": self.motor_thrusts,
            "w_control": self.w_control,
            "error_pred": self.mpc_gpy_pred,
        }
        # Save results
        with open(os.path.join(self.mpc_dir, "results.pkl"), "wb") as f:
            pickle.dump(mpc_dict, f)
        with open(os.path.join(self.mpc_dir, 'meta_data.json'), "w") as f:
            json.dump(self.mpc_meta, f, indent=4)
        trajectory_tracking_results(self.mpc_dir, self.t_act, self.x_ref, self.x_act if self.use_groundtruth else self.x_est,
                                    self.u_ref, self.motor_thrusts,  w_control=self.w_control, file_type='png')

        # Check MHE is running and if it is continue to save MHE results
        if self.x_est is not None:
            mhe_error = np.zeros_like(self.x_est)
            a_est_b_traj = np.zeros((3, len(self.x_est)))
            rospy.loginfo("Filling in MHE dataset and saving...")
            for i in tqdm(range(len(self.x_est))):
                u = self.motor_thrusts[i]
                a_meas = self.y[i][6:9]
                a_meas = np.stack(a_meas + v_dot_q(g, q_inv).T)
                # Model Accel Est
                a_thrust = cs.vertcat(0, 0, u[0] + u[1] + u[2] + u[3] * self.quad.max_thrust / self.quad.mass)
                q = self.x_est[i][3:7]
                q_inv = quaternion_inverse(q)
                g = cs.vertcat(0, 0, -9.81)
                a_est_b = v_dot_q(v_dot_q(a_thrust, q) + g, q_inv)
                a_est_b = np.squeeze(a_est_b.T)
                a_est_b_traj[i] = a_est_b

                # MHE Model Error
                a_error = np.concatenate((np.zeros((1, 7)), a_meas - a_est_b, np.zeros((1, 3))), axis=None)
                mhe_error[i] = a_error

            # Organize arrays to dictionary
            mhe_dict = {
                "t": self.t_act,
                "x_est": self.x_est,
                "x_act": self.x_act,
                "sensor_meas": self.y,
                "motor_thrusts": self.motor_thrusts,
                "error_pred": self.mhe_gpy_pred,
                "error": mhe_error,
                "a_est_b": a_est_b_traj,
                "accel_est": self.accel_est,
            } 
            # Save results
            with open(os.path.join(self.mhe_dir, "results.pkl"), "wb") as f:
                pickle.dump(mhe_dict, f)
            with open(os.path.join(self.mhe_dir, 'meta_data.json'), "w") as f:
                json.dump(self.mhe_meta, f, indent=4)
            state_estimation_results(self.mhe_dir, self.t_act, self.x_act, self.x_est, self.y,
                                     mhe_error, self.accel_est, file_type='png')
        # --- Reset all vectors ---
        # Vectors to store Reference Trajectory
        self.seq_len = None
        self.ref_traj_name = None
        self.ref_v = None
        self.x_ref = None
        self.t_ref = None
        self.u_ref = None
        # Vectors to store Tracking and Estimation Results
        self.x_act = np.zeros((0, 13))
        self.x_est = np.zeros((0, 13))
        self.y = np.zeros((0, 9))
        self.accel_est = np.zeros((0, 3))
        self.motor_thrusts = np.zeros((0, 4))
        self.w_control = np.zeros((0, 3))
        self.collective_thrusts = np.zeros((0, 1))
        self.mhe_gpy_pred = np.zeros((0, 3))
        self.mpc_gpy_pred = np.zeros((0, 3))
        # System States
        self.p_act = None
        self.q_act = None
        self.v_act = None
        self.w_act = None
        self.p_meas = None
        self.w_meas = None
        self.a_meas = None

    def pose_callback(self, msg):
        if not self.record:
            return
        
        self.p_act = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.p_meas = self.p_act
        self.q_act = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]

        if self.v_act is not None and self.w_act is not None:
            x = self.p_act + self.q_act + self.v_act + self.w_act
            self.x_act = np.append(self.x_act, np.array(x)[np.newaxis, :], axis=0)
        self.t_act = np.append(self.t_act, msg.header.stamp.to_time())

    def twist_callback(self, msg):
        if not self.record:
            return
        
        self.v_act = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        self.w_act = [msg.twist.angular.x*100, msg.twist.angular.y*100, msg.twist.angular.z*100]
    
    def imu_callback(self, msg):
        if not self.record:
            return
        
        self.w_meas = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.a_meas = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        if self.p_meas is not None:
            y = self.p_meas + self.w_meas + self.a_meas
            self.y = np.append(self.y, np.array(y)[np.newaxis, :], axis=0)

    def motor_thrust_callback(self, msg):
        if not self.record:
            return
        
        motor_thrusts = tuple(msg.angular_velocities)
        self.motor_thrusts = np.append(self.motor_thrusts, [motor_thrusts], axis=0)
        self.motor_thrusts = np.append(self.motor_thrusts, [motor_thrusts], axis=0)
    
    def odom_gz_callback(self, msg):
        if not self.record:
            return
        
        p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        q = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z]
        v_b = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]

        v_w = v_dot_q(np.array(v_b), np.array(q)).tolist()

        x = p + q + v_w + w
        
        self.x_act = np.append(self.x_act, np.array(x)[np.newaxis, :], axis=0)

        self.t_act = np.append(self.t_act, msg.header.stamp.to_time())

    def state_est_callback(self, msg):
        if not self.record:
            return
        
        p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        q = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z]
        v_w = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]

        x = p + q + v_w + w

        self.x_est = np.append(self.x_est, np.array(x)[np.newaxis, :], axis=0)

    def acceleration_est_callback(self, msg):
        if not self.record:
            return
        
        a_est = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

        self.accel_est = np.append(self.accel_est, np.array(a_est)[np.newaxis, :], axis=0)

    def ref_callback(self, msg):
        if self.x_ref is not None:
            return
        
        self.seq_len = msg.seq_len
        if self.seq_len == 0:
            return
        
        self.ref_traj_name = msg.traj_name
        self.ref_v = msg.v_input

        # Save reference trajectory, relative times and inputs
        self.x_ref = np.array(msg.trajectory).reshape(self.seq_len, -1)
        self.t_ref = np.array(msg.dt)
        self.u_ref = np.array(msg.inputs).reshape(self.seq_len, -1)

    def control_callback(self, msg):
        if not self.record:
            return
        
        w_control = [msg.body_rate.x, msg.body_rate.y, msg.body_rate.z]
        # collective_thrust = msg.thrust
        
        self.w_control = np.append(self.w_control, np.array(w_control)[np.newaxis, :], axis=0)
        self.w_control = np.append(self.w_control, np.array(w_control)[np.newaxis, :], axis=0)
        # self.collective_thrusts = np.append(self.collective_thrusts, collective_thrust)

    def control_gz_callback(self, msg):
        if not self.record:
            return
        
        w_control = [msg.bodyrates.x, msg.bodyrates.y, msg.bodyrates.z]
        # collective_thrust = msg.thrusts

        self.w_control = np.append(self.w_control, np.array(w_control)[np.newaxis, :], axis=0)
        self.w_control = np.append(self.w_control, np.array(w_control)[np.newaxis, :], axis=0)
        # self.collective_thrusts = np.append(self.collective_thrusts, collective_thrust)

    def record_callback(self, msg):
        if not self.record:
            return
        
        if self.record and msg.data == False:
            self.record = msg.data
            # Run thread for saving the recorded data
            _save_record_thread = threading.Thread(target=self.save_recording_data(), args=(), daemon=True)
            _save_record_thread.start()

        self.record = msg.data

def main():
    rospy.init_node("visualizer_node")
    VisualizerWrapper()

    rospy.spin()

if __name__ == "__main__":
    main()
