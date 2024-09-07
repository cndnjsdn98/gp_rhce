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
from src.visualization.visualization import trajectory_tracking_results, state_estimation_results, test
from src.quad_opt.quad import custom_quad_param_loader

class VisualizerWrapper:
    def __init__(self):
        # Get Params
        self.quad_name = None
        while self.quad_name is None:
            self.quad_name = rospy.get_param("gp_mpc/quad_name", default=None)
            rospy.sleep(1)
            
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
        self.results_dir = rospy.get_param("results_dir", default=None)
        # if self.results_dir is None:
        #     self.results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results')
        assert self.results_dir is not None
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
        self.mpc_dataset_name = "%s_mpc_%s%s"%(self.env, "gt_" if self.use_groundtruth else "", self.quad_name)
        self.mhe_dataset_name = "%s_mhe_%s"%(self.env, self.quad_name)
        self.mpc_dir = os.path.join(self.results_dir, self.mpc_dataset_name)
        self.mhe_dir = os.path.join(self.results_dir, self.mhe_dataset_name)
        # Create Directory
        safe_mkdir_recursive(self.mpc_dir)
        safe_mkdir_recursive(self.mhe_dir)

        self.record = False

        # Initialize vectors to store Reference Trajectory
        self.seq_len = None
        self.ref_traj_name = None
        self.ref_v = None
        self.x_ref = None
        self.t_ref = None
        self.u_ref = None
        # Initialize vectors to store Tracking and Estimation Results
        self.t_act = np.zeros((0, 1))
        self.t_imu = np.zeros((0, 1))
        self.t_est = np.zeros((0, 1))
        self.t_acc_est = np.zeros((0, 1))
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

        # Subscriber topic names
        # Note: groundtruth measurements will be coming from pose+twist (arena) or odom_gz (gazebo)
        pose_topic = rospy.get_param("/gp_mhe/pose_topic", default = "/mocap/" + self.quad_name + "/pose")
        twist_topic = rospy.get_param("/gp_mhe/twist_topic", default ="/mocap/" + self.quad_name + "/twist")
        odom_topic = rospy.get_param("/gp_mhe/odom_topic", default = "/" + self.quad_name + "/ground_truth/odometry")
        imu_topic = rospy.get_param("/gp_mhe/imu_topic", default = "/mavros/imu/data_raw") 
        state_est_topic = rospy.get_param("/gp_mhe/state_est_topic", default = "/" + self.quad_name + "/state_est")
        acceleration_est_topic = rospy.get_param("/gp_mhe/acceleration_est_topic", default = "/" + self.quad_name + "/acceleration_est")
        motor_thrust_topic = rospy.get_param("/gp_mpc/motor_thrust_topic", default = "/" + self.quad_name + "/motor_thrust")
        ref_topic = rospy.get_param("/gp_mpc/ref_topic", default = "/gp_mpc/reference")
        control_topic = rospy.get_param("/gp_mpc/control_topic", default = "/mavros/setpoint_raw/attitude")
        control_gz_topic = rospy.get_param("/gp_mpc/control_gz_topic", default = "/" + self.quad_name + "/autopilot/control_command_input")
        record_topic = rospy.get_param("/gp_mpc/record_topic", default = "/" + self.quad_name + "/record")

        # Subscribers
        # self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback, queue_size=10, tcp_nodelay=True)
        # self.twist_sub = rospy.Subscriber(twist_topic, TwistStamped, self.twist_callback, queue_size=10, tcp_nodelay=True)
        self.imu_sub = rospy.Subscriber(imu_topic, Imu, self.imu_callback, queue_size=10, tcp_nodelay=True)
        self.motor_thrust_sub = rospy.Subscriber(motor_thrust_topic, Actuators, self.motor_thrust_callback, queue_size=10, tcp_nodelay=True)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=10, tcp_nodelay=True)
        self.state_est_sub = rospy.Subscriber(state_est_topic, Odometry, self.state_est_callback, queue_size=10, tcp_nodelay=True)
        self.acceleration_est_sub = rospy.Subscriber(acceleration_est_topic, Imu, self.acceleration_est_callback, queue_size=10, tcp_nodelay=True)
        self.ref_sub = rospy.Subscriber(ref_topic, ReferenceTrajectory, self.ref_callback, queue_size=10, tcp_nodelay=True)
        self.control_sub = rospy.Subscriber(control_topic, AttitudeTarget, self.control_callback, queue_size=10, tcp_nodelay=True)
        self.control_gz_sub = rospy.Subscriber(control_gz_topic, ControlCommand, self.control_gz_callback, queue_size=10, tcp_nodelay=True)
        self.record_sub = rospy.Subscriber(record_topic, Bool, self.record_callback, queue_size=10, tcp_nodelay=True)

        rospy.loginfo("Visualizer on standby listening...")

    def save_recording_data(self):
        # Remove Exceeding data entry if needed
        # Data Sampled at 100Hz
        min_len = np.min((len(self.x_act), len(self.motor_thrusts)))
        # self.x_act = self.x_act[:min_len]
        self.t_act = self.t_act - self.t_act[0]
        # self.motor_thrusts = self.motor_thrusts[:min_len]

        self.w_control = self.w_control[:self.seq_len]
        while len(self.w_control) < self.seq_len:
            self.w_control = np.append(self.w_control, self.w_control[-1][np.newaxis], axis=0)
        
        while len(self.x_act) < self.seq_len * 2:
            self.x_act  = np.append(self.x_act, self.x_act[-1][np.newaxis], axis=0)
            self.t_act  = np.append(self.t_act, self.t_act[-1])

        while len(self.motor_thrusts) < self.seq_len * 2:
            self.motor_thrusts = np.append(self.motor_thrusts, self.motor_thrusts[-1][np.newaxis], axis=0)

        if len(self.x_est) > 0:
            self.t_est = self.t_est - self.t_est[0]
            while len(self.x_est) < self.seq_len * 2:
                self.x_est = np.append(self.x_est, self.x_est[-1][np.newaxis], axis=0)
                self.t_est = np.append(self.t_est, self.t_est[-1])
            self.x_est = self.x_est[:self.seq_len * 2]
            self.t_est = self.t_est[:self.seq_len * 2]

        # Save MPC results
        state_in = np.zeros((self.seq_len, self.x_act.shape[1]))
        state_out = np.zeros((self.seq_len, self.x_act.shape[1]))
        u_in = np.zeros((self.seq_len, 4))
        mpc_error = np.zeros_like(state_in)
        x_pred_traj = np.zeros_like(state_in)
        mpc_t = np.zeros((self.seq_len, 1))
        rospy.loginfo("Filling in MPC dataset and saving...")
        for i in tqdm(range(self.seq_len)): 
            ii = i * 2
            x0 = self.x_act[ii]
            xf = self.x_act[ii+1]

            u = self.motor_thrusts[i]
            _u = np.append(u, [0]) # Considering there is no mass change

            dt = self.t_act[ii+1] - self.t_act[ii]

            # Dynamic Model Pred
            x_pred = self.quad_opt.forward_prop(x0, _u, t_horizon=dt)
            x_pred = x_pred[-1, np.newaxis, :]
            x_pred_traj[i] = x_pred

            # MPC Model error
            x_err = xf - x_pred
            mpc_error[i] = x_err / dt if dt != 0 else 0

            # Save to array for plots
            state_in[i] = self.x_act[ii] if self.use_groundtruth else self.x_est[ii]
            state_out[i] = self.x_act[ii+1] if self.use_groundtruth else self.x_est[ii+1]
            u_in[i] = u
            mpc_t[i] = self.t_act[ii]

        # Organize arrays to dictionary
        mpc_dict = {
            "t": mpc_t,
            "t_ref": self.t_ref,
            "state_in": state_in,
            "state_out": state_out,
            "x" : self.x_act if self.use_groundtruth else self.x_est,
            "x_act": self.x_act,
            "x_ref": self.x_ref,
            "error": mpc_error,
            "x_pred": x_pred_traj,
            "input_in": u_in,
            "w_control": self.w_control,
            "error_pred": self.mpc_gpy_pred,
        }
        # Save results
        with open(os.path.join(self.mpc_dir, "results.pkl"), "wb") as f:
            pickle.dump(mpc_dict, f)
        with open(os.path.join(self.mpc_dir, 'meta_data.json'), "w") as f:
            json.dump(self.mpc_meta, f, indent=4)
        trajectory_tracking_results(self.mpc_dir, self.t_ref, mpc_t, self.x_ref, state_in,
                                    self.u_ref, u_in, mpc_error, w_control=self.w_control, file_type='png')

        # Check MHE is running and if it is continue to save MHE results
        if len(self.x_est) > 0:
            # self.t_est = self.t_est - self.t_est[0]
            # while len(self.x_est) < self.seq_len * 2:
            #     self.x_est = np.append(self.x_est, self.x_est[-1][np.newaxis], axis=0)
            #     self.t_est = np.append(self.t_est, self.t_est[-1])

            self.t_acc_est = self.t_acc_est - self.t_acc_est[0]
            while len(self.accel_est) < self.seq_len * 2:
                self.accel_est = np.append(self.accel_est, self.accel_est[-1][np.newaxis], axis=0)
                self.t_acc_est = np.append(self.t_acc_est, self.t_acc_est[-1])

            self.t_imu = self.t_imu - self.t_imu[0]
            while len(self.y) < self.seq_len * 2:
                self.y = np.append(self.y, self.y[-1][np.newaxis], axis=0)
                self.t_imu = np.append(self.t_imu, self.t_imu[-1])
            self.y = self.y[:self.seq_len * 2]
            self.t_imu = self.t_imu[:self.seq_len * 2]
            
            mhe_error = np.zeros_like(self.x_est)
            a_est_b_traj = np.zeros((len(self.x_est), 3))
            rospy.loginfo("Filling in MHE dataset and saving...")
            g = cs.vertcat(0, 0, -9.81)
            for i in tqdm(range(self.seq_len*2)):
                u = self.motor_thrusts[i]
                q = self.x_est[i][3:7]
                q_inv = quaternion_inverse(q)

                a_meas = self.y[i][6:9]
                a_meas = np.stack(a_meas + v_dot_q(g, q_inv).T)

                # Model Accel Est
                a_thrust = cs.vertcat(0, 0, u[0] + u[1] + u[2] + u[3] * self.quad.max_thrust / self.quad.mass)
                
                a_est_b = v_dot_q(v_dot_q(a_thrust, q) + g, q_inv)
                a_est_b = np.squeeze(a_est_b.T)
                a_est_b_traj[i] = a_est_b

                # MHE Model Error
                a_error = np.concatenate((np.zeros((1, 7)), a_meas - a_est_b, np.zeros((1, 3))), axis=None)
                mhe_error[i] = a_error

            # Organize arrays to dictionary
            mhe_dict = {
                "t": self.t_imu,
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
            state_estimation_results(self.mhe_dir, self.t_act, self.x_act, self.t_est, self.x_est, self.t_imu, self.y,
                                     mhe_error, self.t_acc_est, self.accel_est, file_type='png')

        # --- Reset all vectors ---
        # Vectors to store Reference Trajectory
        self.seq_len = None
        self.ref_traj_name = None
        self.ref_v = None
        self.x_ref = None
        self.t_ref = None
        self.u_ref = None
        # Vectors to store Tracking and Estimation Results
        self.t_act = np.zeros((0, 1))
        self.t_imu = np.zeros((0, 1))
        self.t_est = np.zeros((0, 1))
        self.t_acc_est = np.zeros((0, 1))
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
        rospy.loginfo("Recording Complete.")
        
    # def pose_callback(self, msg):
    #     if not self.record:
    #         return
        
    #     self.p_meas = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    #     self.p_act = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    #     self.q_act = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]

    #     if self.v_act is not None and self.w_act is not None:
    #         x = self.p_act + self.q_act + self.v_act + self.w_act
    #         self.x_act = np.append(self.x_act, np.array(x)[np.newaxis, :], axis=0)
    #     self.t_act = np.append(self.t_act, msg.header.stamp.to_time())

    # def twist_callback(self, msg):
    #     if not self.record:
    #         return
        
    #     self.v_act = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
    #     self.w_act = [msg.twist.angular.x*100, msg.twist.angular.y*100, msg.twist.angular.z*100]
    
    def imu_callback(self, msg):
        if not self.record:
            return
        
        self.w_meas = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.a_meas = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        if self.p_meas is not None:
            y = self.p_meas + self.w_meas + self.a_meas
            self.y = np.append(self.y, np.array(y)[np.newaxis, :], axis=0)
            self.t_imu = np.append(self.t_imu, msg.header.stamp.to_time())

    def motor_thrust_callback(self, msg):
        if not self.record:
            return
        
        motor_thrusts = tuple(msg.angular_velocities)
        self.motor_thrusts = np.append(self.motor_thrusts, [motor_thrusts], axis=0)
        self.motor_thrusts = np.append(self.motor_thrusts, [motor_thrusts], axis=0)
    
    def odom_callback(self, msg):
        if not self.record:
            return
        
        p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        q = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z]
        v = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        self.p_meas = p

        if self.env == "gazebo":
            v = v_dot_q(np.array(v), np.array(q)).tolist()

        x = p + q + v + w
        
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
        self.t_est = np.append(self.t_est, msg.header.stamp.to_time())

    def acceleration_est_callback(self, msg):
        if not self.record:
            return
        
        a_est = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

        self.accel_est = np.append(self.accel_est, np.array(a_est)[np.newaxis, :], axis=0)
        self.t_acc_est = np.append(self.t_acc_est, msg.header.stamp.to_time())

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
        # self.w_control = np.append(self.w_control, np.array(w_control)[np.newaxis, :], axis=0)
        # self.collective_thrusts = np.append(self.collective_thrusts, collective_thrust)

    def control_gz_callback(self, msg):
        if not self.record:
            return
        
        w_control = [msg.bodyrates.x, msg.bodyrates.y, msg.bodyrates.z]
        # collective_thrust = msg.thrusts

        self.w_control = np.append(self.w_control, np.array(w_control)[np.newaxis, :], axis=0)
        # self.w_control = np.append(self.w_control, np.array(w_control)[np.newaxis, :], axis=0)
        # self.collective_thrusts = np.append(self.collective_thrusts, collective_thrust)

    def record_callback(self, msg):  
        if not self.record and msg.data == True:
            # Recording has begun
            rospy.loginfo("Recording...")     

        if self.record and msg.data == False:
            # Recording ended
            self.record = msg.data
            # Run thread for saving the recorded data
            _save_record_thread = threading.Thread(target=self.save_recording_data(), args=(), daemon=True)
            _save_record_thread.start()

        self.record = msg.data

def main():
    rospy.init_node("visualizer")
    visualizer = VisualizerWrapper()

    rospy.spin()

if __name__ == "__main__":
    main()
