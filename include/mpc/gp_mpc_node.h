#ifndef _GP_MPC_NODE_H
#define _GP_MPC_NODE_H

#include <ros/ros.h>

#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/CommandBool.h>
#include <std_msgs/Bool.h>
#include <nav_msgs/Odometry.h>
#include <mav_msgs/Actuators.h>
#include <gp_rhce/ReferenceTrajectory.h>
#include <quadrotor_msgs/ControlCommand.h>
#include <thread>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <gp_mpc.h>

#include <torch/script.h>

#include <iostream>
#include <filesystem>

class Node {
private:
    // System environment
    std::string environment_, quad_name_; 
    bool use_groundtruth_;

    // MPC
    GP_MPC* gp_mpc_;

    // System States
    Eigen::VectorXd x_, u_, p_, v_, w_; 
    Eigen::Quaterniond q_;
    bool x_available_;
    std::thread status_thread_;

    // MPC variables
    Eigen::VectorXd x_opt_, u_opt_;
    int control_freq_factor_;
    bool optimize_next_ = false, x_initial_reached_ = false,
         landing_ = false,  ground_level_ = true;
    std::thread mpc_thread_;
    double quad_max_thrust_, quad_mass_; // only used in Gazebo env

    // Reference Trajectory Variables
    Eigen::MatrixXd x_ref_, u_ref_;
    Eigen::VectorXd t_ref_, x_ref_prov_, u_ref_prov_;
    std::string ref_traj_name_;
    double land_z_, land_dz_, land_thr_, init_thr_, init_v_;
    int ref_len_;
    bool ref_received_;
    
    // Recording Parameters 
    double optimization_dt_ = 0;
    int mpc_idx_ = 0, mpc_seq_num_ = 0, 
        last_state_est_seq_num_ = 0;

    // GP Parameters
    bool with_gp_;
    int n_features_;
    std::string gp_model_dir_, gp_model_name_;
    std::vector<int> x_features_, y_features_;
    std::vector<torch::jit::IValue> gp_input_;
    std::vector<torch::jit::script::Module> gp_model_;
    Eigen::MatrixXd gp_corr_ref_, gp_corr_, mpc_param_;
    Eigen::Vector3d vb_;
    Eigen::Quaterniond q_inv_;

    // Subscriber Topics 
    std::string ref_topic_, state_est_topic_, odom_topic_, land_topic_;

    // Publisher Topics
    std::string control_topic_, motor_thrust_topic_, record_topic_, status_topic_;
    // Gazebo Specific Publisher Topic
    std::string control_gz_topic_;

    // ROS Service Topics
    std::string mavros_set_mode_srvc_, mavros_arming_srvc_;

    // Subscribers
    ros::Subscriber ref_sub_, state_est_sub_, land_sub_;

    // Publishers
    ros::Publisher control_pub_, motor_thrust_pub_, record_pub_, status_pub_;
    // Gazebo Specific Publisher
    ros::Publisher control_gz_pub_;

    // ROS Services
    ros::ServiceClient set_mode_client_, arming_client_;

    // Initialize methods
    void initLaunchParameters(ros::NodeHandle& nh);
    void initSubscribers(ros::NodeHandle& nh);
    void initPublishers(ros::NodeHandle& nh);
    void initRosService(ros::NodeHandle& nh);

    // Callback methods
    void referenceCallback(const gp_rhce::ReferenceTrajectory::ConstPtr& msg);
    void stateEstCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void landCallback(const std_msgs::Bool::ConstPtr& msg);
    
public:
    // Constructor
    Node(ros::NodeHandle& nh);

    // Destructor
    ~Node();

    // Helper methods
    void runMPC();
    void setReferenceTrajectory();

    void run();
};

// Utils
inline Eigen::Vector3d transformToBodyFrame(const Eigen::Quaterniond& q, const Eigen::Vector3d& v) {
    return q.inverse() * v;
}
inline Eigen::Vector3d transformToWorldFrame(const Eigen::Quaterniond& q, const Eigen::Vector3d& v) {
    return q * v;
}

inline Eigen::MatrixXd tensorToEigen(const std::vector<at::Tensor>& tensor) {
    // Check that vector is not empty and get the dimensions of the first tensor
    if (tensor.empty()) {
        throw std::runtime_error("tensor is empty.");
    }

    // Assuming each tensor in tensor is a 1D tensor with the same size
    int rows = tensor.size();               // Number of rows = number of tensors
    int cols = tensor[0].size(0);           // Number of columns = size of each tensor

    // Initialize an Eigen::MatrixXd of appropriate size
    Eigen::MatrixXd eigen_matrix(rows, cols);

    // Copy data from each tensor into the Eigen matrix
    for (int i = 0; i < rows; ++i) {
        // Ensure that the tensor is 1D and has the correct size
        if (tensor[i].dim() != 1 || tensor[i].size(0) != cols) {
            throw std::runtime_error("Tensor dimensions do not match expected size.");
        }

        // Copy the tensor data into the corresponding row of the Eigen matrix
        eigen_matrix.row(i) = Eigen::Map<Eigen::VectorXd>(tensor[i].data_ptr<double>(), cols);
    }

    return eigen_matrix;
}

#endif  // _GP_MPC_NODE_H