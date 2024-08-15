#ifndef GP_MPC_NODE_H
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

class Node {
private:
    // System environment
    std::string environment_, quad_name_; 
    bool use_groundtruth_;

    // MPC
    GP_MPC* gp_mpc_;

    // System States
    std::vector<double> x_, u_, p_, q_, v_, w_;
    bool x_available_;
    std::thread status_thread_;

    // MPC variables
    std::vector<double> x_opt_, u_opt_;
    int control_freq_factor_;
    bool optimize_next_ = false, x_initial_reached_ = false,
         landing_ = false,  ground_level_ = true;
    std::thread mpc_thread_;
    double quad_max_thrust_, quad_mass_; // only used in Gazebo env

    // Reference Trajectory Variables
    std::vector<std::vector<double>> x_ref_, u_ref_;
    std::vector<double> t_ref_, x_ref_prov_, u_ref_prov_;
    std::string ref_traj_name_;
    double land_z_, land_dz_, land_z_thr_, init_thr_, init_v_;
    int ref_len_;

    // Recording Parameters 
    double optimization_dt_ = 0;
    int mpc_idx_ = 0, mpc_seq_num_ = 0, 
        last_state_est_seq_num_ = 0;

    // GP Parameters
    bool with_gp_;

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

#endif