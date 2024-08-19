#ifndef GP_MHE_NODE_H
#define _GP_MHE_NODE_H

#include <ros/ros.h>

#include <mav_msgs/Actuators.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <mutex>
#include <thread>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <gp_mhe.h>

class Node {
private:
    // System environment
    std::string environment_, quad_name_; 
    
    // MHE
    GP_MHE* gp_mhe_;

    // MHE Parameters
    int last_imu_seq_number_ = 0;
    std::string mhe_type_;
    std::thread mhe_thread_;

    // GP Parameters
    bool with_gp_;

    // Subscriber Topics 
    std::string imu_topic_, pose_topic_, twist_topic_, motor_thrust_topic_,
                mass_change_topic_, record_topic_;
    // Gazebo Specific Subscriber Topics
    std::string odom_gz_topic_;

    // Publisher Topics
    std::string state_est_topic_, acceleration_est_topic_, status_topic_;

    // Publishers
    ros::Publisher state_est_pub_, acceleration_est_pub_, status_pub_;

    // Subscribers
    ros::Subscriber imu_sub_, record_sub_, motor_thrust_sub_,
        pose_sub_;
    // Gazebo specific Subscribers
    ros::Subscriber odom_gz_sub_;

    // System States
    std::vector<double> x_est_, p_, w_, a_, y_, u_, acceleration_est_;
    std::vector<std::vector<double>> u_hist_, y_hist_, u_hist_cp_, y_hist_cp_;
    bool hist_received_ = false;

    // Recording Parameters 
    double optimization_dt_ = 0;
    int mhe_idx_ = 0, mhe_seq_num_ = 0, last_u_count_=0;
    bool record_ = false;

    // Initialize methods
    void initLaunchParameters(ros::NodeHandle& nh);
    void initSubscribers(ros::NodeHandle& nh);
    void initPublishers(ros::NodeHandle& nh);

    // Callback methods
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
    void recordMheCallback(const std_msgs::Bool::ConstPtr& msg);
    void motorThrustCallback(const mav_msgs::Actuators::ConstPtr& msg);
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
    // Gazebo specific callback methods
    void odomGzCallback(const nav_msgs::Odometry::ConstPtr& msg);
    
    // Mutex
    std::mutex lock_;
public:
    // Constructor
    Node(ros::NodeHandle& nh);

    // Destructor
    ~Node();

    // Helper methods
    void runMHE();
};

#endif