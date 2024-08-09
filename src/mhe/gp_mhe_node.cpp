/**
 * @file gp_mhe_node.cpp
 * @author Wnooo Choo
 * @date July 2024
 * 
 * @copyright
 * Copyright (C) 2024.
 */

#include <gp_mhe_node.h>

using namespace ros;
using namespace Eigen;

// Constructor
Node::Node(ros::NodeHandle& nh) {
    initLaunchParameters(nh);
    initSubscribers(nh);
    initPublishers(nh);
    gp_mhe_ = new GP_MHE(mhe_type_);
    if (gp_mhe_ == nullptr) {
        ROS_ERROR("FAILED TO CREATE GP_MHE INSTANCE");
        std::exit(EXIT_FAILURE);
    } 
    assert(quad_name_ != "");
    ROS_INFO("MHE: %s Loaded in %s", quad_name_.c_str(), environment_.c_str());

    // Init x_est_
    if (mhe_type_ == "kinematic") {
        x_est_ = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.81};
    } else if (mhe_type_ == "dynamic") {
        x_est_ = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    }
}


/// Destructor
Node::~Node() {
    ROS_INFO("MHE Destructor is called!");
    if (mhe_thread_.joinable()) {
        mhe_thread_.join();
    }
    delete gp_mhe_;
}


void Node::initLaunchParameters(ros::NodeHandle& nh) {
    std::string ns = ros::this_node::getNamespace();

    // System environment
    nh.param<std::string> (ns + "/environment", environment_, "arena");
    nh.param<std::string>(ns + "/quad_name", quad_name_, "");

    // MHE Parameters
    nh.param<std::string>(ns + "/mhe_type", mhe_type_, "kinematic");

    // Subscriber topic names
    nh.param<std::string>(ns + "/imu_topic", imu_topic_, "/mavros/imu/data_raw");
    nh.param<std::string>(ns + "/pose_topic", pose_topic_, "/mavros/vision_pose/pose");
    nh.param<std::string>(ns + "/twist_topic", twist_topic_, "/mocap/" + quad_name_ + "/twist");
    nh.param<std::string>(ns + "/motor_thrust_topic", motor_thrust_topic_, "/" + quad_name_ + "/motor_thrust");
    nh.param<std::string>(ns + "/record_topic", record_topic_, "/" + quad_name_ + "/record");
    // Gazebo Specific Subscriber topic names
    nh.param<std::string>(ns + "/odom_gz_topic", odom_gz_topic_, "/" + quad_name_ + "/ground_truth/odometry");

    // Publisher topic names
    nh.param<std::string>(ns + "/state_est_topic", state_est_topic_, "/" + quad_name_ + "/state_est");
    nh.param<std::string>(ns + "/acceleration_est_topic", acceleration_est_topic_, "/" + quad_name_ + "/acceleration_est");
}

void Node::initSubscribers(ros::NodeHandle& nh) {
    // Gazebo specific Subscribers
    odom_gz_sub_ = nh.subscribe<nav_msgs::Odometry> (
        odom_gz_topic_, 10, &Node::odomGzCallback, this);
    // Init Subscribers
    motor_thrust_sub_ = nh.subscribe<mav_msgs::Actuators> (
        motor_thrust_topic_, 10, &Node::motorThrustCallback, this);
    pose_sub_ = nh.subscribe<geometry_msgs::PoseStamped> (
        pose_topic_, 10, &Node::poseCallback, this);
    twist_sub_ = nh.subscribe<geometry_msgs::TwistStamped> (
        twist_topic_, 10, &Node::twistCallback, this);    
    record_sub_ = nh.subscribe<std_msgs::Bool> (
        record_topic_, 10, &Node::recordMheCallback, this);
    imu_sub_ = nh.subscribe<sensor_msgs::Imu> (
            imu_topic_, 10, &Node::imuCallback, this);
    
}

void Node::initPublishers(ros::NodeHandle& nh) {
    acceleration_est_pub_ = nh.advertise<sensor_msgs::Imu> (acceleration_est_topic_, 10, true);
    state_est_pub_ = nh.advertise<nav_msgs::Odometry> (state_est_topic_, 10, true);
}


void Node::imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
    std::vector<double> p, w, a;
    w = {msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z};
    a = {msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z};
    
    if (position_.empty()) {
        ROS_WARN("Position measurement not received yet. Skipping time step...");
        return;
    } else {
        p = position_;
    }

    y_.clear();
    // Concatenate p, w, and a into y
    y_.insert(y_.end(), p.begin(), p.end());
    y_.insert(y_.end(), w.begin(), w.end());
    y_.insert(y_.end(), a.begin(), a.end());

    // Check if there are any skipped messages
    int skipped_messages = 0;
    if (last_imu_seq_number_ > 0 && record_) {
        skipped_messages = int(msg->header.seq - last_imu_seq_number_ - 1);
        if (skipped_messages > 0) {
            ROS_WARN("MHE Recording time skipped messages: %d", skipped_messages);
        }
    }

    // Save history of control actions for MHE
    int extra_u_len = 0;
    lock_.lock();
    if (last_u_count_ == 2) {
        extra_u_len = 0;
    } else if (last_u_count_ == 0) {
        extra_u_len = 2; 
    } else if (last_u_count_ == 1) {
        extra_u_len = 1;
    }
    last_u_count_++; 

    // Add IMU measurements
    if (y_hist_.empty()) {
        y_hist_.resize(MHE_N, std::vector<double>(y_.size()));
        for (auto& row: y_hist_) {
            std::copy(y_.begin(), y_.end(), row.begin());
        }
    }
    // Add current measurement to array and also add the number of missed measurements to be up to sync
    for (int i = 0; i <= skipped_messages; i++) {
        y_hist_.push_back(y_);
    }

    // Correct y_hist length
    if (y_hist_.size() >= MHE_N) {
        y_hist_.erase(y_hist_.begin(), y_hist_.end() - MHE_N);
    }
    // Correct u_hist length
    if (u_hist_.size() > MHE_N + extra_u_len) {
        u_hist_.erase(u_hist_.begin(), u_hist_.end() - (MHE_N + extra_u_len));
    }
    lock_.unlock();

    // Run MHE
    if (mhe_thread_.joinable()) {
        mhe_thread_.join();
    }
    mhe_thread_ = std::thread(&Node::runMHE, this);
}

void Node::recordMheCallback(const std_msgs::Bool::ConstPtr& msg) {
    record_ = msg->data;
}

void Node::motorThrustCallback(const mav_msgs::Actuators::ConstPtr& msg) {
    std::vector<double> thrusts;
    for (auto &data : msg->angular_velocities) {
        thrusts.push_back(data);
    }
    // Need to add it twice every time because MPC runs 50Hz while this runs at 100Hz
    u_hist_.push_back(thrusts);
    u_hist_.push_back(thrusts);
}

void Node::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    position_ = {msg->pose.position.x, 
                 msg->pose.position.y, 
                 msg->pose.position.z};
    quaternion_ = {msg->pose.orientation.w, msg->pose.orientation.x,  
                   msg->pose.orientation.y, msg->pose.orientation.z};
}

void Node::twistCallback(const geometry_msgs::TwistStamped::ConstPtr& msg) {
    velocity_ = {msg->twist.linear.x,
                 msg->twist.linear.y,
                 msg->twist.linear.z};
    angular_velocity_ = {msg->twist.angular.x,
                         msg->twist.angular.y,
                         msg->twist.angular.z};
}

void Node::odomGzCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    position_ = {msg->pose.pose.position.x,
                 msg->pose.pose.position.y,
                 msg->pose.pose.position.z}; 
    // Load as Eigen to perform Rotation
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, 
                         msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y, 
                         msg->pose.pose.orientation.z);
    Eigen::Vector3d v(msg->twist.twist.linear.x, 
                     msg->twist.twist.linear.y, 
                     msg->twist.twist.linear.z);
    // Apply rotation to get velocity in world frame
    v = q * v;
    // Save to std::vector
    quaternion_ = {q.w(), q.x(), q.y(), q.z()};
    velocity_ = {v.x(), v.y(), v.z()};
    
    angular_velocity_ = {msg->twist.twist.angular.x,
                         msg->twist.twist.angular.y,
                         msg->twist.twist.angular.z};
    
}

void Node::runMHE() {
    try {
        // optimize MHE
        if (gp_mhe_->solveMHE(y_hist_, u_hist_) == ACADOS_SUCCESS) {
            gp_mhe_->getStateEst(x_est_);
            optimization_dt_ += gp_mhe_->getOptimizationTime();
            mhe_idx_++;
        } else {
            ROS_WARN("Tried to run an MHE optimization but was unsuccessful.");
        }  
    } catch (const std::runtime_error& e) {
        ROS_WARN("Tried to run an MHE optimization but MHE is not ready yet.");
        return;
    }  

    if (!x_est_.empty()) {
        if (mhe_type_ == "kinematic") {
            std::vector<double> acceleration_est(3, 0.0);
            std::copy(x_est_.begin() + 13, x_est_.end(), acceleration_est.begin());
            sensor_msgs::Imu acceleration_est_msg;
            acceleration_est_msg.header.stamp = ros::Time::now();
            acceleration_est_msg.header.seq = mhe_seq_num_;
            acceleration_est_msg.linear_acceleration.x = acceleration_est[0];
            acceleration_est_msg.linear_acceleration.y = acceleration_est[1];
            acceleration_est_msg.linear_acceleration.z = acceleration_est[2];
            acceleration_est_pub_.publish(acceleration_est_msg);
        }

        // Update the state estimate of the quad
        nav_msgs::Odometry state_est_msg;
        state_est_msg.header.stamp = ros::Time::now();
        state_est_msg.header.seq = mhe_seq_num_++;
        state_est_msg.header.frame_id = "world";
        state_est_msg.child_frame_id = quad_name_ + "/base_link";

        state_est_msg.pose.pose.position.x = x_est_[0];
        state_est_msg.pose.pose.position.y = x_est_[1];
        state_est_msg.pose.pose.position.z = x_est_[2];

        state_est_msg.pose.pose.orientation.w = x_est_[3];
        state_est_msg.pose.pose.orientation.x = x_est_[4];
        state_est_msg.pose.pose.orientation.y = x_est_[5];
        state_est_msg.pose.pose.orientation.z = x_est_[6];

        state_est_msg.twist.twist.linear.x = x_est_[7];
        state_est_msg.twist.twist.linear.y = x_est_[8];
        state_est_msg.twist.twist.linear.z = x_est_[9];

        state_est_msg.twist.twist.angular.x = x_est_[10];
        state_est_msg.twist.twist.angular.y = x_est_[11];
        state_est_msg.twist.twist.angular.z = x_est_[12];
        state_est_pub_.publish(state_est_msg);
    }
    return;
}



int main(int argc, char** argv) {

    ros::init(argc, argv, "gp_mhe_node");
    ros::NodeHandle node("~");

    // Instantiate MHE node
    Node mhe_node(node);

    // keep spinning while ROS is running
    ros::spin();

    return 0;
}

