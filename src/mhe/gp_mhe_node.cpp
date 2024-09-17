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
    assert(quad_name_ != "");
    gp_mhe_ = new GP_MHE(mhe_type_);
    if (gp_mhe_ == nullptr) {
        ROS_ERROR("FAILED TO CREATE GP_MHE INSTANCE");
        std::exit(EXIT_FAILURE);
    } 

    // Init vectors
    x_est_ = VectorXd::Zero(MHE_NX);
    x_est_[IDX_Q_W] = 1;
    p_ = VectorXd::Zero(N_POSITION_STATES);
    w_ = VectorXd::Zero(N_RATE_STATES);
    a_ = VectorXd::Zero(N_ACCEL_STATES);
    y_ = VectorXd::Zero(gp_mhe_->getMeasStateLen() + MHE_NU);
    u_ = VectorXd::Zero(N_MOTORS);
    y_hist_ = MatrixXd::Zero(MHE_N, gp_mhe_->getMeasStateLen() + MHE_NU);
    y_hist_cp_ = MatrixXd::Zero(MHE_N, gp_mhe_->getMeasStateLen() + MHE_NU);
    u_hist_ = MatrixXd::Zero(MHE_N, N_MOTORS);
    u_hist_cp_ = MatrixXd::Zero(MHE_N, N_MOTORS);
    if (mhe_type_ == "kinematic") {
        acceleration_est_ = VectorXd::Zero(3);
    }
    
    if (p_.isZero()) {
        ROS_INFO("MHE: Waiting for Sensor Measurement...");
        ros::Rate rate(1);
        while (p_.isZero() && ros::ok()) {
            ros::spinOnce();
            rate.sleep();
        }
    }
    if (mhe_type_ == "dynamic" && u_.isZero()) {
        ROS_INFO("Motor Thrusts not received yet. Skipping time step...");
        ros::Rate rate(1);
        while (u_.isZero() && ros::ok()) {
            ros::spinOnce();
            rate.sleep();
        }
    }

    optimization_dt_ = 0;
    mhe_idx_ = 0;
    
    ROS_INFO("MHE: %s Loaded in %s", quad_name_.c_str(), environment_.c_str());
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
    nh.param<std::string>(ns + "/pose_topic", pose_topic_, "/mocap/" + quad_name_ +"/pose");
    nh.param<std::string>(ns + "/motor_thrust_topic", motor_thrust_topic_, "/" + quad_name_ + "/motor_thrust");
    nh.param<std::string>(ns + "/record_topic", record_topic_, "/" + quad_name_ + "/record");
    // Gazebo Specific Subscriber topic names
    nh.param<std::string>(ns + "/odom_gz_topic", odom_gz_topic_, "/" + quad_name_ + "/ground_truth/odometry");

    // Publisher topic names
    nh.param<std::string>(ns + "/state_est_topic", state_est_topic_, "/" + quad_name_ + "/state_est");
    nh.param<std::string>(ns + "/acceleration_est_topic", acceleration_est_topic_, "/" + quad_name_ + "/acceleration_est");
}

void Node::initSubscribers(ros::NodeHandle& nh) {
    ros::TransportHints transport_hints;
    transport_hints.tcpNoDelay(true);
    
    if (environment_ == "gazebo") {
        // Gazebo specific Subscribers
        odom_gz_sub_ = nh.subscribe<nav_msgs::Odometry> (
            odom_gz_topic_, 10, &Node::odomGzCallback, this, transport_hints=transport_hints);
    }
    // Init Subscribers
    motor_thrust_sub_ = nh.subscribe<mav_msgs::Actuators> (
        motor_thrust_topic_, 10, &Node::motorThrustCallback, this, transport_hints=transport_hints);
    pose_sub_ = nh.subscribe<geometry_msgs::PoseStamped> (
        pose_topic_, 10, &Node::poseCallback, this, transport_hints=transport_hints);
    record_sub_ = nh.subscribe<std_msgs::Bool> (
        record_topic_, 10, &Node::recordMheCallback, this, transport_hints=transport_hints);
    imu_sub_ = nh.subscribe<sensor_msgs::Imu> (
            imu_topic_, 3, &Node::imuCallback, this, transport_hints=transport_hints);
    
}

void Node::initPublishers(ros::NodeHandle& nh) {
    acceleration_est_pub_ = nh.advertise<sensor_msgs::Imu> (acceleration_est_topic_, 10, true);
    state_est_pub_ = nh.advertise<nav_msgs::Odometry> (state_est_topic_, 10, true);
}


void Node::imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
    // ROS_INFO("%.4f",msg->header.stamp.toSec());
    if (p_.isZero()) {
        // ROS_WARN("Position measurement not received yet. Skipping time step...");
        return;
    } 
    if (mhe_type_ == "dynamic" && u_.isZero()) {
        return;
    }

    // Concatenate p, w, and a into y
    if (mhe_type_ == "kinematic") {
        y_ << p_[0],
              p_[1],
              p_[2],
              msg->angular_velocity.x,
              msg->angular_velocity.y,
              msg->angular_velocity.z,
              msg->linear_acceleration.x, 
              msg->linear_acceleration.y, 
              msg->linear_acceleration.z;
    } else {
        y_ << p_[0],
              p_[1],
              p_[2],
              msg->angular_velocity.x,
              msg->angular_velocity.y,
              msg->angular_velocity.z;
    }


    // Check if there are any skipped messages
    int skipped_messages = 0;
    if (last_imu_seq_number_ > 0 && record_) {
        skipped_messages = int(msg->header.seq - last_imu_seq_number_ - 1);
        if (skipped_messages > 0) {
            ROS_WARN("MHE Recording time skipped messages: %d", skipped_messages);
        }
    }
    lock_.lock();
    // Fill empty vectors with current sensor measurements and motor thrust
    if (!hist_received_) {
        for (int i = 0; i < MHE_N; ++i) {
            y_hist_.row(i) = y_;
            u_hist_.row(i) = u_;
        }
        hist_received_ = true;
    }

    // Shift all elements to the left once
    y_hist_.block(0, 0, MHE_N-1, y_hist_.cols()) = y_hist_.block(1, 0, MHE_N-1, y_hist_.cols());
    u_hist_.block(0, 0, MHE_N-1, u_hist_.cols()) = u_hist_.block(1, 0, MHE_N-1, u_hist_.cols());
    // Insert the new element at the last position
    y_hist_.row(MHE_N-1) = y_;
    u_hist_.row(MHE_N-1) = u_;

    // Add current measurement to array and also add the number of missed measurements to be up to sync
    for (int i = 0; i < skipped_messages; ++i) {
        y_hist_.block(0, 0, MHE_N-1, y_hist_.cols()) = y_hist_.block(1, 0, MHE_N-1, y_hist_.cols());
        u_hist_.block(0, 0, MHE_N-1, u_hist_.cols()) = u_hist_.block(1, 0, MHE_N-1, u_hist_.cols());
        y_hist_.row(MHE_N-1) = y_;
        u_hist_.row(MHE_N-1) = u_;
    }
    lock_.unlock();

    // Run MHE
    if (mhe_thread_.joinable()) {
        mhe_thread_.join();
    }
    mhe_thread_ = std::thread(&Node::runMHE, this);
}

void Node::recordMheCallback(const std_msgs::Bool::ConstPtr& msg) {
    if (record_ && !msg->data) {
        optimization_dt_ /= mhe_idx_;
        ROS_INFO("Estimation complete. Mean MHE opt. time: %.3f ms", optimization_dt_*1000);
        optimization_dt_ = 0;
    } else if (!record_ && msg->data) {
        optimization_dt_ = 0;
        mhe_idx_ = 0;
    }
    record_ = msg->data;
}

void Node::motorThrustCallback(const mav_msgs::Actuators::ConstPtr& msg) {
    u_ << msg->angular_velocities[0],
          msg->angular_velocities[1],
          msg->angular_velocities[2],
          msg->angular_velocities[3];
}

void Node::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    p_ << msg->pose.position.x, 
          msg->pose.position.y, 
          msg->pose.position.z;
}

void Node::odomGzCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    p_ << msg->pose.pose.position.x,
          msg->pose.pose.position.y,
          msg->pose.pose.position.z;
}

void Node::runMHE() {
    try {
        // optimize MHE
        lock_.lock();
        y_hist_cp_ = y_hist_;
        u_hist_cp_ = u_hist_;
        lock_.unlock();
        if (gp_mhe_->solveMHE(y_hist_cp_, u_hist_cp_) == ACADOS_SUCCESS) {
            gp_mhe_->getStateEst(x_est_);
            optimization_dt_ += gp_mhe_->getOptimizationTime();
            ++mhe_idx_;
        } else {
            ROS_WARN("Tried to run an MHE optimization but was unsuccessful.");
        }  
    } catch (const std::runtime_error& e) {
        ROS_WARN("Tried to run an MHE optimization but MHE is not ready yet.");
        return;
    }  

    if (mhe_type_ == "kinematic") {
        // std::copy(x_est_.begin() + 13, x_est_.end(), acceleration_est_.begin());
        acceleration_est_ = x_est_.segment(MHE_NX - N_ACCEL_STATES, N_ACCEL_STATES);
        sensor_msgs::Imu acceleration_est_msg;
        acceleration_est_msg.header.stamp = ros::Time::now();
        acceleration_est_msg.header.seq = mhe_seq_num_;
        acceleration_est_msg.linear_acceleration.x = acceleration_est_[0];
        acceleration_est_msg.linear_acceleration.y = acceleration_est_[1];
        acceleration_est_msg.linear_acceleration.z = acceleration_est_[2];
        acceleration_est_pub_.publish(acceleration_est_msg);
    }

    // Update the state estimate of the quad
    nav_msgs::Odometry state_est_msg;
    state_est_msg.header.stamp = ros::Time::now();
    state_est_msg.header.seq = mhe_seq_num_++;
    state_est_msg.header.frame_id = "world";
    state_est_msg.child_frame_id = quad_name_ + "/base_link";

    state_est_msg.pose.pose.position.x = x_est_(0);
    state_est_msg.pose.pose.position.y = x_est_(1);
    state_est_msg.pose.pose.position.z = x_est_(2);

    state_est_msg.pose.pose.orientation.w = x_est_(3);
    state_est_msg.pose.pose.orientation.x = x_est_(4);
    state_est_msg.pose.pose.orientation.y = x_est_(5);
    state_est_msg.pose.pose.orientation.z = x_est_(6);

    state_est_msg.twist.twist.linear.x = x_est_(7);
    state_est_msg.twist.twist.linear.y = x_est_(8);
    state_est_msg.twist.twist.linear.z = x_est_(9);

    state_est_msg.twist.twist.angular.x = x_est_(10);
    state_est_msg.twist.twist.angular.y = x_est_(11);
    state_est_msg.twist.twist.angular.z = x_est_(12);
    state_est_pub_.publish(state_est_msg);
    return;
}



int main(int argc, char** argv) {

    ros::init(argc, argv, "gp_mhe_node");
    ros::NodeHandle node("~");

    // Instantiate MHE node
    Node mhe_node(node);

    // keep spinning while ROS is running
    ros::spin();
    // ros::Rate rate(100);
    // while (ros::ok()) {
    //     ros::spinOnce();
    //     rate.sleep();
    // }

    return 0;
}

