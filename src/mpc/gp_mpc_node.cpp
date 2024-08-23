/**
 * @file gp_mpc_node.cpp
 * @author Wnooo Choo
 * @date July 2024
 * 
 * @copyright
 * Copyright (C) 2024.
 */

#include <gp_mpc_node.h>

using namespace ros;
using namespace Eigen;

// Constructor
Node::Node(ros::NodeHandle& nh) {
    initLaunchParameters(nh);
    initSubscribers(nh);
    initPublishers(nh);
    initRosService(nh);
    gp_mpc_ = new GP_MPC();
    if (gp_mpc_ == nullptr) {
        ROS_ERROR("FAILED TO CREATE GP_MPC INSTANCE");
        std::exit(EXIT_FAILURE);
    } 
    assert(quad_name_ != "");

    // initialize vector sizes
    x_.resize(MPC_NX);
    p_.resize(N_POSITION_STATES);
    q_.resize(N_QUATERNION_STATES);
    v_.resize(N_VELOCITY_STATES);
    w_.resize(N_RATE_STATES);
    x_opt_.resize(MPC_NX);
    u_opt_.resize(MPC_NU);  
    x_ref_prov_.resize(MPC_NX);
    u_ref_prov_.resize(MPC_NU);
    
    x_available_ = false;
    // Start thread to publish mpc status
    status_thread_ = std::thread(&Node::run, this);

    ROS_INFO("MPC: Waiting for System States...");
    ros::Rate rate(1);
    while (!x_available_ && ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("MPC: %s Loaded in %s", quad_name_.c_str(), environment_.c_str());
}

/// Destructor
Node::~Node() {
    ROS_INFO("MPC Destructor is called!");
    landing_ = true;
    ROS_INFO("Landing...");
    // Destructor that waits for ground_level_ to become true
    ros::Rate rate(1); // Adjust the rate as needed
    while (ros::ok() && !ground_level_) {
        ros::spinOnce(); // Process any pending callbacks
        rate.sleep();    // Sleep to avoid high CPU usage
    }

    if (mpc_thread_.joinable()) {
        mpc_thread_.join();
    }
    if (status_thread_.joinable()) {
        status_thread_.join();
    }
    delete gp_mpc_;
    // TODO: can i clear ros params
}

void Node::initLaunchParameters(ros::NodeHandle& nh) {
    std::string ns = ros::this_node::getNamespace();

    // System environment
    nh.param<std::string> (ns + "/environment", environment_, "arena");
    nh.param<std::string>(ns + "/quad_name", quad_name_, "");
    nh.param<bool>(ns + "/use_groundtruth", use_groundtruth_, true);

    // MPC Parameters
    nh.param<int>(ns + "/control_freq_factor", control_freq_factor_, 5);
    // Gazebo Specific MPC Parameters
    nh.param<double>(ns + "/quad_max_thrust", quad_max_thrust_, 5);
    nh.param<double>(ns + "/quad_mass", quad_mass_, 5);

    // Subscriber topic names
    nh.param<std::string>(ns + "/ref_topic", ref_topic_, ns + "/reference");
    nh.param<std::string>(ns + "/state_est_topic", state_est_topic_, "/" + quad_name_ + "/state_est");
    nh.param<std::string>(ns + "/odom_topic", odom_topic_, "/" + quad_name_ + "/ground_truth/odometry");
    nh.param<std::string>(ns + "/land", land_topic_, "/" + quad_name_ + "/land");

    // Publisher topic names
    nh.param<std::string>(ns + "/control_topic", control_topic_, "/mavros/setpoint_raw/attitude");
    nh.param<std::string>(ns + "/motor_thrust_topic", motor_thrust_topic_, "/" + quad_name_ + "/motor_thrust");
    nh.param<std::string>(ns + "/record_topic", record_topic_, "/" + quad_name_ + "/record"); 
    nh.param<std::string>(ns + "/status_topic", status_topic_, "/mpc/busy"); 
    // Gazebo Specific Publisher topic names
    nh.param<std::string>(ns + "/control_gz_topic", control_gz_topic_, "/" + quad_name_ + "/autopilot/control_command_input");

    // ROS Service topic names
    nh.param<std::string>(ns + "/mavros_set_mode_srvc", mavros_set_mode_srvc_, "/mavros/set_mode");
    nh.param<std::string>(ns + "/mavros_arming_srvc", mavros_arming_srvc_, "/mavros/cmd/arming");

    // Initial flight parameters
    nh.param<double>(ns + "/init_thr", init_thr_, 0.5); // Error allowed for initial position
    nh.param<double>(ns + "/init_v", init_v_, 0.3); // Velocity approaching initial position

    // Landing Parameters
    nh.param<double>(ns + "/land_thr", land_thr_, 0.05); // Error allowed for landing
    nh.param<double>(ns + "/land_z", land_z_, 0.05); // landing height
    nh.param<double>(ns + "/land_dz", land_dz_, 0.1); // landing velocity
}

void Node::initSubscribers(ros::NodeHandle& nh) {
    ref_sub_ = nh.subscribe<gp_rhce::ReferenceTrajectory> (
        ref_topic_, 10, &Node::referenceCallback, this);
    if (use_groundtruth_) {
        state_est_sub_ = nh.subscribe<nav_msgs::Odometry> (
            odom_topic_, 10, &Node::stateEstCallback, this);
    } else {
        state_est_sub_ = nh.subscribe<nav_msgs::Odometry> (
            state_est_topic_, 10, &Node::stateEstCallback, this);
    }
    land_sub_ = nh.subscribe<std_msgs::Bool> (
        land_topic_, 10, &Node::landCallback, this);
}

void Node::initPublishers(ros::NodeHandle& nh) {
    control_pub_ = nh.advertise<mavros_msgs::AttitudeTarget> (control_topic_, 10, true);
    motor_thrust_pub_ = nh.advertise<mav_msgs::Actuators> (motor_thrust_topic_, 10, true);
    record_pub_ = nh.advertise<std_msgs::Bool> (record_topic_, 10, true); 
    status_pub_ = nh.advertise<std_msgs::Bool> (status_topic_, 10, true);
    // Gazebo Specific Publisher
    control_gz_pub_ = nh.advertise<quadrotor_msgs::ControlCommand> (control_gz_topic_, 10, true);
}

void Node::initRosService(ros::NodeHandle& nh) {
    set_mode_client_ = nh.serviceClient<mavros_msgs::SetMode> (mavros_set_mode_srvc_);
    arming_client_ = nh.serviceClient<mavros_msgs::CommandBool> (mavros_arming_srvc_);
}

void Node::landCallback(const std_msgs::Bool::ConstPtr& msg) {
    if (msg->data) {
        // Lower drone to a safe height
        landing_ = true;
        ROS_INFO("Landing...");
        // Stop recording
        std_msgs::Bool msg;
        msg.data = false;
        record_pub_.publish(msg);
    }
}

void Node::referenceCallback(const gp_rhce::ReferenceTrajectory::ConstPtr& msg) {
    if (x_ref_.empty()) {
        ref_len_ = msg->seq_len;
        // TODO: ACCEPT HOVER COMMANDS
        // if (ref_len_ == 0) {
        //     // Hover-in-place mode
        //     x_ref_.clear();
        //     x_ref_.push_back(x_);
        //     u_ref_.clear();
        //     t_ref_.clear();
        //     return;
        // }

        // Save reference name
        ref_traj_name_ = msg->traj_name;

        // Save reference trajectory, inputs and relative times 
        std::vector<double> trajectory = msg->trajectory;
        x_ref_.resize(ref_len_, std::vector<double>(MPC_NX));
        for (std::size_t i = 0; i < ref_len_; i++) {
            std::copy(trajectory.begin() + (i * MPC_NX), trajectory.begin() + (i * MPC_NX) + MPC_NX, x_ref_[i].begin());
        }
        std::vector<double> u_ref = msg->inputs;
        u_ref_.resize(ref_len_, std::vector<double>(MPC_NU));
        for (std::size_t i = 0; i < ref_len_; i++) {
            for (std::size_t j = 0; j < MPC_NU; j ++) {
                u_ref_[i][j] = u_ref[i * MPC_NU + j];
            }
        }
        t_ref_.insert(t_ref_.end(), msg->dt.begin(), msg->dt.end());
        
        if (!t_ref_.empty()) {  // Check if the vector is not empty
            ROS_INFO("New trajectory received. Time duration: %.2f s", t_ref_.back());
        } else {
            ROS_WARN("Trajectory vector is empty.");    
        }
    } else {
        ROS_WARN("Ignoring new trajectory received. Still in execution of another trajectory.");
    }
}

void Node::stateEstCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // double time = ros::Time::now().toNSec() * 1e6;
    // std::cout << time << " " << msg->header.seq << std::endl;
    p_ = {msg->pose.pose.position.x,
          msg->pose.pose.position.y,
          msg->pose.pose.position.z}; 
    q_ = {msg->pose.pose.orientation.w, 
          msg->pose.pose.orientation.x,
          msg->pose.pose.orientation.y, 
          msg->pose.pose.orientation.z};
    v_ = {msg->twist.twist.linear.x, 
          msg->twist.twist.linear.y, 
          msg->twist.twist.linear.z};
    w_ = {msg->twist.twist.angular.x,
          msg->twist.twist.angular.y,
          msg->twist.twist.angular.z};

    if (environment_ == "gazebo") {
        // If its Gazebo transform v_B to v_W
        // Load as Eigen to perform Rotation
        Eigen::Quaterniond q_eig(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                            msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        Eigen::Vector3d v_eig(msg->twist.twist.linear.x, 
                        msg->twist.twist.linear.y, 
                        msg->twist.twist.linear.z);
        // Apply rotation to get velocity in world frame
        v_eig = q_eig * v_eig;
        // Save to std::vector
        v_ = {v_eig.x(), v_eig.y(), v_eig.z()};
    }

    // concatenate p, q, v, w into x_
    std::copy(p_.begin(), p_.end(), x_.begin());
    std::copy(q_.begin(), q_.end(), x_.begin() + IDX_QUATERNION_START);
    std::copy(v_.begin(), v_.end(), x_.begin() + IDX_VELOCITY_START);
    std::copy(w_.begin(), w_.end(), x_.begin() + IDX_RATE_START);


    x_available_ = true;

    // Only optimize once every two messages (ie. state est: 100Hz, control: 50Hz)
    if (!optimize_next_) {
        if (mpc_thread_.joinable()) {
            mpc_thread_.join();
        }
        // If currently on trjaectory tracking, check for any skipped messages.
        if (x_initial_reached_) {
            int skipped_messages = (int)(msg->header.seq - last_state_est_seq_num_ - 1);
            if (skipped_messages > 0) {
                ROS_WARN("MPC Recording time skipped messages: %d", skipped_messages);
            }
            // Adjust current index in traj.
            mpc_idx_ += skipped_messages / 2;
            // If odd number of skipped messages, do optimization
            if (skipped_messages > 0 && skipped_messages % 2 == 1) {
                mpc_thread_ = std::thread(&Node::runMPC, this);
                // mpc_thread_.detach(); // Use detach if you don't need to join later
                last_state_est_seq_num_ = msg->header.seq;
                optimize_next_ = false;
                return;
            }
        }
        optimize_next_ = true;
        return;
    } 
    
    if (msg->header.seq > last_state_est_seq_num_ + 2 && x_initial_reached_) {
        // If one message was skipped at this point, then the reference is already late. Compensate by
        // optimizing twice in a row and hope to do it fast...
        std::thread mpc_thread(&Node::runMPC, this);
        mpc_thread.detach();
        optimize_next_ = true;
        ROS_WARN("Odometry skipped at Optimization step. Last: %d, current: %d", msg->header.seq, last_state_est_seq_num_);
        last_state_est_seq_num_ = msg->header.seq;
        return;
    }

    // Everything is operating as normal
    if (mpc_thread_.joinable()) {
        mpc_thread_.join();
    }
    mpc_thread_ = std::thread(&Node::runMPC, this);
    last_state_est_seq_num_ = msg->header.seq;
    optimize_next_ = false;
    
}

void Node::runMPC() {
    if (!x_available_) {
        ROS_WARN("State Estimation Not available.");
        return;
    }

    // Set Reference Trajectory
    setReferenceTrajectory();
    // optimize MPC
    try {
        if (gp_mpc_->solveMPC(x_, u_opt_) == ACADOS_SUCCESS) {
            gp_mpc_->getControls(x_opt_, u_opt_);
            optimization_dt_ += gp_mpc_->getOptimizationTime();
        } else {
            ROS_WARN("Tried to run an MPC optimization but was unsuccessful.");
        }  
    } catch (const std::runtime_error& e) {
        ROS_WARN("Tried to run an MPC optimization but MPC is not ready yet.");
        return;
    }

    // Publish Controls
    if (environment_ == "arena") {
        mavros_msgs::AttitudeTarget next_control;
        next_control.header.stamp = ros::Time::now();
        next_control.header.seq = mpc_seq_num_++; 
        next_control.body_rate.x = x_opt_[IDX_RATE_X];
        next_control.body_rate.y = x_opt_[IDX_RATE_Y];
        next_control.body_rate.z = x_opt_[IDX_RATE_Z];
        next_control.type_mask = 128;
        double thrust = (u_opt_[0] + u_opt_[1] + u_opt_[2] + u_opt_[3]) / 4;
        if (ground_level_) {
            thrust *= 0.5;
        } 
        next_control.thrust = thrust;
        control_pub_.publish(next_control);
    } else if (environment_ == "gazebo") {
        quadrotor_msgs::ControlCommand next_control;
        next_control.header.stamp = ros::Time::now();
        next_control.header.seq = mpc_seq_num_++;
        next_control.armed = true;
        next_control.bodyrates.x = x_opt_[IDX_RATE_X];
        next_control.bodyrates.y = x_opt_[IDX_RATE_Y];
        next_control.bodyrates.z = x_opt_[IDX_RATE_Z];
        next_control.control_mode = 2;
        double collective_thrust = (u_opt_[0] + u_opt_[1] + u_opt_[2] + u_opt_[3]);
        collective_thrust *= quad_max_thrust_;
        collective_thrust /= quad_mass_ ;
        if (ground_level_) {
            collective_thrust *= 0.01;
        }
        next_control.collective_thrust = collective_thrust ;
        control_gz_pub_.publish(next_control);
    }
    // Publish motor thrusts
    mav_msgs::Actuators motor_thrusts;
    motor_thrusts.header.stamp = ros::Time::now();
    motor_thrusts.angular_velocities.assign(u_opt_.begin(), u_opt_.end());
    motor_thrust_pub_.publish(motor_thrusts);
}

void Node::setReferenceTrajectory() {
    if (!x_available_) {
        return;
    }

    // Check if landing mode
    if (landing_) {
        std::vector<std::vector<double>> x_ref(1, std::vector<double>(MPC_NX, 0.0));
        std::vector<std::vector<double>> u_ref(1, std::vector<double>(MPC_NU, 0.0));
        std::copy(x_ref_.back().begin(), x_ref_.back().begin() + IDX_VELOCITY_START, x_ref[0].begin());
        double dz = (land_z_ > x_[2]) ? land_dz_ : (land_z_ < x_[2]) ? -land_dz_ : 0.0;
        x_ref[0][2] = (dz > 0) ? std::min(land_z_, x_[2] + dz) : std::max(land_z_, x_[2] + dz);
        
        // Reached landing height
        if (std::abs(x_[2] - land_z_) < land_thr_) {
            if (!ground_level_) {
                ROS_INFO("Vehicle at Ground Level");
                ground_level_ = true;
            }
            // TODO: Disarming not working
            // if (environment_ == "arena") {
            //     mavros_msgs::CommandBool disarm_cmd;
            //     disarm_cmd.request.value = false;
            //     if (arming_client_.call(disarm_cmd) &&
            //                 !disarm_cmd.response.success) {
            //         // Disarm Failed
            //         return gp_mpc_->setReference(x_ref, u_ref);
            //     }
            //     ROS_INFO("Vehicle Disarmed");
            // }
            x_ref_.clear();
            u_ref_.clear();
            t_ref_.clear();
            // TODO: Check if it gets cleared here correctly. its not accepting new trajectory
            x_initial_reached_ = false;
            mpc_idx_ = 0;
            landing_ = false;
        }
        return gp_mpc_->setReference(x_ref, u_ref);
    }

    // Check if reference trajectory is set up. If not, pick current position and keep hovering
    if (x_ref_.empty()) {
        std::vector<std::vector<double>> x_ref(1, std::vector<double>(MPC_NX, 0.0));
        std::vector<std::vector<double>> u_ref(1, std::vector<double>(MPC_NU, 0.0));
        // Select current position as provisional hovering position
        if (x_ref_prov_.empty()) {
            x_ref_prov_.insert(x_ref_prov_.end(), x_.begin(), x_.begin() + N_POSITION_STATES + N_QUATERNION_STATES);
            x_ref_prov_.insert(x_ref_prov_.end(), N_VELOCITY_STATES + N_RATE_STATES , 0);
            ROS_INFO("Selecting current position as provisional setpoint.");
        }
        if (u_ref_prov_.empty()) {
            u_ref_prov_ = {0, 0, 0, 0};
        }
        // Fill x_ref with x_ref_prov_
        for (auto& row: x_ref) {
            std::copy(x_ref_prov_.begin(), x_ref_prov_.end(), row.begin());
        }
        // Fill u_ref with u_ref_prov_x_available_
        for (auto& row: u_ref) {
            std::copy(u_ref_prov_.begin(), u_ref_prov_.end(), row.begin());
        }
        return gp_mpc_->setReference(x_ref, u_ref);
    }

    // If reference exists then exit out of provisional hovering mode
    if (!x_ref_prov_.empty()) {
        x_ref_prov_.clear();
        u_ref_prov_.clear();
        ground_level_ = false;
        ROS_INFO("Abandoning provisional setpoint.");
    }

    // Check if starting position of trajectory has been reached
    if (!x_initial_reached_) {
        std::vector<std::vector<double>> x_ref(1, std::vector<double>(MPC_NX, 0.0));
        std::vector<std::vector<double>> u_ref(1, std::vector<double>(MPC_NU, 0.0));
        // Compute distance to initial position of trajectory
        // Convert std::vector to Eigen::VectorXd
        Eigen::VectorXd x_vec = Eigen::Map<const Eigen::VectorXd>(x_.data(), x_.size());
        Eigen::VectorXd x_ref_vec = Eigen::Map<const Eigen::VectorXd>(x_ref_[0].data(), x_ref_[0].size());
        // Create quaternion error from x and x_ref
        Eigen::Quaterniond q_eig(x_vec[3], x_vec[4], x_vec[5], x_vec[6]); // Note the order: w, x, y, z
        Eigen::Quaterniond q_ref(x_ref_vec[3], x_ref_vec[4], x_ref_vec[5], x_ref_vec[6]);
        Eigen::Quaterniond q_error = q_eig * q_ref.inverse();
        Eigen::VectorXd e(x_vec.size() - N_RATE_STATES - 1); // Adjust size based on actual dimensions
        // Fill e with concatenation of differences
        e << (x_vec.head(N_POSITION_STATES) - x_ref_vec.head(N_POSITION_STATES)),
              q_error.vec(),
             (x_vec.segment(IDX_VELOCITY_START, N_VELOCITY_STATES) - x_ref_vec.segment(IDX_VELOCITY_START, N_VELOCITY_STATES));
        if (std::sqrt(e.dot(e)) < init_thr_) {
            // Initial position of trajectory has been reached
            x_initial_reached_ = true;
            optimization_dt_ = 0;
            ROS_INFO("Reached initial position of trajectory.");
            // Tell MHE to begin recording
            std_msgs::Bool msg;
            msg.data = true;
            record_pub_.publish(msg);
            // Set reference to initial linear and angualr position of trajectory
            std::copy(x_ref_[0].begin(), x_ref_[0].begin() + IDX_VELOCITY_START, x_ref[0].begin());
        } else {
            // Initial Position has not been reached yet 
            // Fly the drone toward initial position of trajectory
            double dx = (x_ref_[0][0] > x_[0]) ? init_v_ : (x_ref_[0][0] < x_[0]) ? -init_v_ : 0.0;
            double dy = (x_ref_[0][1] > x_[1]) ? init_v_ : (x_ref_[0][1] < x_[1]) ? -init_v_ : 0.0;
            double dz = (x_ref_[0][2] > x_[2]) ? init_v_ : (x_ref_[0][2] < x_[2]) ? -init_v_ : 0.0;
            // Set reference position in the direction of or at the init. position of trajectory
            x_ref[0][0] = (dx > 0) ? std::min(x_ref_[0][0], x_[0] + dx) : std::max(x_ref_[0][0], x_[0] + dx);
            x_ref[0][1] = (dy > 0) ? std::min(x_ref_[0][1], x_[1] + dy) : std::max(x_ref_[0][1], x_[1] + dy);
            x_ref[0][2] = (dz > 0) ? std::min(x_ref_[0][2], x_[2] + dz) : std::max(x_ref_[0][2], x_[2] + dz);
            // Set angular position of drone to init. angular position of trajectory
            std::copy(x_ref_[0].begin()+IDX_QUATERNION_START, x_ref_[0].begin()+IDX_VELOCITY_START, x_ref[0].begin()+IDX_QUATERNION_START);
        }
        return gp_mpc_->setReference(x_ref, u_ref);
    }
    
    // Executing Trajectory Tracking
    if (mpc_idx_ < ref_len_) {
        std::vector<std::vector<double>> x_ref(MPC_N, std::vector<double>(MPC_NX, 0.0));
        std::vector<std::vector<double>> u_ref(MPC_N, std::vector<double>(MPC_NU, 0.0));
        for (int i = 0; i < MPC_N; ++i) {
            // TODO: Not exactly same as np.arange but could be okay
            int ii = std::min(i * control_freq_factor_ + mpc_idx_, ref_len_-1);
            std::copy(x_ref_[ii].begin(), x_ref_[ii].end(), x_ref[i].begin());
            std::copy(u_ref_[ii].begin(), u_ref_[ii].end(), u_ref[i].begin());
        }
        mpc_idx_++;
        return gp_mpc_->setReference(x_ref, u_ref);    

    } else if (mpc_idx_ == ref_len_) { 
        // End of reference reached
        std::vector<std::vector<double>> x_ref(1, std::vector<double>(MPC_NX, 0.0));
        std::vector<std::vector<double>> u_ref(1, std::vector<double>(MPC_NU, 0.0));
        // Compute MSE
        optimization_dt_ /= mpc_idx_;
        ROS_INFO("Tracking complete. Mean MPC opt. time: %.3f ms", optimization_dt_*1000);

        // Lower drone to a safe height
        landing_ = true;
        ROS_INFO("Landing...");

        // Set reference to final linear and angualr position of trajectory
        std::copy(x_ref_.back().begin(), x_ref_.back().begin() + IDX_VELOCITY_START, x_ref[0].begin());

        // Stop recording
        x_initial_reached_ = false;
        std_msgs::Bool msg;
        msg.data = false;
        record_pub_.publish(msg);
        return gp_mpc_->setReference(x_ref, u_ref);
    }
}

void Node::run() {
    ros::Rate rate(1);
    while (!ros::isShuttingDown()) {
        std_msgs::Bool msg;
        msg.data = !(x_ref_.empty() && x_available_);
        status_pub_.publish(msg);
        rate.sleep();
    }
}

int main(int argc, char** argv) {

    ros::init(argc, argv, "gp_mpc_node");
    ros::NodeHandle node("~");

    // Instantiate MPC node
    Node mpc_node(node);

    // keep spinning while ROS is running
    ros::spin();

    return 0;
}

