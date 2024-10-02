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
namespace fs = std::filesystem;

// Constructor
Node::Node(ros::NodeHandle& nh) {
    // initialize vector sizes
    x_ = VectorXd::Zero(MPC_NX);
    p_ = VectorXd::Zero(N_POSITION_STATES);
    q_.coeffs() << 0, 0, 0, 1;
    v_ = VectorXd::Zero(N_VELOCITY_STATES);
    w_ = VectorXd::Zero(N_RATE_STATES);
    x_opt_ = VectorXd::Zero(MPC_NX);
    u_opt_ = VectorXd::Zero(MPC_NU);  
    x_ref_prov_ = VectorXd::Zero(MPC_NX);
    u_ref_prov_ = VectorXd::Zero(MPC_NU);
    x_available_ = false;
    ref_received_ = false;

    initLaunchParameters(nh);
    initSubscribers(nh);
    initPublishers(nh);
    initRosService(nh);

    gp_mpc_ = new GP_MPC();
    if (gp_mpc_ == nullptr) {
        ROS_ERROR("FAILED TO CREATE GP_MPC INSTANCE");
        std::exit(EXIT_FAILURE);
    } 
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
    assert(quad_name_ != "");

    // MPC Parameters
    nh.param<int>(ns + "/control_freq_factor", control_freq_factor_, 5);
    // Gazebo Specific MPC Parameters
    nh.param<double>(ns + "/quad_max_thrust", quad_max_thrust_, 5);
    nh.param<double>(ns + "/quad_mass", quad_mass_, 5);
    nh.param<bool>(ns + "/with_gp", with_gp_, false);

    // Load GP models and initialize GP related vectors/matrices
    if (with_gp_) {
        nh.param<std::string> ("/gp_model_dir", gp_model_dir_, "");
        nh.param<std::string> (ns + "/gp_model_name", gp_model_name_, "");
        assert(gp_model_dir_ != "" || gp_model_name_ != "");
        nh.param<std::vector<int>> (ns + "/x_features", x_features_, {7, 8, 9});
        nh.param<std::vector<int>> (ns + "/y_features", y_features_, {7, 8, 9});
        n_features_ = x_features_.size();
        gp_corr_ = MatrixXd::Zero(MPC_N, n_features_);
        mpc_param_ = MatrixXd::Zero(MPC_N, MPC_NP);
        mpc_param_(0, MPC_NP-1) = 1; // TRIGGER VALUE
        gp_model_.resize(n_features_);
        try {
            for (int i = 0; i < n_features_; ++i) {
                std::string gp_model_file = gp_model_dir_ + gp_model_name_ + "/scripted_gpy_model_" + std::to_string(x_features_[i]) + ".pth";
                if (fs::exists(gp_model_file)) {
                    gp_model_[i] = torch::jit::load(gp_model_file);
                } else {
                    ROS_ERROR("GP Model %s does not exists", gp_model_name_.c_str());
                }
            }
            ROS_INFO("MPC: Loaded GP Models");
        } catch (const c10::Error& e) {
            ROS_ERROR("MPC: Error Loading the GP Model");
            return;
        }
    }

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
    ros::TransportHints transport_hints;
    transport_hints.tcpNoDelay(true);
    
    ref_sub_ = nh.subscribe<gp_rhce::ReferenceTrajectory> (
        ref_topic_, 10, &Node::referenceCallback, this, transport_hints=transport_hints);
    if (use_groundtruth_) {
        state_est_sub_ = nh.subscribe<nav_msgs::Odometry> (
            odom_topic_, 10, &Node::stateEstCallback, this, transport_hints=transport_hints);
    } else {
        state_est_sub_ = nh.subscribe<nav_msgs::Odometry> (
            state_est_topic_, 10, &Node::stateEstCallback, this, transport_hints=transport_hints);
    }
    land_sub_ = nh.subscribe<std_msgs::Bool> (
        land_topic_, 5, &Node::landCallback, this);
}

void Node::initPublishers(ros::NodeHandle& nh) {
    control_pub_ = nh.advertise<mavros_msgs::AttitudeTarget> (control_topic_, 5, true);
    motor_thrust_pub_ = nh.advertise<mav_msgs::Actuators> (motor_thrust_topic_, 5, true);
    record_pub_ = nh.advertise<std_msgs::Bool> (record_topic_, 5, true); 
    status_pub_ = nh.advertise<std_msgs::Bool> (status_topic_, 5, true);
    // Gazebo Specific Publisher
    control_gz_pub_ = nh.advertise<quadrotor_msgs::ControlCommand> (control_gz_topic_, 1, true);
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
    if (!ref_received_) {
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
        x_ref_ = MatrixXd(ref_len_, MPC_NX);
        for (std::size_t i = 0; i < ref_len_; i++) {
            x_ref_.row(i) = Eigen::Map<const Eigen::VectorXd>(trajectory.data() + (i * MPC_NX), MPC_NX);
        }
        
        std::vector<double> u_ref = msg->inputs;
        u_ref_ = MatrixXd(ref_len_, MPC_NU);
        for (std::size_t i = 0; i < ref_len_; i++) {
            u_ref_.row(i) = Eigen::Map<const Eigen::VectorXd>(u_ref.data() + i * MPC_NU, MPC_NU);
        }

        t_ref_ = VectorXd::Zero(ref_len_);
        t_ref_ = Eigen::Map<const Eigen::VectorXd>(msg->dt.data(), ref_len_);
        
        if (with_gp_) {
            gp_input_.resize(1);
            std::vector<at::Tensor> gp_corr_ref;
            gp_corr_ref.resize(n_features_);
            for (int i = 0; i < n_features_; ++i) {
                torch::Tensor input_tensor = torch::from_blob(x_ref_.col(i).data(), {ref_len_}, torch::kDouble);
                gp_input_[0] = input_tensor;
                // When GP model only predicts mean:
                torch::Tensor pred = gp_model_[i].forward(gp_input_).toTensor();
                gp_corr_ref[i] = pred.to(torch::kDouble);

                // When GP models predicts mean AND variance:
                // auto output = gp_model_[i].forward(gp_input_).toTuple();
                // torch::Tensor output_tensor = output->elements()[1].toTensor();
            }
            gp_corr_ref_ = tensorToEigen(gp_corr_ref);
        }
        if (!t_ref_.isZero()) {  // Check if the vector is not empty
            ROS_INFO("New trajectory received. Time duration: %.2f s", t_ref_(t_ref_.size() - 1));
            ref_received_ = true;
        } else {
            ROS_WARN("Trajectory vector is empty.");    
        }
    } else {
        ROS_WARN("Ignoring new trajectory received. Still in execution of another trajectory.");
    }
}

void Node::stateEstCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    p_ << msg->pose.pose.position.x,
          msg->pose.pose.position.y,
          msg->pose.pose.position.z; 
    q_.coeffs() << msg->pose.pose.orientation.x,
                   msg->pose.pose.orientation.y, 
                   msg->pose.pose.orientation.z,
                   msg->pose.pose.orientation.w;
    q_.normalize();
    v_ << msg->twist.twist.linear.x, 
          msg->twist.twist.linear.y, 
          msg->twist.twist.linear.z;
    w_ << msg->twist.twist.angular.x,
          msg->twist.twist.angular.y,
          msg->twist.twist.angular.z;

    if (environment_ == "gazebo") {
        // Gazebo gives velocity in Body frame
        vb_ << msg->twist.twist.linear.x, 
               msg->twist.twist.linear.y, 
               msg->twist.twist.linear.z;
        v_ = transformToWorldFrame(q_, v_);
    } else {
        vb_ = transformToBodyFrame(q_, v_);
    }

    // concatenate p, q, v, w into x_
    x_.head(N_POSITION_STATES) = p_;
    x_.segment(IDX_QUATERNION_START, N_QUATERNION_STATES) << q_.w(), q_.x(), q_.y(), q_.z();
    x_.segment(IDX_VELOCITY_START, N_VELOCITY_STATES) = v_;
    x_.tail(N_RATE_STATES) = w_;


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
        next_control.body_rate.x = x_opt_(IDX_RATE_X);
        next_control.body_rate.y = x_opt_(IDX_RATE_Y);
        next_control.body_rate.z = x_opt_(IDX_RATE_Z);
        next_control.type_mask = 128;
        double thrust = (u_opt_(0) + u_opt_(1) + u_opt_(2) + u_opt_(3)) / 4;
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
        next_control.bodyrates.x = x_opt_(IDX_RATE_X);
        next_control.bodyrates.y = x_opt_(IDX_RATE_Y);
        next_control.bodyrates.z = x_opt_(IDX_RATE_Z);
        next_control.control_mode = 2;
        double collective_thrust = (u_opt_(0) + u_opt_(1) + u_opt_(2) + u_opt_(3));
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
    motor_thrusts.angular_velocities = std::vector<double>(u_opt_.data(), u_opt_.data() + u_opt_.size());
    motor_thrust_pub_.publish(motor_thrusts);
}

void Node::setReferenceTrajectory() {
    if (!x_available_) {
        return;
    }

    // Check if landing mode
    if (landing_) {
        Eigen::MatrixXd x_ref = Eigen::MatrixXd::Zero(1, MPC_NX);
        Eigen::MatrixXd u_ref = Eigen::MatrixXd::Zero(1, MPC_NU);
        // Copy only the Position and Quaternion states from x_ref_
        x_ref.row(0).head(N_POSITION_STATES + N_QUATERNION_STATES) = x_ref_.row(x_ref_.rows() - 1).head(N_POSITION_STATES + N_QUATERNION_STATES);
        double dz = (land_z_ > x_(2)) ? land_dz_ : (land_z_ < x_(2)) ? -land_dz_ : 0.0;
        x_ref(0, 2) = (dz > 0) ? std::min(land_z_, x_(2) + dz) : std::max(land_z_, x_(2) + dz);

        // Reached landing height
        if (std::abs(x_(2) - land_z_) < land_thr_) {
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
            x_ref_.resize(0, 0);
            u_ref_.resize(0, 0);
            t_ref_.resize(0);  
            // TODO: Check if it gets cleared here correctly. its not accepting new trajectory
            x_initial_reached_ = false;
            mpc_idx_ = 0;
            landing_ = false;
            ref_received_ = false;
        }
        if (with_gp_) {
            gp_mpc_->setParams(mpc_param_);
        }
        return gp_mpc_->setReference(x_ref, u_ref);
    }

    // Check if reference trajectory is set up. If not, pick current position and keep hovering
    if (!ref_received_) {
        Eigen::MatrixXd x_ref(1, MPC_NX);
        Eigen::MatrixXd u_ref(1, MPC_NU);
        // Select current position as provisional hovering position
        if (x_ref_prov_.isZero()) {
            x_ref_prov_.head(N_POSITION_STATES + N_QUATERNION_STATES) = x_.segment(0, N_POSITION_STATES + N_QUATERNION_STATES);
            x_ref_prov_.tail(N_VELOCITY_STATES + N_RATE_STATES).setZero();
            ROS_INFO("Selecting current position as provisional setpoint.");
        }
        if (u_ref_prov_.isZero()) {
            u_ref_prov_.setZero();
        }
        // Fill x_ref with x_ref_prov_
        x_ref.row(0).head(MPC_NX) = x_ref_prov_;
        // Fill u_ref with u_ref_prov_
        u_ref.row(0).head(MPC_NU) = u_ref_prov_;

        if (with_gp_) {
            gp_mpc_->setParams(mpc_param_);
        }
        return gp_mpc_->setReference(x_ref, u_ref);
    }

    // If reference exists then exit out of provisional hovering mode
    if (!x_ref_prov_.isZero()) {
        x_ref_prov_.setZero();
        u_ref_prov_.setZero();
        ground_level_ = false;
        ROS_INFO("Abandoning provisional setpoint.");
    }

    // Check if starting position of trajectory has been reached
    if (!x_initial_reached_) {
        Eigen::MatrixXd x_ref = Eigen::MatrixXd::Zero(1, MPC_NX);
        Eigen::MatrixXd u_ref = Eigen::MatrixXd::Zero(1, MPC_NU);
        // Compute distance to initial position of trajectory
        Eigen::VectorXd x_ref_vec = x_ref_.row(0);
        // Create quaternion error from x and x_ref
        Eigen::Quaterniond q_ref(x_ref_vec(3), x_ref_vec(4), x_ref_vec(5), x_ref_vec(6));
        Eigen::Quaterniond q_error = q_ * q_ref.inverse();
        Eigen::VectorXd e(N_STATES - N_RATE_STATES - 1); // Adjust size based on actual dimensions
        // Fill e with concatenation of differences
        e << (x_.head(N_POSITION_STATES) - x_ref_vec.head(N_POSITION_STATES)),
              q_error.vec(),
             (x_.segment(IDX_VELOCITY_START, N_VELOCITY_STATES) - x_ref_vec.segment(IDX_VELOCITY_START, N_VELOCITY_STATES));
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
            x_ref.row(0) = x_ref_.row(0);
        } else {
            // Initial Position has not been reached yet 
            // Fly the drone toward initial position of trajectory
            double dx = (x_ref_(0, 0) > x_(0)) ? init_v_ : (x_ref_(0, 0) < x_(0)) ? -init_v_ : 0.0;
            double dy = (x_ref_(0, 1) > x_(1)) ? init_v_ : (x_ref_(0, 1) < x_(1)) ? -init_v_ : 0.0;
            double dz = (x_ref_(0, 2) > x_(2)) ? init_v_ : (x_ref_(0, 2) < x_(2)) ? -init_v_ : 0.0;
            // Set reference position in the direction of or at the init. position of trajectory
            x_ref(0, 0) = (dx > 0) ? std::min(x_ref_(0, 0), x_(0) + dx) : std::max(x_ref_(0, 0), x_(0) + dx);
            x_ref(0, 1) = (dy > 0) ? std::min(x_ref_(0, 1), x_(1) + dy) : std::max(x_ref_(0, 1), x_(1) + dy);
            x_ref(0, 2) = (dz > 0) ? std::min(x_ref_(0, 2), x_(2) + dz) : std::max(x_ref_(0, 2), x_(2) + dz);
            // Set angular position of drone to initial angular position of trajectory
            x_ref.row(0).segment(IDX_QUATERNION_START, N_QUATERNION_STATES) = x_ref_.row(0).segment(IDX_QUATERNION_START, N_QUATERNION_STATES);
        }
        if (with_gp_) {
            gp_mpc_->setParams(mpc_param_);
        }
        return gp_mpc_->setReference(x_ref, u_ref);
    }
    
    // Executing Trajectory Tracking
    if (mpc_idx_ < ref_len_) {
        Eigen::MatrixXd x_ref = Eigen::MatrixXd::Zero(MPC_N, MPC_NX);
        Eigen::MatrixXd u_ref = Eigen::MatrixXd::Zero(MPC_N, MPC_NU);
        
        // Downsample x_ref_ as it is given in 50Hz 
        // but the MPC horizon is modelled at different rate given by control_freq_factor
        // In this case 10Hz
        for (int i = 0; i < MPC_N; ++i) {
            // Use min to avoid index overflow
            int ii = std::min(i * control_freq_factor_ + mpc_idx_, ref_len_ - 1);
            // Copy the relevant row from x_ref_ and u_ref_ to x_ref and u_ref
            x_ref.row(i) = x_ref_.row(ii);
            u_ref.row(i) = u_ref_.row(ii);
        }
        if (with_gp_) {
            mpc_param_.row(0).head(MPC_NX) = x_;
            for (int i = 0; i < n_features_; ++i) {
                // torch::Tensor vel = torch::tensor(x_(x_features_[i]));
                gp_input_[0]  = torch::tensor(x_(x_features_[i])).unsqueeze(0);
                gp_corr_(0, i) = gp_model_[i].forward(gp_input_).toTensor().item<double>();
            }
            for (int i = 1; i < MPC_N; ++i) {
                int ii = std::min(i * control_freq_factor_ + mpc_idx_, ref_len_ - 1);
                gp_corr_.row(i) = gp_corr_ref_.row(ii);
            }
            mpc_param_.block(0, MPC_NX, MPC_N, n_features_) = gp_corr_;
            gp_mpc_->setParams(mpc_param_);
        }
        mpc_idx_++;
        return gp_mpc_->setReference(x_ref, u_ref);    

    } else if (mpc_idx_ == ref_len_) { 
        // End of reference reached
        Eigen::MatrixXd x_ref = Eigen::MatrixXd::Zero(1, MPC_NX);
        Eigen::MatrixXd u_ref = Eigen::MatrixXd::Zero(1, MPC_NU);
        // Compute Optimization dt
        optimization_dt_ /= mpc_idx_;
        ROS_INFO("Tracking complete. Mean MPC opt. time: %.3f ms", optimization_dt_*1000);

        // Lower drone to a safe height
        landing_ = true;
        ROS_INFO("Landing...");

        // Set reference to final linear and angualr position of trajectory
        x_ref.row(0).head(IDX_VELOCITY_START) = x_ref_.row(ref_len_ - 1).head(N_POSITION_STATES + N_QUATERNION_STATES);
        // Stop recording
        x_initial_reached_ = false;
        std_msgs::Bool msg;
        msg.data = false;
        record_pub_.publish(msg);
        if (with_gp_) {
            mpc_param_ = MatrixXd::Zero(MPC_N, MPC_NP);
            mpc_param_(0, MPC_NP-1) = 1; // TRIGGER VALUE
            gp_mpc_->setParams(mpc_param_);
        }
        return gp_mpc_->setReference(x_ref, u_ref);
    }
}

void Node::run() {
    ros::Rate rate(1);
    while (!ros::isShuttingDown()) {
        std_msgs::Bool msg;
        msg.data = !(!ref_received_ && x_available_);
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

