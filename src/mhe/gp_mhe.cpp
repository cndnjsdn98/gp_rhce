/**
 * @file gp_mhe.cpp
 * @author Wnooo Choo
 * @date July 2024
 * 
 * @copyright
 * Copyright (C) 2024.
 */

#include <gp_mhe.h>

using namespace ros;
using namespace Eigen;

// Constructor
GP_MHE::GP_MHE(std::string& mhe_type) {

    acados_ocp_capsule_ = mhe_acados_create_capsule();
    // Create MHE Acados model
    status_ = mhe_acados_create(acados_ocp_capsule_);
    
    if (status_) {
        throw std::runtime_error("MHE_acados_create() failed with status: " + std::to_string(status_));
    }
    
    nlp_config_ = mhe_acados_get_nlp_config(acados_ocp_capsule_);
    nlp_dims_ = mhe_acados_get_nlp_dims(acados_ocp_capsule_);
    nlp_in_ = mhe_acados_get_nlp_in(acados_ocp_capsule_);
    nlp_out_ = mhe_acados_get_nlp_out(acados_ocp_capsule_);
    nlp_solver_ = mhe_acados_get_nlp_solver(acados_ocp_capsule_);

    // Initialize MHE parameters
    mhe_type_ = mhe_type;
    // gpy_corr_.setZero();
    if (mhe_type_ == "kinematic") {
        x0_bar_ << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.81;
        // x0_bar_[IDX_Q_W] = 1;
        // x0_bar_[IDX_A_Z] = 9.81;
        n_meas_states_ = N_POSITION_STATES + N_RATE_STATES + N_ACCEL_STATES;
    } else if (mhe_type_ == "dynamic") {
        x0_bar_ << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        // x0_bar_[IDX_Q_W] = 1;
        n_meas_states_ = N_POSITION_STATES + N_RATE_STATES;
    }

    for (int i = 0; i < MHE_N; ++i ) {
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, i, "x", &x0_bar_);  
    }

    yref_0_.setZero();
    yref_.setZero();
    u_.setZero();

    // std::fill_n(yref_0_, NY0, 0.0);
    // std::fill_n(yref_, NY, 0);
    // std::fill_n(u_, NP, 0);
}   

// Deconstructor
GP_MHE::~GP_MHE() {
    mhe_acados_free_capsule(acados_ocp_capsule_);
}

void GP_MHE::setHistory(const Eigen::MatrixXd& y_history, const Eigen::MatrixXd& u_history) {
    // yref_0_ = y_history.row(0);
    // yref_0_.segment(n_meas_states_ + MHE_NU, MHE_NX) = x0_bar_;

    double yref_0[NY0] = {};    
    if (mhe_type_ == "kinematic") {
        Eigen::Map<Eigen::Matrix<double, N_KIN_MEAS_STATES, 1>> (&yref_0[0], N_KIN_MEAS_STATES, 1) = y_history.row(0);
        Eigen::Map<Eigen::Matrix<double, NX, 1>> (&yref_0[N_KIN_MEAS_STATES+MHE_NU], NX, 1) = x0_bar_;
    } else {
        Eigen::Map<Eigen::Matrix<double, N_DYN_MEAS_STATES, 1>> (&yref_0[0], N_DYN_MEAS_STATES, 1) = y_history.row(0);
        Eigen::Map<Eigen::Matrix<double, NX, 1>> (&yref_0[N_DYN_MEAS_STATES+MHE_NU], NX, 1) = x0_bar_;
    }

    ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, 0, "yref", &yref_0);
    
    for (int i = 1; i < MHE_N; ++i) { 
        // yref_ = y_history.row(i);

        double yref[NY] = {};
        std::fill_n(yref, NY, 0);
        if (mhe_type_ == "kinematic") {
            Eigen::Map<Eigen::Matrix<double, N_KIN_MEAS_STATES, 1>> (&yref[0], N_KIN_MEAS_STATES, 1) = y_history.row(i);
        } else {
            Eigen::Map<Eigen::Matrix<double, N_DYN_MEAS_STATES, 1>> (&yref[0], N_DYN_MEAS_STATES, 1) = y_history.row(i);
        }
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", &yref);
    }

    if (mhe_type_ == "dynamic") {
        for (int i = 0; i < MHE_N; ++i) { 
            // u_ = u_history.row(i);

            double u[NP] = {};
            Eigen::Map<Eigen::Matrix<double, NP, 1>> (&u[0], NP, 1) = u_history.row(i);
            // for (int i = 0; i < 4; ++i) {
            //     std::cout << u[i] * 8.7 / 0.795 << " ";
            // }
            // std::cout << std::endl;
            // for (int i = 0; i < 4; ++i) {
            //     std::cout << u[i] << " ";
            // }
            // std::cout << std::endl;
            mhe_acados_update_params(acados_ocp_capsule_, i, u, NP);
        }
    }
}

int GP_MHE::solveMHE(const Eigen::MatrixXd& y_history, const Eigen::MatrixXd& u_history) {
    // Set the history values
    setHistory(y_history, u_history);
    // Solve MHE
    // If successful retrieve results
    status_ = mhe_acados_solve(acados_ocp_capsule_);
    if (status_ != ACADOS_SUCCESS) {
        return status_;
    } 

    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", &x0_bar_[0]);
    return status_;
}

void GP_MHE::getStateEst(Eigen::VectorXd& x_est) {
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, MHE_N, "x", &x_est[0]);
}

double GP_MHE::getOptimizationTime() {
    ocp_nlp_get(nlp_config_, nlp_solver_, "time_tot", &optimization_dt_);
    return optimization_dt_;
}

int GP_MHE::getMeasStateLen() {
    return n_meas_states_;
}

