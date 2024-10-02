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
    if (mhe_type_ == "kinematic") {
        x0_bar_ << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.81;
        n_meas_states_ = N_POSITION_STATES + N_RATE_STATES + N_ACCEL_STATES;
    } else if (mhe_type_ == "dynamic") {
        x0_bar_ << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        n_meas_states_ = N_POSITION_STATES + N_RATE_STATES;
    }

    for (int i = 0; i < MHE_N; ++i ) {
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, i, "x", &x0_bar_);  
    }

    yref_0_.setZero();
    yref_.setZero();
    u_.setZero();
}   

// Deconstructor
GP_MHE::~GP_MHE() {
    mhe_acados_free_capsule(acados_ocp_capsule_);
}

void GP_MHE::setHistory(const Eigen::MatrixXd& y_history, const Eigen::MatrixXd& u_history) {
    yref_0_.head(n_meas_states_) = y_history.row(0);
    yref_0_.segment(n_meas_states_ + MHE_NU, NX) = x0_bar_;
    // ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, 0, "yref", static_cast<void*>(yref_0_.data()));
    ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, 0, "yref", &yref_0_);

    for (int i = 1; i < MHE_N; ++i) { 
        yref_.head(n_meas_states_) = y_history.row(i);
        // ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", static_cast<void*>(yref_.data()));
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", &yref_);
    }

    if (mhe_type_ == "dynamic") {
        for (int i = 0; i < MHE_N; ++i) { 
            u_ = u_history.row(i);
            mhe_acados_update_params(acados_ocp_capsule_, i, u_.data(), NP);
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

    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", &x0_bar_(0));
    return status_;
}

void GP_MHE::getStateEst(Eigen::VectorXd& x_est) {
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, MHE_N, "x", &x_est(0));
}

double GP_MHE::getOptimizationTime() {
    ocp_nlp_get(nlp_config_, nlp_solver_, "time_tot", &optimization_dt_);
    return optimization_dt_;
}

int GP_MHE::getMeasStateLen() {
    return n_meas_states_;
}

