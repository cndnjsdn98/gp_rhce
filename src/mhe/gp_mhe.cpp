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
    // If i want to create with new estimation window use this:
    // status_ = mhe_acados_create_with_discretization(acados_ocp_capsule_, N_, new_time_steps_);
    
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
    gpy_corr_ = {0, 0, 0};
    if (mhe_type_ == "kinematic") {
        x0_bar_ = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.81};
    } else if (mhe_type_ == "dynamic") {
        x0_bar_ = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    }
    // x0_bar_.resize(MHE_NX);
    yref_0_.resize(NY0);
    yref_.resize(NY);
    u_.resize(NP);
}   

// Deconstructor
GP_MHE::~GP_MHE() {
    mhe_acados_free_capsule(acados_ocp_capsule_);
}

void GP_MHE::setHistory(const std::vector<std::vector<double>>& y_history, const std::vector<std::vector<double>>& u_history) {
    std::copy(y_history[0].begin(), y_history[0].end(), yref_0_.begin());
    std::fill(yref_0_.begin() + N_MEAS_STATES, yref_0_.begin() + N_MEAS_STATES + NU, 0);
    std::copy(x0_bar_.begin(), x0_bar_.end(), yref_0_.begin() + N_MEAS_STATES + NU);


    ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, 0, "yref", static_cast<void*>(yref_0_.data()));
    for (int i = 1; i < MHE_N; ++i) { 
        std::copy(y_history[i].begin(), y_history[i].end(), yref_.begin());
        std::fill(yref_.begin() + N_MEAS_STATES, yref_.begin() + N_MEAS_STATES + NU, 0);
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", static_cast<void*>(yref_.data()));
    }

    if (mhe_type_ == "dynamic") {
        for (int i = 0; i < MHE_N; ++i) { 
            std::copy(u_history[i].begin(), u_history[i].end(), u_.begin());
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "p", static_cast<void*>(u_.data()));
        }
    }
}

int GP_MHE::solveMHE(const std::vector<std::vector<double>>& y_history, const std::vector<std::vector<double>>& u_history) {
    // Set the history values
    setHistory(y_history, u_history);
    // Solve MHE
    // If successful retrieve results
    status_ = mhe_acados_solve(acados_ocp_capsule_);
    if (status_ != ACADOS_SUCCESS) {
        return status_;
    } 
    ocp_nlp_get(nlp_config_, nlp_solver_, "time_tot", &optimization_dt_);
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", &x0_bar_[0]);
    return status_;
}

void GP_MHE::getStateEst(std::vector<double>& x_est) {
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, MHE_N, "x", &x_est[0]);
    // for (const double& value : x_est) {
    //     std::cout << value << " ";
    // }
    // std::cout << std::endl;
}

double GP_MHE::getOptimizationTime() {
    return optimization_dt_;
}