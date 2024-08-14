/**
 * @file gp_mpc.cpp
 * @author Wnooo Choo
 * @date July 2024
 * 
 * @copyright
 * Copyright (C) 2024.
 */

#include <gp_mpc.h>

using namespace ros;
using namespace Eigen;

// Constructor
GP_MPC::GP_MPC() {

    acados_ocp_capsule_ = mpc_acados_create_capsule();
    // Create MPC Acados model
    status_ = mpc_acados_create(acados_ocp_capsule_);
    // If i want to create with new estimation window use this:
    // status_ = mpc_acados_create_with_discretization(acados_ocp_capsule_, N_, new_time_steps_);
    
    if (status_) {
       throw std::runtime_error("MPC_acados_create() failed with status: " + std::to_string(status_));
    }

    nlp_config_ = mpc_acados_get_nlp_config(acados_ocp_capsule_);
    nlp_dims_ = mpc_acados_get_nlp_dims(acados_ocp_capsule_);
    nlp_in_ = mpc_acados_get_nlp_in(acados_ocp_capsule_);
    nlp_out_ = mpc_acados_get_nlp_out(acados_ocp_capsule_);
    nlp_solver_ = mpc_acados_get_nlp_solver(acados_ocp_capsule_);

    // Initialize MPC parameters
    params_.resize(MPC_NP);
    x_init_.resize(MPC_NX);
    x_ref_.resize(MPC_NX + MPC_NU);
    xt_ref_.resize(MPC_NX);
    u0_.resize(MPC_NU);
}   

// Deconstructor
GP_MPC::~GP_MPC() {
    mpc_acados_free(acados_ocp_capsule_);
    mpc_acados_free_capsule(acados_ocp_capsule_);
}

int GP_MPC::solveMPC(const std::vector<double>& x_init, const std::vector<double>& u0) {
    x_init_ = x_init;
    u0_ = u0;
    // Set Initial conditions, equality constraint
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, 0, "lbx", static_cast<void*>(x_init_.data()));
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, 0, "ubx", static_cast<void*>(x_init_.data()));
    // solve
    status_ = mpc_acados_solve(acados_ocp_capsule_);
    return status_;
}

void GP_MPC::setReference(const std::vector<std::vector<double>>& x_ref, const std::vector<std::vector<double>>& u_ref) {
    // Check size of reference trajectory
    int ref_len = x_ref.size();
    // Set yref for given ref length
    for (int i = 0; i < ref_len-1; ++i) {
        std::copy(x_ref[i].begin(), x_ref[i].end(), x_ref_.begin());
        std::copy(u_ref[i].begin(), u_ref[i].end(), x_ref_.begin() + MPC_NX);
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", static_cast<void*>(x_ref_.data()));
    }
    // Set yref for remainder of estimation horizon with final element of the reference traj.
    std::copy(x_ref.back().begin(), x_ref.back().end(), x_ref_.begin());
    std::copy(u_ref.back().begin(), u_ref.back().end(), x_ref_.begin() + MPC_NX);
    for (int i = ref_len-1; i < MPC_N; ++i) {
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", static_cast<void*>(x_ref_.data()));
    }
    std::copy(x_ref.back().begin(), x_ref.back().end(), xt_ref_.begin());
    ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, MPC_N, "yref", static_cast<void*>(xt_ref_.data()));
    
}

void GP_MPC::setParams(const std::vector<std::vector<double>>& params) {
    for (int i = 0; i < MPC_N; i++) {
        std::copy(params[i].begin(), params[i].end(), params_.begin());
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, 0, "p", static_cast<void*>(params_.data()));
    }
}

void GP_MPC::getControls(std::vector<double>& x_opt, std::vector<double>& u_opt) {
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", &x_opt[0]);
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", &u_opt[0]);
}

double GP_MPC::getOptimizationTime() {
    ocp_nlp_get(nlp_config_, nlp_solver_, "time_tot", &optimization_dt_);
    return optimization_dt_;
}
