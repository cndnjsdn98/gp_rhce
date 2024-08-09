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
        std::cout << "MPC_acados_create() returned status %d. Exiting.\n", status_; 
        exit(1);
    }

    nlp_config_ = mpc_acados_get_nlp_config(acados_ocp_capsule_);
    nlp_dims_ = mpc_acados_get_nlp_dims(acados_ocp_capsule_);
    nlp_in_ = mpc_acados_get_nlp_in(acados_ocp_capsule_);
    nlp_out_ = mpc_acados_get_nlp_out(acados_ocp_capsule_);
    nlp_solver_ = mpc_acados_get_nlp_solver(acados_ocp_capsule_);

    // Initialize MPC parameters
    x_init_.reserve(MPC_NX);
    u0_.reserve(MPC_NU);
    x_ref_.reserve(MPC_NY);
    xt_ref_.reserve(MPC_NX);
}   

// Deconstructor
GP_MPC::~GP_MPC() {
    mpc_acados_free(acados_ocp_capsule_);
    mpc_acados_free_capsule(acados_ocp_capsule_);
}

int GP_MPC::solveMPC(const std::vector<double>& x_init, const std::vector<double>& u0) {
    setInitialStates(x_init, u0);
    status_ = mpc_acados_solve(acados_ocp_capsule_);
    return status_;
}

void GP_MPC::setInitialStates(const std::vector<double>& x_init, const std::vector<double>& u0) {
    x_init_.clear();
    x_init_.insert(x_init_.end(), x_init.begin(), x_init.end());
    u0_.clear();
    u0_.insert(u0_.end(), u0.begin(), u0.end());
    // x_init_ = x_init;
    // u0_ = u0;
    // Set Initial conditions, equality constraint
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, 0, "lbx", static_cast<void*>(x_init_.data()));
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, 0, "ubx", static_cast<void*>(x_init_.data()));
    // ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, 0, "lbu", static_cast<void*>(u0_.data()));
    // ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, 0, "ubu", static_cast<void*>(u0_.data()));
    if (verbose_) {
        std::cout << "\n--- Set Init ---\n";
        for (const double& value : x_init_) {
            std::cout << value << ", ";
        }
        std::cout<<std::endl;
    }
    // Set initial guesses
    for (int i = 0; i < MPC_N; i++) {
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, i, "x", static_cast<void*>(x_init_.data()));
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, i, "u", static_cast<void*>(u0_.data()));
    }
    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, MPC_N, "x", static_cast<void*>(x_init_.data()));
}

void GP_MPC::setReference(const std::vector<std::vector<double>>& x_ref, const std::vector<std::vector<double>>& u_ref) {
    // Check size of reference trajectory
    int ref_len = x_ref.size();
    if (verbose_) {
        std::cout << "\n--- Set Reference ---\n";
    }
    for (int i = 0; i < ref_len-1; ++i) {
        x_ref_.clear();
        x_ref_.insert(x_ref_.end(), x_ref[i].begin(), x_ref[i].end());
        x_ref_.insert(x_ref_.end(), u_ref[i].begin(), u_ref[i].end());
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", static_cast<void*>(x_ref_.data()));
        if (verbose_) {
            for (const double& value : x_ref_) {
                std::cout << value << ", ";
            }
            std::cout << i << std::endl;
        }
    }
    x_ref_.clear();
    x_ref_.insert(x_ref_.end(), x_ref.back().begin(), x_ref.back().end());
    x_ref_.insert(x_ref_.end(), u_ref.back().begin(), u_ref.back().end());
    for (int i = ref_len-1; i < MPC_N; ++i) {
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", static_cast<void*>(x_ref_.data()));
        if (verbose_) {
            for (const double& value : x_ref_) {
                std::cout << value << ", ";
            }
            std::cout<< i << std::endl;
        }
    }
    xt_ref_.clear();
    xt_ref_.insert(xt_ref_.end(), x_ref.back().begin(), x_ref.back().end());
    if (verbose_) {
        for (const double& value : xt_ref_) {
            std::cout << value << ", ";
        }
        std::cout<< MPC_N << std::endl;
    }
    ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, MPC_N, "yref", static_cast<void*>(xt_ref_.data()));
    
}

void GP_MPC::setParams(const std::vector<std::vector<double>>& params) {
    for (int i = 0; i < MPC_N; i++) {
        params_.clear();
        params_.insert(x_init_.end(), params[i].begin(), params[i].end());
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, 0, "p", static_cast<void*>(params_.data()));
    }
}

void GP_MPC::getControls(std::vector<double>& x_opt, std::vector<double>& u_opt) {
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", &x_opt[0]);
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", &u_opt[0]);
    if (verbose_) {
        std::cout << "\n--- u opt ---" << std::endl;
        for (const double& value : u_opt) { 
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

double GP_MPC::getOptimizationTime() {
    ocp_nlp_get(nlp_config_, nlp_solver_, "time_tot", &optimization_dt_);
    return optimization_dt_;
}
