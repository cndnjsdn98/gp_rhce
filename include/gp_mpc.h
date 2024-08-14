#ifndef GP_MPC_H
#define _GP_MPC_H
// ros
#include <ros/ros.h>
#include <mav_msgs/Actuators.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

// acados
#include <mpc/acados_solver_mpc.h>

#define NX     MPC_NX
#define NZ     MPC_NZ
#define NU     MPC_NU
#define NP     MPC_NP
#define NBX    MPC_NBX
#define NBX0   MPC_NBX0
#define NBU    MPC_NBU
#define NSBX   MPC_NSBX
#define NSBU   MPC_NSBU
#define NSH    MPC_NSH
#define NSG    MPC_NSG
#define NSPHI  MPC_NSPHI
#define NSHN   MPC_NSHN
#define NSGN   MPC_NSGN
#define NSPHIN MPC_NSPHIN
#define NSBXN  MPC_NSBXN
#define NS     MPC_NS
#define NSN    MPC_NSN
#define NG     MPC_NG
#define NBXN   MPC_NBXN
#define NGN    MPC_NGN
#define NY0    MPC_NY0
#define NY     MPC_NY
#define NYN    MPC_NYN
#define NH     MPC_NH
#define NPHI   MPC_NPHI
#define NHN    MPC_NHN
#define NPHIN  MPC_NPHIN
#define NR     MPC_NR

#define N_POSITION_STATES 3
#define N_QUATERNION_STATES 4
#define N_VELOCITY_STATES 3
#define N_RATE_STATES 3
#define N_STATES 13

#define IDX_POSITION_START 0
#define IDX_QUATERNION_START N_POSITION_STATES
#define IDX_VELOCITY_START N_POSITION_STATES + N_QUATERNION_STATES
#define IDX_RATE_START N_POSITION_STATES + N_QUATERNION_STATES + N_VELOCITY_STATES
#define IDX_RATE_X IDX_RATE_START
#define IDX_RATE_Y IDX_RATE_X + 1
#define IDX_RATE_Z IDX_RATE_Y + 1

class GP_MPC {
private:
    // acados_ocp_capsule
    mpc_solver_capsule *acados_ocp_capsule_;

    // Acados MPC Solver status
    int status_;

    // Acados
    double* new_time_steps_ = nullptr;
    ocp_nlp_config *nlp_config_;
    ocp_nlp_dims *nlp_dims_;
    ocp_nlp_in *nlp_in_;
    ocp_nlp_out *nlp_out_;
    ocp_nlp_solver *nlp_solver_;
    void *nlp_opts_;

    // Initial condition
    int idxbx0_[NBX0];
    // Constraints
    double lbx0_[NBX0];
    double ubx0_[NBX0];
    
    // MPC Parameters
    std::vector<double> x_init_, u0_, params_, xt_ref_, x_ref_;
    double optimization_dt_;

public:
    // Constructor
    GP_MPC();
    
    // Deconstructor
    ~GP_MPC();

    int solveMPC(const std::vector<double>& x_init, const std::vector<double>& u0);
    void setReference(const std::vector<std::vector<double>>& x_ref, const std::vector<std::vector<double>>& u_ref);
    void setParams(const std::vector<std::vector<double>>& params);
    void getControls(std::vector<double>& x_opt, std::vector<double>& u_opt);
    double getOptimizationTime();
};

#endif