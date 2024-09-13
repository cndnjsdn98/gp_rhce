#ifndef GP_MHE_H
#define _GP_MHE_H
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

// acados
#include <mhe/acados_solver_mhe.h>

#define NX     MHE_NX
#define NZ     MHE_NZ
#define NU     MHE_NU
#define NP     MHE_NP
#define NBX    MHE_NBX
#define NBX0   MHE_NBX0
#define NBU    MHE_NBU
#define NSBX   MHE_NSBX
#define NSBU   MHE_NSBU
#define NSH    MHE_NSH
#define NSG    MHE_NSG
#define NSPHI  MHE_NSPHI
#define NSHN   MHE_NSHN
#define NSGN   MHE_NSGN
#define NSPHIN MHE_NSPHIN
#define NSBXN  MHE_NSBXN
#define NS     MHE_NS
#define NSN    MHE_NSN
#define NG     MHE_NG
#define NBXN   MHE_NBXN
#define NGN    MHE_NGN
#define NY0    MHE_NY0
#define NY     MHE_NY
#define NYN    MHE_NYN
#define NH     MHE_NH
#define NPHI   MHE_NPHI
#define NHN    MHE_NHN
#define NPHIN  MHE_NPHIN
#define NR     MHE_NR

#define N_POSITION_STATES 3
#define N_QUATERNION_STATES 4
#define N_VELOCITY_STATES 3
#define N_RATE_STATES 3
#define N_ACCEL_STATES 3
#define N_MOTORS 4

#define IDX_Q_W 3

#define IDX_POSITION_START 0
#define IDX_RATE_START N_POSITION_STATES
#define IDX_ACCEL_START N_POSITION_STATES + N_RATE_STATES

class GP_MHE {
private:
    // acados_ocp_capsule
    mhe_solver_capsule *acados_ocp_capsule_;

    // Acados MHE Solver status
    int status_;
    int n_meas_states_;

    // Acados
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
    
    // MHE Parameters
    std::string mhe_type_;
    std::vector<double> yref_0_, yref_, gpy_corr_, x0_bar_, u_;
    double optimization_dt_;

public:
    // Constructor
    GP_MHE(std::string& mhe_type);
    
    // Deconstructor
    ~GP_MHE();

    int solveMHE(const std::vector<std::vector<double>>& y_history, const std::vector<std::vector<double>>& u_history);
    void setHistory(const std::vector<std::vector<double>>& y_history, const std::vector<std::vector<double>>& u_history);
    void getStateEst(std::vector<double>& x_est);
    double getOptimizationTime();
    int getMeasStateLen();
};

#endif