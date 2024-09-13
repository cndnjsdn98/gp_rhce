

export PYTHONPATH=$PYTHONPATH:$HOME/GP_RHCE/catkin_ws/src/gp_rhce/

export PYTHONPATH=$PYTHONPATH:$HOME/catkin_ws/src/gp_rhce/

roslaunch gp_rhce traj_publisher.launch quad_name:=hummingbird env:=gazebo flight_mode:=lemniscate

on python dynamic:
[INFO] [1726004967.173568, 4923.290000]: MHE Recording complete. Mean MHE opt. time 4.620 ms
[INFO] [1726004967.179865, 4923.290000]: MHE Filling in dataset and saving...
100%|███████████████████████████████████████████████████████████████████| 5202/5202 [00:00<00:00, 5308.80it/s]
x_est   (5202, 13)
x_act   (5202, 13)
sensor_meas   (5202, 9)
gpy_est   (5202, 1)
timestamp   (5202,)
rotor_speed   (5202, 4)
mhe_error   (5202, 13)
a_est_b   (5202, 3)
mass_est   (5202,)
mass_act   (5202,)
[INFO] [1726004973.029476, 4926.210000]: MHE Saving recorded results complete.
[INFO] [1726004975.975454, 4927.690000]: MPC Filling in dataset and saving...
100%|███████████████████████████████████████████████████████████████████| 2600/2600 [00:01<00:00, 1645.96it/s]
state_in   (2600, 13)
state_ref   (2600, 13)
error   (2600, 13)
input_in   (2600, 4)
state_out   (2600, 13)
state_pred   (2600, 13)
timestamp   (2600,)
dt   (2600,)
error_pred   (2600, 3)
[INFO] [1726004977.677948, 4928.540000]: Tracking complete Total Control RMSE: 0.21244 m Max vel: 7.954 m/s Mean MPC opt. time: 1.031 ms

on python kinematic:
[INFO] [1726077492.868007, 2936.480000]: MHE Recording complete. Mean MHE opt. time 7.180 ms
[INFO] [1726077492.870626, 2936.480000]: MHE Filling in dataset and saving...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5202/5202 [00:00<00:00, 5719.79it/s]
x_est   (5202, 16)
x_act   (5202, 13)
sensor_meas   (5202, 9)
gpy_est   (5202, 1)
timestamp   (5202,)
rotor_speed   (5202, 4)
mhe_error   (5202, 13)
a_est_b   (5202, 3)
imu_est   (5202, 3)
mass_est   (5202,)
mass_act   (5202,)
[INFO] [1726077494.016051, 2937.050000]: MHE Saving recorded results complete.
[INFO] [1726077501.671480, 2940.880000]: MPC Filling in dataset and saving...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2600/2600 [00:01<00:00, 1664.69it/s]
state_in   (2600, 13)
state_ref   (2600, 13)
error   (2600, 13)
input_in   (2600, 4)
state_out   (2600, 13)
state_pred   (2600, 13)
timestamp   (2600,)
dt   (2600,)
error_pred   (2600, 3)
[INFO] [1726077503.349512, 2941.710000]: Tracking complete Total Control RMSE: 0.21250 m Max vel: 7.951 m/s Mean MPC opt. time: 0.897 ms
[INFO] [1726077503.371075, 2941.730000]: Entering provisional hovering mode while reference is available at: 
