

export PYTHONPATH=$PYTHONPATH:$HOME/GP_RHCE/catkin_ws/src/gp_rhce/

export PYTHONPATH=$PYTHONPATH:$HOME/catkin_ws/src/gp_rhce/

roslaunch gp_rhce traj_publisher.launch quad_name:=hummingbird env:=gazebo flight_mode:=lemniscate

on python dynamic:
[INFO] [1726440866.201549, 444.800000]: MHE Recording complete. Mean MHE opt. time 1.624 ms

on python kinematic:
[INFO] [1726440454.614764, 239.060000]: MHE Recording complete. Mean MHE opt. time 2.665 ms

on c++ dynamic:
[ INFO] [1726444354.371630581, 2188.330000000]: Estimation complete. Mean MHE opt. time: 3.250 ms

on c++ kinematic:
[ INFO] [1726443987.603339709, 2005.000000000]: Estimation complete. Mean MHE opt. time: 3.762 ms
