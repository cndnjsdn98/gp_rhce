

export PYTHONPATH=$PYTHONPATH:$HOME/GP_RHCE/catkin_ws/src/gp_rhce/

export PYTHONPATH=$PYTHONPATH:$HOME/catkin_ws/src/gp_rhce/

roslaunch gp_rhce traj_publisher.launch quad_name:=hummingbird env:=gazebo flight_mode:=lemniscate