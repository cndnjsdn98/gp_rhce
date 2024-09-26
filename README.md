
# Download libtorch
save libtorch file in include/gp
Can be anywhere honestly, just make sure to change the directory pointing to libtorch in cmakelist.txt if you do

# Install cuda
https://developer.nvidia.com/cuda-downloads

## Set the CMAKE_CUDA_COMPILER Path
Once CUDA is installed, locate the nvcc compiler, which is typically found in the following directory:
/usr/local/cuda/bin/nvcc

## Ensure Consistent CUDA Environment
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH


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
