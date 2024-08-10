#!/usr/bin/env python

import subprocess
import rospy
import time
def main():
    rospy.init_node('sequential_mpc', anonymous=True)

    compile = rospy.get_param("/compile", default=True)

    if compile:
        # Launch the Python node
        rospy.loginfo("Launching Python node to generate MPC Acados model...")
        model_gen_node = subprocess.Popen(['rosrun', 'gp_rhce', 'quad_optimizer_mpc.py'])
        
        # Wait for the Python node to complete
        model_gen_node.wait()
        if model_gen_node.returncode == 0:
            rospy.loginfo("MPC Model Compilation completed successfully. Launching gp_mpc node...")
        else:
            rospy.logerr("MP Model Compilation failed with return code %d", model_gen_node.returncode)
            return

    time.sleep(3)

    # Launch the C++ MPC node
    mpc_node = subprocess.Popen(['rosrun', 'gp_rhce', 'gp_mpc_node'])

    # Wait for the C++ node to complete (optional)
    mpc_node.wait()
    rospy.loginfo("Closing MPC Node")

if __name__ == '__main__':
    main()