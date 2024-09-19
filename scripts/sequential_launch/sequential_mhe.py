#!/usr/bin/env python

import subprocess
import rospy
import time
def main():
    rospy.init_node('sequential_mhe', anonymous=True)

    compile = rospy.get_param("/compile", default=True)

    # Launch the Python node
    rospy.loginfo("Launching Python node to retrieve Quadrotor Params...")
    model_gen_node = subprocess.Popen(['rosrun', 'gp_rhce', 'quad_optimizer_mhe.py'])
    
    # Wait for the Python node to complete
    model_gen_node.wait()
    if model_gen_node.returncode == 0:
        rospy.loginfo("Launching gp_mhe node...")
    else:
        rospy.logerr("Python node failed with return code %d", model_gen_node.returncode)
        return  
    
    time.sleep(3)

    # Launch the C++ MHE node
    mhe_node = subprocess.Popen(['rosrun', 'gp_rhce', 'gp_mhe_node'])

    # Wait for the C++ node to complete (optional)
    mhe_node.wait()
    rospy.loginfo("Closing MHE Node")

if __name__ == '__main__':
    main()