from reacher import reacher_kinematics
from reacher import reacher_robot_utils
from reacher import reacher_sim_utils
import pybullet as p
import time
import numpy as np
from absl import app
from absl import flags
from pupper_hardware_interface import interface
from sys import platform

KP = 5.0  # Amps/rad
KD = 2.0  # Amps/(rad/s)
MAX_CURRENT = 3.0  # Amps

UPDATE_DT = 0.01  # seconds

HIP_OFFSET = 0.0335  # meters
L1 = 0.08  # meters
L2 = 0.11  # meters


def main(argv):
    reacher = reacher_sim_utils.load_reacher()
    desired_position = reacher_sim_utils.create_debug_sphere()
    current_position = reacher_sim_utils.create_debug_sphere()

    joint_ids = reacher_sim_utils.get_joint_ids(reacher)
    param_ids = reacher_sim_utils.get_param_ids(reacher)
    reacher_sim_utils.zero_damping(reacher)

    p.setPhysicsEngineParameter(numSolverIterations=10)

    p.setRealTimeSimulation(1)
    counter = 0
    last_command = time.time()
    joint_angles = np.zeros(6)

    # Use this function to disable/enable certain motors. The first six elements
    # determine activation of the motors attached to the front of the PCB, which
    # are not used in this lab. The last six elements correspond to the activations
    # of the motors attached to the back of the PCB, which you are using.
    # The 7th element will correspond to the motor with ID=1, 8th element ID=2, etc
    # hardware_interface.send_dict({"activations": [0, 0, 0, 0, 0, 0, x, x, x, x, x, x]})

    RUN_AS_ROBOT_2 = True

    while (1):
        if time.time() - last_command > UPDATE_DT:
            last_command = time.time()
            counter += 1

            # Get joint angles from simulation
            slider_angles = np.zeros_like(joint_angles)
            for i in range(len(param_ids)):
                c = param_ids[i]
                targetPos = p.readUserDebugParameter(c)
                slider_angles[i] = targetPos

            # Get xyz from simulation
            xyz = []
            if RUN_AS_ROBOT_2:
                xyz = reacher_kinematics.calculate_forward_kinematics_robot(
                    slider_angles[3:])
            else:
                for i in range(len(param_ids), len(param_ids) + 3):
                    xyz.append(p.readUserDebugParameter(i))

            xyz = np.asarray(xyz)

            # Calculate what the joint angles should be using inverse kinematics
            ret = reacher_kinematics.calculate_inverse_kinematics(
                xyz, np.asarray([0.0, 0.0, 0.0]))

            # Wraps angles between -pi, pi
            joint_angles[:3] = np.arctan2(np.sin(ret), np.cos(ret))
            joint_angles[3:] = slider_angles[3:]

            for i in range(len(joint_ids)):
                p.setJointMotorControl2(reacher,
                                        joint_ids[i],
                                        p.POSITION_CONTROL,
                                        joint_angles[i],
                                        force=2.)

            # Update the position of the debug sphere

            end_effector_pos = xyz  # added for debugging
            p.resetBasePositionAndOrientation(desired_position,
                                              posObj=end_effector_pos,
                                              ornObj=[0, 0, 0, 1])

            end_effector_pos = reacher_kinematics.calculate_forward_kinematics_robot(
                joint_angles[:3])
            p.resetBasePositionAndOrientation(current_position,
                                              posObj=end_effector_pos,
                                              ornObj=[0, 0, 0, 1])


app.run(main)
