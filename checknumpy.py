import numpy as np
HIP_OFFSET = 0.05  # length of N
L1 = 0.08  # length of link 1, the hip to knee
L2 = 0.11  # length of link 2, the knee to foot


def calculate_forward_kinematics_robot(joint_angles):
    """Calculate xyz coordinates of end-effector given joint angles.

    Use forward kinematics equations to calculate the xyz coordinates of the end-effector
    given some joint angles.

    Args:
      joint_angles: numpy array of 3 elements [TODO names]. Numpy array of 3 elements.
    Returns:
      xyz coordinates of the end-effector in the arm frame. Numpy array of 3 elements.
    """
    def generate_y_rotation(theta):
        return np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])

    def generate_z_rotation(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

    r1 = np.array([0, 0, L2])
    r2 = np.array([0, 0, L1]).transpose() + \
        np.dot(generate_y_rotation(joint_angles[0]), r1)
    r3 = np.array([0, HIP_OFFSET, 0]).transpose() + \
        np.dot(generate_y_rotation(joint_angles[1]), r2)
    r4 = np.dot(generate_z_rotation(joint_angles[2]), r3)


calculate_forward_kinematics_robot([0, 0, 0])
