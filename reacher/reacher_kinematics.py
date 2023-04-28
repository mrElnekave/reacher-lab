import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
L1 = 0.08  # length of link 1
L2 = 0.11  # length of link 2
TOLERANCE = 0.01  # tolerance for inverse kinematics
PERTURBATION = 0.0001  # perturbation for finite difference method


def calculate_forward_kinematics_robot(joint_angles):
    """Calculate xyz coordinates of end-effector given joint angles.

    Use forward kinematics equations to calculate the xyz coordinates of the end-effector
    given some joint angles.

    Args:
      joint_angles: numpy array of 3 elements [TODO names]. Numpy array of 3 elements.
    Notes:
      joint_angles[1] is capped [-1/4pi, pi]
    Returns:
      xyz coordinates of the end-effector in the arm frame. Numpy array of 3 elements.
    """
    def generate_y_rotation(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

    def generate_z_rotation(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

    # if joint_angles[1] < 1/4 * np.pi:
    #     joint_angles[1] = 1/4 * np.pi

    joint_angles *= -1
    r1 = np.array([0, 0, L2])
    r2 = np.array([0, 0, L1]).transpose() + \
        np.dot(generate_y_rotation(joint_angles[2]), r1)
    r3 = np.array([0, -HIP_OFFSET, 0]).transpose() + \
        np.dot(generate_y_rotation(joint_angles[1]), r2)
    r4 = np.dot(generate_z_rotation(joint_angles[0]), r3)

    return r4


def ik_cost(end_effector_pos, guess):
    """Calculates the inverse kinematics loss.

    Calculate the Euclidean distance between the desired end-effector position and
    the end-effector position resulting from the given 'guess' joint angles.

    Args:
      end_effector_pos: desired xyz coordinates of end-effector. Numpy array of 3 elements.
      guess: guess at joint angles to achieve desired end-effector position. Numpy array of 3 elements.
    Returns:
      Euclidean distance between end_effector_pos and guess. Returns float.
    """
    # TODO for students: Implement this function. ~1-5 lines of code.
    cost = .5 * np.linalg.norm(
        calculate_forward_kinematics_robot(guess) - end_effector_pos) ** 2

    return cost


def partial_derivative_calc(angle_index, space_index, joint_angles):
    """
    Get the partial derivative of the forward kinematics function with respect to, the cartesian space index, and the joint angle index.

    Args:

    Returns:
        partial_div: float, with the space index relative to the angle index
    """
    delta = 0.0001
    joint_angles_copy = copy.deepcopy(joint_angles)
    joint_angles_copy[angle_index] += delta
    partial_div = (calculate_forward_kinematics_robot(joint_angles_copy)[space_index] - calculate_forward_kinematics_robot(
        joint_angles)[space_index]) / delta
    return partial_div


def calculate_jacobian(joint_angles) -> np.ndarray:
    """Calculate the jacobian of the end-effector position wrt joint angles.

    Calculate the jacobian, which is a matrix of the partial derivatives
    of the forward kinematics with respect to the joint angles 
    arranged as such:

    dx/dtheta1 dx/dtheta2 dx/dtheta3
    dy/dtheta1 dy/dtheta2 dy/dtheta3
    dz/dtheta1 dz/dtheta2 dz/dtheta3

    Args:
      joint_angles: joint angles of robot arm. Numpy array of 3 elements.

    Returns:
      Jacobian matrix. Numpy 3x3 array.
    """
    jacobian = np.zeros([3, 3])
    for cartesian in range(3):
        for angle in range(3):
            jacobian[cartesian][angle] = partial_derivative_calc(
                angle, cartesian, joint_angles)
    return jacobian


def calculate_inverse_kinematics(end_effector_pos, starting_joint_angles):
    """Calculates joint angles given desired xyz coordinates.
    Use gradient descent to minimize the inverse kinematics loss function. The
    joint angles that minimize the loss function are the joint angles that give 
    the smallest error from the actual resulting end-effector position to the
    desired end-effector position. You should use the jacobain function
    you wrote above.
    Args:
      end_effector_pos: Desired xyz coordinates of end-effector. Numpy array of 3 elements.
      starting_joint_angles: Where the robot starts. In terms of angles.
    Returns:
      Joint angles that correspond to given desired end-effector position. Numpy array with 3 elements.
      Returns None when IK times out because the end-effector position is infeasible.
    """
    # sphere cast limit for maximum distance

    # TODO for students: Implement this function. ~10-20 lines of code.
    joint_angles = starting_joint_angles

    current_cost = ik_cost(end_effector_pos, starting_joint_angles)
    epsilon = 10**(-2)

    loop_counter = 0
    max_loop = 1 << 7

    alpha = 1

    while current_cost > epsilon and loop_counter < max_loop:

        jacobian: np.ndarray = calculate_jacobian(joint_angles)
        current_dif = calculate_forward_kinematics_robot(
            joint_angles) - end_effector_pos

        gradient = np.matmul(jacobian.transpose(), current_dif)

        # step with - gradient for grad descent

        joint_angles += -1 * gradient * alpha

        current_cost = ik_cost(end_effector_pos, joint_angles)

        loop_counter += 1

    return joint_angles
