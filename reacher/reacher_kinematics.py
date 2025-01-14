import numpy as np


HIP_OFFSET = 0.0335
L1 = 0.08  # length of link 1
L2 = 0.11  # length of link 2

# Variables for gradient descent

# tolerance for inverse kinematics, needs to be really close to the actual position
TOLERANCE = 2e-5
# perturbation for finite difference method
PERTURBATION = 0.0001
# step size for gradient descent
STEP_SIZE = 20
# maximum number of iterations of gradient descent
MAX_ITERATIONS = 1000


def _generate_y_rotation(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def _generate_z_rotation(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


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
    # clockwise rotation to counter clockwise rotation
    angle_1, angle_2, angle_3 = joint_angles * -1
    r_knee_to_foot = np.array([0, 0, L2])
    r_hip_to_knee = np.array([0, 0, L1]).transpose() + \
        np.matmul(_generate_y_rotation(angle_3), r_knee_to_foot)
    r_pelvis_to_foot = np.array([0, -HIP_OFFSET, 0]).transpose() + \
        np.matmul(_generate_y_rotation(angle_2), r_hip_to_knee)
    r_base_to_foot = np.matmul(_generate_z_rotation(angle_1), r_pelvis_to_foot)

    return r_base_to_foot


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


def partial_derivative_calc(angle_index, joint_angles) -> np.ndarray(3):
    """
    Get the partial derivative of the forward kinematics function with respect to the joint angle index.
    Does all space indices at once

    Args:
        angle_index: int, the index of the joint angle
        joint_angles: numpy array of 3 elements. Numpy array of 3 elements.
    Returns:
        partial_div: float, with the the angle index.
    """
    offset = np.zeros(3)
    offset[angle_index] = PERTURBATION
    partial_div = (calculate_forward_kinematics_robot(joint_angles + offset) -
                   calculate_forward_kinematics_robot(joint_angles)) / PERTURBATION

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
    jacobian = np.zeros((3, 3))
    # traverse the cartesian space and the joint angles
    for angle_index in range(3):
        # set the column of the jacobian to the partial derivative
        jacobian[:, angle_index] = partial_derivative_calc(
            angle_index, joint_angles)

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
    max_reach_of_arm = L1 + L2
    if np.linalg.norm(end_effector_pos) > max_reach_of_arm:
        end_effector_pos = end_effector_pos / \
            np.linalg.norm(end_effector_pos) * max_reach_of_arm

    # initialize variables
    joint_angles = starting_joint_angles
    iteration = 0

    print("Original cost is: ", ik_cost(end_effector_pos, joint_angles))
    # while the cost is greater than the tolerance and the iteration is less than the max iterations
    while ik_cost(end_effector_pos, joint_angles) > TOLERANCE and iteration < MAX_ITERATIONS:
        # calculate the jacobian
        jacobian = calculate_jacobian(joint_angles)

        # distance to the goal, is also the gradient, as its the direction to the goal
        gradient_xy = calculate_forward_kinematics_robot(
            joint_angles) - end_effector_pos

        # calculate the gradient
        gradient = np.matmul(jacobian.transpose(), gradient_xy)
        # update the joint angles
        joint_angles -= STEP_SIZE * gradient
        iteration += 1

    # return the joint angles
    return joint_angles
