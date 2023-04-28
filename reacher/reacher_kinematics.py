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
    Returns:
      xyz coordinates of the end-effector in the arm frame. Numpy array of 3 elements.
    """
    # TODO for students: Implement this function. ~25-35 lines of code.
    end_effector_xyz = np.array([0.0, 0.0, 0.0])

    return end_effector_xyz


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
    cost = 0.0
    raise cost

# write a function that uses newtons method to update the joint angles


def newtons(guess, gradient):
    """Update joint angles using a single step of Newton's method.

    Args:
      guess: guess at joint angles to achieve desired end-effector position. Numpy array of 3 elements.
      gradient: gradient of the loss function with respect to the joint angles. Numpy array of 3 elements.
    Returns:
      updated_guess. Numpy array of 3 elements.
    """
    # TODO for students: Implement this function. ~1-5 lines of code.
    updated_guess = np.array([0.0, 0.0, 0.0])
    return updated_guess

# write a function that uses finite difference method to compute the gradient of the loss function


def finite_difference(end_effector_pos, guess):
    """Calculate the gradient of the loss function using finite difference method.

    1. Calculate the loss function at the given guess
    2. Calculate the loss function at a guess that is slightly perturbed in each direction by PERTURBATION (defined above)
    3. Calculate the gradient using the finite difference method

    Args:
      end_effector_pos: desired xyz coordinates of end-effector. Numpy array of 3 elements.
      guess: guess at joint angles to achieve desired end-effector position. Numpy array of 3 elements.
    Returns:
      gradient of the loss function with respect to the joint angles. Numpy array of 3 elements.
    """
    # TODO for students: Implement this function. ~5-10 lines of code.
    gradient = np.array([0.0, 0.0, 0.0])
    return gradient


def calculate_inverse_kinematics(end_effector_pos, guess):
    """Calculates joint angles given desired xyz coordinates.

    Use gradient descent to minimize the inverse kinematics loss function. The
    joint angles that minimize the loss function are the joint angles that give 
    the smallest error from the actual resulting end-effector position to the
    desired end-effector position. 

    1. Calculate the gradient of the loss function with respect to the joint angles using finite 
    difference method
    2. Use Newton's method to update the joint angles
    3. Repeat steps 1 and 2 until the loss is below TOLERANCE (defined above)

    Args:
      end_effector_pos: Desired xyz coordinates of end-effector. Numpy array of 3 elements.
      guess: Guess at joint angles that achieve desired end-effector position. Numpy array of 3 elements.
      This is our previous ik estimate of joint angles
    Returns:
      Joint angles that correspond to given desired end-effector position. Numpy array with 3 elements.
      Returns None when IK times out because the end-effector position is infeasible.
    """
    # TODO: write a loop to iteratively update the joint angles until the loss is below TOLERANCE\

    joint_angles = np.array([0.0, 0.0, 0.0])

    loss = np.inf  # initialize loss to infinity

    loss_gradient = np.array([0.0, 0.0, 0.0])  # initialize gradient

    # remove this line when you write your loop
    raise Exception("Loop termination condition not implemented yet.")
    while loss > TOLERANCE:
        # 1. recompute loss
        # 2. use finite difference method to compute gradient of loss
        # 3. update joint angles using newtons method
        break

    return joint_angles
