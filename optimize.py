import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint
from scipy.optimize import minimize

def objective_function(eta_new=.2, N=5, vm=1., vf=.99, v1=0, c=1, r=1):
  """
  The optimization code, verified.
  Given parameters including

  Params:
    vm
    vf
    v1
    c
    r
    eta
  and the solution stage number
    N
  
  derive the corresponding charging strategy with maximal power

  and the real efficiency eta

  return: 
    power, eta
  """
  # constants
  eta=eta_new
  i3=(vm-vf)/r
  # derived constatns
  tau_0 = c * r
  C = c
  R = r
  W =  c / 2 * (vf**2 - v1**2)
  Q = (1/eta - 1) * W
  bounds = [(i3, vm/r)] * N   # bounds for I

  # helper functions
  def build_matrix(N):
    """
    Builds a matrix with the given properties.

    Args:
      N: The size of the matrix.

    Returns:
      A numpy array representing the matrix.
    """

    # Initialize the matrix with zeros.
    matrix = np.zeros((N, N))

    # Set the diagonal elements to 1.
    for i in range(N):
      matrix[i, i] = 1

    # Set the off-diagonal elements to -1.
    for i in range(N-1):
      matrix[i, i+1] = -1

    return matrix


  ## Objective funtion
  def objective(I):
          return - 1 / (-tau_0 +  (vm-v1)* C / I[0] + np.sum(tau_0 * (I[:-1] / I[1:] - 1)) + tau_0 * (I[-1]/i3 - 1) )


  ## Constraints definition

  # Nonlinear
  def constraint(I):
          return tau_0 * I[0] * (vm-v1 - I[0] * R) + np.sum(tau_0 * R * I[1:] * (I[:-1] - I[1:])) + tau_0 * R * i3 * (I[-1] - i3 * R)- Q
  
  nonlinear_constraint = NonlinearConstraint(constraint, 0, 0)
  # Linear
  matrix = build_matrix(N)
  zeros = np.zeros(N)
  infs = np.zeros(N) + np.inf
  linear_constraint = LinearConstraint(matrix,zeros, infs )

  # Initial Guess
  I0 = np.zeros(N) + i3

  # Optimize
  res = minimize(
          objective, I0, 
          method='trust-constr', 
          constraints=[nonlinear_constraint, linear_constraint], 
          bounds=bounds,
          options={'verbose': 1},
          )

  def Q_r(I):
          return tau_0 * I[0] * (vm - v1 + I[0]*R) + np.sum(tau_0* R * I[1:]  *(I[:-1]-I[1:]))+ tau_0*R*i3*(I[-1]- i3*R)

  real_eta = 1/ (1 + Q_r(res.x)/W)

  # Result
  # ax = plot(np.append(res.x, i3), '*')
  # print(res)

  # print(f"Initial Power:{-objective(I0)}")
  # print(f"Current Power:{-objective(res.x)}")
  # print(f"Expected eta: {eta} \nCurrent eta: {real_eta} \n")

  return -objective(res.x), real_eta
