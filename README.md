# PowerUP!

> Solving a constrained optimization problem
>
> Osgood @2024-04-12 19:33:36


## Problem Reformulated

In total $N+1$ process, where $I_{N+1}$ is fixed.

Variable: Intensity $I_1, I_2, \cdots, I_{N}$

$$
\begin{align*}
\max &\quad P \\
\text{s.t.} &\quad \eta = \frac{P}{P + J}
\end{align*}
$$

where $P = \frac{W_{in}}{\tau}$, $J= \frac{Q}{\tau}$, and

$$
\tau = -\tau_0 + \frac{(V_m-V_1)C}{I_1} + \sum_{i=2}^N \tau_0 (\frac{I_{i-1}}{I_i}-1) + \tau_0 (\frac{I_{N}}{I_{N+1}} - 1)\\
Q = \tau_0 I_1(V_m-V_1 - I_1 R) + \sum_{i=2}^{N} \tau_0 R I_i (I_{i-1} - I_i) + \tau_0 I_{N+1}(I_N - I_{N+1}R)
$$

where $W_{in}$ is the input energy, $\tau$ is the time, $Q$ is the heat, $V_d$ is the voltage, and $R$ is the resistance, all of which are constants. $\tau_0$ and $\tau_1$ are also constants.

Here we have

$$
\eta = \frac{P}{P + J} = \frac{W_{in}}{W_{in} + Q} \to Q = \frac{W_{in}}{\eta} - W_{in} = (\frac{1}{\eta} - 1) W_{in} \equiv Q_0
$$

So we can rewrite the problem as

$$
\begin{align*}
\max &\quad \frac{1}{-\tau_0 + \frac{(V_m-V_1)C}{I_1} + \sum_{i=2}^N \tau_0 (\frac{I_{i-1}}{I_i}-1) + \tau_0 (\frac{I_{N}}{I_{N+1}} - 1)}
\\
\text{s.t.} &\quad \tau_0 I_1(V_m -V_1 - I_1 R) + \sum_{i=2}^{N} \tau_0 R I_i (I_{i-1} - I_i) + \tau_0 R I_{N+1}(I_N - I_{N+1}R) 
= Q_0
\end{align*}
$$

where $\tau_0 = CR, Q_0 = (\frac{1}{\eta}-1)\frac{C}{2}(V_f^2-V_1^2)$.

$$
\eta = \frac{P}{P + J} = \frac{W_{in}}{W_{in}+Q} = \frac{1}{1 + \frac{Q}{W_{in}}}
$$

$$
Q/W_{in} = \frac{\tau_0 I_1(V_m - V_1 - I_1 R) + \sum_{i=2}^N \tau_0 R I_i (I_{i-1}-I_i) + \tau_0 R I_{N+1}(I_N - I_{N+1}R)}{\frac{C}{2} (V_f^2 - V_1^2)}
$$

Constants Table:

| Variable | Range | Selected Value |
|:--------:|:-----:|:--------------:|
| $V_m$    | 0.5-1.5 | 1 |
| $V_f$    | 0.5-1.5 | 0.99 |
| $V_1$    | 0-1 | 0 |
| $C$      | 0.5-1.5 | 1.1 |
| $R$      | 0.5-1.5 | 1 |
| $\eta$   | 0.1-0.9 | 0.20 |

## How to solve

<!-- This is a constrained non-linear optimization problem. There are many methods to approach it, but it's all based on Lagrangian multiplier, where we write the objective function and constraints into a single function, and then minimize it. -->

<!-- We can firstly visualize the objective function and see the trajectory posed by the constraints to gain some intuition. -->

<!-- Then depends on the landscape, we use different optimization methods and track their performance. -->

We choose an optimization method, and plug in our problem, evaluate the result.

## Detailed Steps

<!-- 1. Write the objective function and constraints into a single Lagrangian function.

    We have
    $$
    L(I, \lambda) = \frac{1}{-\tau_0 + \frac{(V_d)C}{I_1} + \sum_{i=2}^N \tau_0 (\frac{I_{i-1}}{I_i}-1)} + \lambda (\tau_0 I_1(V_d - I_1 R) + \sum_{i=2}^{N} \tau_0 R I_i (I_{i-1} - I_i) - Q_0)
    $$
    where $I = [I_1, I_2, \cdots, I_{N}]$ and $\lambda$ is the Lagrangian multiplier.

    Constants

    $$
    \begin{align*}
    I_i &\in \color{red}{[0, 10]} \quad \forall i = 1, 2, \cdots, N-1\\
    \lambda &\in \color{red}[-10, 10]
    \end{align*}
    $$ -->


1. Choose appropriate optimization method.

    Categories of optimization methods:

    - Direct Search: Nelder-Mead, Powell, COBYLA, etc.
    - Gradient-based: BFGS, L-BFGS-B, TNC, etc.
    - Stochastic: Simulated Annealing, Genetic Algorithm, etc.

    When to use which method?

    - Direct Search: When the objective function is not smooth or not differentiable.
    - Gradient-based: When the objective function is smooth and differentiable.
    - Stochastic: When the objective function is complex and hard to optimize.

    For our problem here, choose L-BFGS-B, TNC, or COBYLA.

    In a well-known optimization library, such as `scipy.optimize`, there are [many optimization methods](https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize):

    - Unconstrained minimization of multivariate scalar functions (minimize)
    - **Constrained minimization of multivariate scalar functions (minimize)**
    - Global optimization
    - Least-squares minimization (least_squares)
    - Univariate function minimizers (minimize_scalar)
    - Custom minimizers
    - Root finding
    - Linear programming (linprog)
    - Assignment problems
    - Mixed integer linear programming

    we'll go through the constrained minimization part.
	参考 [[带非线性约束的非凸优化]]文档

3. Plug in the problem

    We can define the objective function, give a reasonable initial guess, and set the bounds.

    ```python
    import numpy as np

    # constants
    tau_0 = 1.3
    V_d = 1.2
    C = 1.1
    R = 14
    Q_0 = 11
    N = 10
    
    def objective(I):
        return 1 / (-tau_0 + C / I[0] + np.sum(tau_0 * (I[:-1] / I[1:] - 1)))
    
    def constraint(I):
        return tau_0 * I[0] * (V_d - I[0] * R) + np.sum(tau_0 * R * I[1:] * (I[:-1] - I[1:])) - Q_0
    
    ```

    - Defining Bounds Constraints:

    ```py
    from scipy.optimize import bounds

    # bounds for I
    bounds = [(0, 10)] * N
    ```

    - Defining Linear Constraints:

      We don't have linear constraints here.

    - Defining Nonlinear Constraints:

    ```py
    from scipy.optimize import NonlinearConstraint

    # Define the constraint
    def constraint(I):
        return tau_0 * I[0] * (V_d - I[0] * R) + np.sum(tau_0 * R * I[1:] * (I[:-1] - I[1:])) - Q_0

    # Define the Jacobian of the constraint
    def cons_J(I):
        return ...

    # Define the Hessian of the constraint
    def cons_H(I, v):
        return ...

    nonlinear_constraint = NonlinearConstraint(constraint, 0, 0, jac=cons_J, hess=cons_H)
    ```

    if the Hessian is not difficult to compute, we can use hessian update strategy like BFGS, L-BFGS-B, etc.

    The Jacobian and Hessian are optional, if not provided, the optimization method will use numerical approximation.

    - Solving the Optimization Problem:

    We then give an initial guess and use the optimization method to solve the problem.

    ```python
    from scipy.optimize import minimize
    I0 = np.ones(N)
    res = minimize(objective, I0, method='trust-constr', constraints=[nonlinear_constraint], bounds=bounds, options={'verbose': 1})
    ```

    Here `trust-constr` is the optimization method, and `verbose` is the option to print the optimization process.
    
4. Evaluate the result.

    To evaluate the result, we can print the result and check the optimization process.

    Since we already set the `verbose` option, we can see the optimization process.


The final code:

```py
import matplotlib.pyplot as plt
import numpy as np

# Constants for the objective function
tau_0 = 0  # Example value
C = 1.1      # Example value

def objective(I):
    return 1 / (-tau_0 + C / I[0] + np.sum(tau_0 * (I[:-1] / I[1:] - 1)))

# Generate random values of I in [0,10]^3
I = np.random.rand(100, 3) * 10

# Evaluate the objective function for each value of I
z = np.zeros(100)
for i in range(100):
    z[i] = objective(I[i])

temp = plt.plot(z)

# Create a 3D plot of the objective function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the random values of I
sc = ax.scatter(I[:, 0], I[:, 1], I[:, 2], c=z, cmap='viridis')

# Add a color bar
plt.colorbar(sc)

# Set titles and labels
ax.set_title('3D Objective Function Visualization')
ax.set_xlabel('I[0]')
ax.set_ylabel('I[1]')
ax.set_zlabel('I[2]')

# Show the plot
plt.show()

# check if all values in I is below 0
print(f"any value of I below 0: {np.any(I < 0)}")

# Check if any value is below 0
below_zero = np.any(z < 0)
print(f"Any value below 0: {below_zero}")
if below_zero:
    print("Values below 0:", z[z < 0])
```

注意到，当$\tau_0$大于0的时候，可能导致我们要优化的分母为0，出现问题。但是，$\tau_0 =CR$一定大于0，所以问题出在，我们应该强制要求$I_{i}>I_{i+1}$，也就是，充电电流必须单调递减，才有可能使得优化不出问题，这一点应该被写入显式的约束条件：

$$
I_{n} \geq I_{n+1}, n \in \{1, N-1\}
$$

This can be written in matrix form:

$$
\begin{bmatrix}
1 & -1 & 0 & \cdots & 0 \\
0 & 1 & -1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1
\end{bmatrix}
\begin{bmatrix}
I_1 \\
I_2 \\
\vdots \\
I_N
\end{bmatrix}
\ge
\begin{bmatrix}
0 \\
0 \\
\vdots \\
0
\end{bmatrix}
$$

write this constraint in `linear_constraint = LinearConstraint(...)` we have:

```py
# prompt: I_{n} \geq I_{n+1}, n \in \{1, N-1\}
# $$
# given N, build this matrix
# $$
# \begin{bmatrix}
# 1 & -1 & 0 & \cdots & 0 \\
# 0 & 1 & -1 & \cdots & 0 \\
# \vdots & \vdots & \vdots & \ddots & \vdots \\
# 0 & 0 & 0 & \cdots & 1
# \end{bmatrix}

import numpy as np

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
  for i in range(1, N):
    matrix[i, i-1] = -1

  return matrix

# Example usage
N = 5
matrix = build_matrix(N)

zeros = np.zeros(N)
infs = np.array(N, -np.inf)

linear_constraint = LinearConstraint(matrix, infs, zeros)

# Print the matrix
print(matrix)
```

## Result

Based on above discussion, we have the final code:

```py
import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint
from scipy.optimize import minimize
from matplotlib.pyplot import plot, scatter


# constants
vm=1
vf=0.99
v1=0
c=1
r=0.05
eta=0.2
i3=(vm-vf)/r
# derived constatns
tau_0 = c * r
C = c
R = r
W =  c / 2 * (vf**2 - v1**2)
Q = (1/eta - 1) * W
N = 10
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
ax = plot(res.x, '*')
# print(res)

print(f"Initial Power:{-objective(I0)}")
print(f"Current Power:{-objective(res.x)}")
print(f"Expected eta: {eta} \nCurrent eta: {real_eta} \n")
```

我们设置$\eta = 0.2$，然后阶段$N=10$，一开始给的猜测值为$I=[1,1,\cdots, 1]$，单位为安培。最终结果如下：

```
Initial Power:99.99999999999991
Current Power:999.8528362589634
Expected eta: 0.2
Current eta: 0.32
```

![alt text](download.png)

横轴为阶段，纵轴为电流，不能说完全正确，但将功率从一开始的100升到了999.8，并且$\eta$的误差在$3\%$左右，是基本成功的，如果选用更小的$N$，或者$\eta$要求低一点，迭代步数增加，结果会更好，取决于想要的精度。


## Power-Efficiency Relationship

上述给出了特定$\eta$下使得功率$P$最大化的充电策略$I$求解方案，并且根据$I$可以反推实际的$\eta_r$，从而绘制出$\eta_r(I), P$关系图，应该如同下面所展示的边界相似：

![[Pasted image 20240412202516.png]]

不过报错了，是何缘由？

而且最奇怪的是，$\eta$小的时候很难优化，大的时候反而迅速出结果，这是不正确的。

## Eta

$\eta$的表达式有误，因此修改为`real_eta = c * (vm - v1 - res.x[-1] * r)`，据此发现，每次运行给出的eta几乎相同，而且不是我们想要的。

在此基础上修改了，得到的结果如下：

```py
# constants
vm=1
vf=0.99
v1=0
c=1
r=0.05
i3=(vm-vf)/r
# derived constatns
tau_0 = c * r
C = c
R = r
W =  c / 2 * (vf**2 - v1**2)

def objective_function(eta_new, N):
  # constants
  vm=1
  vf=0.99
  v1=0
  c=1
  r=0.05
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

  print(f"Initial Power:{-objective(I0)}")
  print(f"Current Power:{-objective(res.x)}")
  print(f"Expected eta: {eta} \nCurrent eta: {real_eta} \n")

  return -objective(res.x), real_eta
```

用这个绘制图像：

```py
import matplotlib.pyplot as plt
# Define the range of eta values
N = 5
eta_values = np.linspace(0.1, 1, 20)  # Avoiding eta=0 to prevent division by zero

# Lists to store results
objective_values = []
real_etas = []

# Calculate objective values and real etas for each eta value
for eta in eta_values:
    objective_value, real_eta = objective_function(eta, N)
    objective_values.append(objective_value)
    real_etas.append(real_eta)

# Plotting
print(eta_values)
print(real_etas)
print(objective_values)
plt.plot(real_etas, objective_values)
plt.xlabel('Real Eta')
plt.ylabel('Objective Value')
plt.title('Relationship between Real Eta and Objective Value')
plt.grid(True)
plt.show()
```

![Image](https://pic4.zhimg.com/80/v2-ae942535666cd0c6c4c7c725f034a4ab.png)

美妙！

## Vf

对Vf进行扫描，得到每个Vf下的(eta, P_max)对，可以连接出一个结果



> 注意到，eta大反而优化快？并且令eta=0.01之类的情况，得到的real_eta=0.2+等大结果。

> 因为eta大容易有解，让eta小的解很少，算法找不到。

首先，我们求解的是一个充电策略，[I_1>= ..., >=I_n]，给出一个eta，和P

确定eta， 相当于确定了一组充电策略

找其中P最大的，作为P_max， (eta, P_max)

随机确定I_i ，请问eta大概率是小的。

N有限的时候，eta取很大是完全可以的，可以大于0.5的

N无限大的时候，相当于恒压充电，此时电流差异为指数，对应的eta很小，且有上界为1/2（即恒压充电时，电池发热量＞=充电功率）

eta的表达式：

1/(1+Q/W_in)

$Q/W_in = f(\Delta I)$

其中$\Delta I = I_{k-1}-I_k$

即：电流策略中，相临电流差异越大，Q/W_in越大，从而eta越小。

对于eta很大的情况，说明电流差距很小，这种策略非常容易给出[1, 1-0.001, 1-0.002,..]

对于eta很小的情况，差距很大，这个时候，I_1有上界([vm-v0]/r)，此时存在少数策略，使得Q/W_in最大

因此，优化算法很难落入eta很小的情况，eta大概率在0.2往上。

> 但是，如果令Vf很小，反而容易给出eta小的情况，eta大的反而算不出
>
> 因为eta与Vf正相关，如果Vf小，eta自然就会小（在确定的充电策略情况下）