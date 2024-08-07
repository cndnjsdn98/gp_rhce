import casadi as cs
import numpy as np

def discretize_dynamics_and_cost(t_horizon, n_points, m_steps_per_point, x, u, dynamics_f, cost_f, ind):
    """
    Integrates the symbolic dynamics and cost equations until the time horizon using a RK4 method.
    :param t_horizon: time horizon in seconds
    :param n_points: number of control input points until time horizon
    :param m_steps_per_point: number of integrations steps per control input
    :param x: 4-element list with symbolic vectors for position (3D), angle (4D), velocity (3D) and rate (3D)
    :param u: 4-element symbolic vector for control input
    :param dynamics_f: symbolic dynamics function written in CasADi symbolic syntax.
    :param cost_f: symbolic cost function written in CasADi symbolic syntax. If None, then cost 0 is returned.
    :param ind: Only used for trajectory tracking. Index of cost function to use.
    :return: a symbolic function that computes the dynamics integration and the cost function at n_control_inputs
    points until the time horizon given an initial state and
    """

    if isinstance(cost_f, list):
        # Select the list of cost functions
        cost_f = cost_f[ind * m_steps_per_point:(ind + 1) * m_steps_per_point]
    else:
        cost_f = [cost_f]

    # Fixed step Runge-Kutta 4 integrator
    dt = t_horizon / n_points / m_steps_per_point
    x0 = x
    q = 0

    for j in range(m_steps_per_point):
        k1 = dynamics_f(x=x, u=u)['x_dot']
        k2 = dynamics_f(x=x + dt / 2 * k1, u=u)['x_dot']
        k3 = dynamics_f(x=x + dt / 2 * k2, u=u)['x_dot']
        k4 = dynamics_f(x=x + dt * k3, u=u)['x_dot']
        x_out = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        x = x_out

        if cost_f and cost_f[j] is not None:
            q = q + cost_f[j](x=x, u=u)['q']

    return cs.Function('F', [x0, u], [x, q], ['x0', 'p'], ['xf', 'qf'])


def _forward_prop_core(x_0, u_seq, t_horizon, discrete_dynamics_f, dt, m_int_steps, mhe=False):
    """
    Propagates forward the state estimate described by the mean vector x_0 and the covariance matrix covar, and a
    sequence of inputs for the system u_seq. These inputs can either be numerical or symbolic.

    :param x_0: initial mean state of the state probability density function. Vector of length m
    :param u_seq: sequence of flattened N control inputs. I.e. vector of size N*4
    :param t_horizon: time horizon corresponding to sequence of inputs
    :param discrete_dynamics_f: symbolic function to compute the discrete dynamics of the system.
    :param dt: Vector of N timestamps, the same length as w_opt / 2, corresponding to the time each input is applied.
    :param m_int_steps: number of intermediate integration steps per control node.
    :return: The sequence of mean and covariance estimates for every corresponding input, as well as the computed
    cost for each stage.
    """
    if not isinstance(x_0, np.array):
        x_0 = np.array(x_0)

    if mhe:
        # Reshape input sequence to a N x 6 (1D) vector. Assume control input dim = 6
        k = np.arange(int(u_seq.shape[0] / 6))
        input_sequence = cs.horzcat(u_seq[6* k], u_seq[6 * k + 1], u_seq[6 * k + 2], 
                                    u_seq[6 * k + 3], u_seq[6 * k + 4], u_seq[6 * k + 5])
    else:
        # Reshape input sequence to a N x 4 (1D) vector. Assume control input dim = 4
        k = np.arange(int(u_seq.shape[0] / 5))
        input_sequence = cs.horzcat(u_seq[5 * k], u_seq[5 * k + 1], u_seq[5 * k + 2], u_seq[5 * k + 3], u_seq[5 * k + 4])

    if mhe:
        N = int(u_seq.shape[0] / 6)
    else:
        N = int(u_seq.shape[0] / 5)

    if dt is None:
        dt = t_horizon / N * np.ones((N, 1))

    if len(dt.shape) == 1:
        dt = np.expand_dims(dt, 1)

    # Initialize sequence of propagated states
    mu_x = [x_0]

    for k in range(N):
        # Get current control input and current state mean and covariance
        u_k = input_sequence[k, :]
        mu_k = mu_x[k]

        # mu(k+1) vector from propagation equations. Pass state through nominal dynamics with GP mean augmentation if GP
        # is available. Otherwise use nominal dynamics only.
        f_func = discrete_dynamics_f(dt[k], 1, m_int_steps, k, mhe=mhe)

        fk = f_func(x0=mu_k, p=u_k)
        mu_next = fk['xf']

        # Add next state estimate to lists
        mu_x += [mu_next]

    mu_x = cs.horzcat(*mu_x)
    mu_prop = np.array(mu_x).T

    return mu_prop
