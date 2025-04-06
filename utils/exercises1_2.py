from scipy.integrate import solve_ivp
from scipy.integrate import quad
import torch
import numpy as np
import utils.exercises1_1 as ex1_1
import matplotlib.pyplot as plt

class MonteCarloSDE:
    def __init__(self, H, M, C, D, R, sigma, T, N):
        """
        Initialize Monte Carlo simulator for LQR problem

        Parameters:
            H, M, C, D, R: Matrix of linear quadratic regulator
            sigma: Noise term
            T: Terminal time
            N: The number of time steps
            dt: time steps
            time_grid: Time grid
            S_values: The solution of Ricatti ODE
        """
        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.T = T
        self.N = N
        self.dt = T/N
        self.time_grid = torch.linspace(0, T, N + 1)
        self.S_values = self.solve_ricatti_ode()

    def ricatti_ode(self, t, S_flat):
        """
        Obtain the expression of Riccati ODE: 
            S'(t) = S(t) M D^{-1} M^{T} S(t) - H^{T} S(t) - S(t) H - C , S(T) = R
        
        Input:
            t: current time
            S_flat: S value at time t
        output:
            S_dot.flatten(): Solve the derivative of S and convert to one dimension
        """
        # Reshape S for 2x2 matrix
        S = torch.tensor(S_flat, dtype = torch.float32).reshape(2,2)
        # Compute the derivative of S(t)
        S_dot = S @ self.M @ torch.linalg.inv(self.D) @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
        
        return S_dot.flatten()

    def solve_ricatti_ode(self):
        """
        Use solve_ivp to solve Ricatti ODE 

        Output:
            Return a dictionary with corresponding time and S values one by one
        """
        # Terminal S
        S_T = self.R.flatten() 
        # Generate reverse index
        index = torch.arange(self.time_grid.size(0) - 1, -1, -1)
        # Reverse time grid
        time_grid_re = torch.index_select(self.time_grid, 0, index)

        # Solve Ricatti ODE
        sol = solve_ivp(self.ricatti_ode, [self.T, 0], S_T, t_eval = time_grid_re, atol = 1e-10, rtol = 1e-10)  
        # Convert back to matrix format
        S_matrix = sol.y.T[::-1].reshape(-1, 2, 2) 

        return dict(zip(tuple(self.time_grid.tolist()), S_matrix))

    def get_nearest_S(self, t):
        """
        Find the value of S that is closest to t

        Input:
            t: time
        Output:
            self.S_values[nearest_t.item()]: S(t) that is closest to t
        """
        # Find the nearest t
        nearest_t = self.time_grid[torch.argmin(torch.abs(self.time_grid - t))]
        
        return self.S_values[nearest_t.item()]
    
    def simulate_byMC(self, x, M_samples):
        """
        Monte Carlo simulation of LQR problems by using implicit Eulerian discretisation
            X_tn+1 = X_tn + dt [H X_tn+1 - M D^{-1} M^{T} S(tn+1) X_tn+1] + σ (W_tn+1 - W_tn )

        Input:
            x: Initial x
            M_samples: Monte Carlo samples
        """
        # Copy the initial state x0_torch to get the initial values of all samples
        X = x.unsqueeze(0).repeat(M_samples, 1)
        torch.manual_seed(1919)
        dW = torch.randn(M_samples, self.N, 2) * np.sqrt(self.dt)
        eye_dim = torch.eye(2)

        # Initialize the value function for all samples
        V_values = torch.zeros(M_samples)

        for n in range(self.N):
            tn = n * self.dt
            S_t = self.get_nearest_S(tn)
            S_t = torch.tensor(S_t, dtype = torch.float32)

            # Compute the coefficient of a = -D^(-1) M^T S(t)
            a = -torch.linalg.inv(self.D) @ self.M.T @ S_t

            # Compute X_next: X_tn+1 [I - [H - M D^{-1} M^{T} S(tn+1)]dt ] = X_tn + σ (W_tn+1 - W_tn )
            A = eye_dim - self.dt * (self.H + self.M @ a)
            b = X + (dW[:, n, :] @ self.sigma.T)
            X_next = b @ torch.linalg.inv(A).T

            # Cost = (X_next^T C X_next + alpha^T R alpha) * dt
            # Compute control a(t, x) = -D^(-1) M^T S(t) x = a x
            a_n = X_next @ a.T
            stage_cost_x = torch.sum(X_next @ self.C * X_next, dim = 1)
            stage_cost_a = torch.sum(a_n  @ self.D * a_n,  dim = 1)

            V_values += (stage_cost_x + stage_cost_a) * self.dt
            X = X_next
        
        # Terminal cost g_T
        terminal_cost = torch.sum(X @ self.R * X, dim = 1)
        V_values += terminal_cost

        return V_values.mean().item()

def error_on_N(H, M, C, D, R, sigma, T, M_samples_fixed, N_step_list, v, t0, x0):
    """
    Calculate the error between Monte Carlo estimator and actual values
    With number of Monte Carlo samples large vary the number of time steps

    Input:
        H, M, C, D, R: Matrix of linear quadratic regulator
        sigma: Noise term
        T: Terminal time
        M_samples_fixed: The number of Monte Carlo samples
        N_step_list: The list of different number of time steps
        v: actual values
        t0: Initial time
        x0: Initial x
    Output:
        err_N: The error between Monte Carlo estimator and actual values
    """
    err_N = []

    for N_step in N_step_list:
        MClqr = MonteCarloSDE(H, M, C, D, R, sigma, T, N_step)
        # Compute Monte Carlo estimator
        v_est = MClqr.simulate_byMC(x0, M_samples_fixed)
        # Compute weak error
        err = abs(v_est - v)
        err_N.append(err)
    
    return err_N

def error_on_M(H, M, C, D, R, sigma, T, N_fixed, M_samples_list, v, t0, x0):
    """
    Calculate the error between Monte Carlo estimator and actual values
    With number of Monte Carlo samples large vary the number of time steps

    Input:
        H, M, C, D, R: Matrix of linear quadratic regulator
        sigma: Noise term
        T: Terminal time
        N_fixed: The number of time steps
        M_samples_list: The list of different number of Monte Carlo samples
        v: actual values
        t0: Initial time
        x0: Initial x
    Output:
        err_M: The error between Monte Carlo estimator and actual values
    """
    err_M = []

    MClqr = MonteCarloSDE(H, M, C, D, R, sigma, T, N_fixed)

    for M_samples in M_samples_list:
        # Compute Monte Carlo estimator
        v_est = MClqr.simulate_byMC(x0, M_samples)
        # Compute weak error
        err = abs(v_est - v)
        err_M.append(err)
    
    return err_M

def loglog_plot(x, y, fixed_para, change_para):
    """
    Plot the error as a log-log plot.

    Input:
        x: list of M_samples or time steps
        y: The error between Monte Carlo estimator and actual values
        fixed_para: fix parameter
        change_para: change parameter
    """

    plt.figure(figsize = (12, 4))
    plt.loglog(x, y, 'o-', label = f'Weak Error')
    
    # Theoretical reference line
    if change_para == 'N':
        # Choose the appropriate proportionality constant C
        C = y[0] * x[0] 
        # Calculate the value of the reference line
        ref_line = C * 1/ np.array(x)
        # Vary the time steps, the expected convergence rate is -1
        plt.loglog(x, ref_line, '--', label = 'Slope -1')
    else:
        C = y[0] * x[0]  
        ref_line = C * 1/np.sqrt(np.array(x)) 
        # Vary the M_samples, the expected convergence rate is -1/2
        plt.loglog(x, ref_line, '--', label = 'Slope -1/2')

    plt.xlabel(f"{change_para} (log)")
    plt.ylabel("Error (log)")
    plt.title(f"Implicit Euler Scheme: error over {change_para} (fixed {fixed_para})")
    plt.legend()
    plt.grid()
    plt.show()

def compute_error_slope(x_values, y_values):
    """
    Perform linear regression on straight line and calculate the slope
    Input:
        x_values: the x values of straight line
        y_values: the y values of straight line
    Output:
        slope: The slope of straight line
    """
    logx = np.log(x_values)
    logy = np.log(y_values)
    slope, _ = np.polyfit(logx, logy, 1)

    return slope