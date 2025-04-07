from scipy.integrate import solve_ivp
from scipy.integrate import quad
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import numpy as np
import matplotlib.pyplot as plt

class Soft_LQR:
    def __init__(self, H, M, C, D, R, sigma, T, N, tau, gamma):
        """
        Initialize soft LQR class

        Parameters:
            H, M, C, D, R: Matrix of linear quadratic regulator
            sigma: Noise term
            T: Terminal time
            N: The number of time steps
            dt: time steps
            tau: strength of entropic regularization
            gamma: strength of variance of prior normal density 
            time_grid: Time grid
            D_eff: Correction term D
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
        self.time_grid = torch.linspace(0, T, N+1)
        self.tau = tau
        self.gamma = gamma
        self.D_eff = self.D + self.tau / (2 * (self.gamma ** 2)) * torch.eye(2)
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
        S = torch.tensor(S_flat, dtype=torch.float32).reshape(2,2) 
        # Compute the derivative of S(t)
        S_dot = S.T @ self.M @ torch.linalg.inv(self.D_eff) @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
        
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
    
    def value_function(self, t, x):
        """
        Compute the value funtion:
            v(t, x) = x^T S(t) x + ∫[t,T] tr(σσ^T S(r)) dr + (T-t)C_{D,tau, gamma}
        
        Input: 
            t: time
            x: initial x
        Output:
            value: the control problem value v(t, x) for the given t, x
        """
        # First term：x^T S(t) x
        S_t = self.get_nearest_S(t)
        S_t = torch.tensor(S_t, dtype = torch.float32)
        first_term = x.T @ S_t @ x
        
        # Second term: ∫[t,T] tr(σσ^T S(r)) dr
        def integrand(r):
            S_r = self.get_nearest_S(r)
            return torch.trace(self.sigma @ self.sigma.T @ S_r)
        
        # Using numerical integration to calculate the integral term
        integral, _ = quad(integrand, t, self.T) 


        # Third term：(T-t)C_{D,tau, gamma}
        # C_{D,tau, gamma} = -tau ln(tau^{m/2}/gamma^{m} * det(∑)^{1/2}), ∑-1 = D+tau/(2*gamma^2)I
        inv_matrix = torch.linalg.inv(self.D_eff)
        det_matrix = torch.det(inv_matrix)
        C = - self.tau * torch.log((self.tau / self.gamma ** 2) * torch.sqrt(det_matrix))
        entropic = (self.T - t) * C

        value = first_term + integral + entropic

        return value
    
    def optimal_control(self, t, x):
        """
        Compute the optimal control distribution:
            pi(·|t, x) = N(-(D+tau/(2*gamma^2)I)^(-1) M^T S(t) x, tau(D+tau/(2*gamma^2)I))
        
        Input: 
            t: time
            x: initial x
        Output:
            control_dist: the optimal control distribution pi(·|t, x) for the given t, x
        """
        S_t = self.get_nearest_S(t)
        S_t = torch.tensor(S_t, dtype=torch.float32)
        # mean
        mean_control = -torch.linalg.inv(self.D_eff) @ self.M.T @ S_t @ x
        # covarian
        cov_control = self.tau * self.D_eff
        # distribution
        control_dist = MultivariateNormal(mean_control, cov_control)
        return control_dist
    
    def simulate_trajectory(self, x0, dW):
        """
        Use Euler scheme to simulate soft LQR trajectory
        Explicit Euler:
            X_tn+1 = X_tn + dt [H X_tn - M D^{-1} M^{T} S(tn) X_tn )] + σ(W_tn+1 - W_tn ),

        Input:
            x0: Initial x
            dW: Brownian motion
        Output:
            np.array(x_traj): the LQR trajectory of x for the given x0, dW
        """
        x_traj = [x0.numpy()]
        x_tn = x0
        
        for n in range(self.N):
            tn = n * self.dt
            S_tn = self.get_nearest_S(tn)
            S_tn = torch.tensor(S_tn, dtype = torch.float32)

            # mean
            mean_control = -torch.linalg.inv(self.D_eff) @ self.M.T @ S_tn @ x_tn     
            
            # covarian
            cov_control = self.tau * self.D_eff

            # distribution
            control_dist = MultivariateNormal(mean_control, cov_control)
            a_n = control_dist.sample()

            # drift = Hx + Ma
            drift = self.H @ x_tn + self.M @ a_n

            # noise = sigma dW
            noise = self.sigma @ dW[n]

            # explicit Euler scheme
            x_next = x_tn + drift * self.dt + noise
            x_tn = x_next
            x_traj.append(x_tn.numpy())

        return np.array(x_traj)

def plot_x_trajectory(initial_states, lqr, soft_lqr, dW):
    """
    Plot the trajectory of initial point x

    Input:
        initial_states: lists of initial point
        lqr: LQR class
        soft_lqr: soft LQR class
        dW: Brownian motion
    """
    plt.figure(figsize = (12, 8))

    for i, x0 in enumerate(initial_states):
        # simulate trajectory
        x_traj_strict = lqr.simulate_trajectory(x0, dW)
        x_traj_soft = soft_lqr.simulate_trajectory(x0, dW)

        # Plot strict lqr trajectory
        plt.plot(x_traj_strict[:, 0], x_traj_strict[:, 1],
                color = 'blue', linestyle = '-',
                label = 'Strict LQR' if i == 0 else None)
        # Plot the initial points of strict lqr
        plt.plot(x0[0], x0[1],
                color = 'blue', marker = 'o', markersize = 10)
        
        # Plot soft lqr trajectory
        plt.plot(x_traj_soft[:, 0], x_traj_soft[:, 1],
                color = 'red', linestyle = '-',
                label = 'Soft LQR' if i == 0 else None)
        # Plot the initial points of strict lqr
        plt.plot(x0[0], x0[1],
                color = 'red', marker = 's', markersize = 5)
        
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Controlled trajectories: Strict vs Soft LQR')
    plt.grid()
    plt.legend()
    plt.show()   