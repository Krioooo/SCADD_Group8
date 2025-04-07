import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D

def Radon_Nikodym_derivative(dist_pi, dist_mu, a_n):
    """
    Compute log Radon-Nikodym derivative log p^θ = log (dπ^θ/dµ) = log dπ^θ - log dµ

    Input:
        dist_pi: the distribution of π^θ
        dist_mu: the distribution of µ
        a_n: control a
    Output:
        RN: log Radon-Nikodym derivative for the given a_n
    """
    RN = dist_pi.log_prob(a_n) - dist_mu.log_prob(a_n)

    return RN

class SoftLQREnvironment:
    def __init__(self, H, M, C, D, R, sigma, T, N, tau, gamma):
        """
        Initialize soft LQR environment

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
            dist_mu: the distribution of µ, µ = N(0, γ^2 Imxm)
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
        self.D_eff = self.D + (self.tau / (2 * (self.gamma ** 2))) * torch.eye(2)
        self.dist_mu = MultivariateNormal(torch.zeros(2), gamma**2*torch.eye(2))
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
    
    def reset(self):
        """
        Initialize x0, initial state distribution ρ = U([-2, 2] x [-2, 2])
        """
        x0 = torch.tensor([torch.empty(1).uniform_(-2, 2), torch.empty(1).uniform_(-2, 2)], dtype = torch.float32)
        return x0
    
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
            cost_opt_cum: the cumulative cost for the given x0, dW
        """
        x_traj = [x0.numpy()]
        cost_opt = []
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

            # cost
            cost_n = x_tn.T @ self.C @ x_tn + a_n.T @ self.D @ a_n
            # Compute log Radon-Nikodym derivative log p^θ
            log_prob = Radon_Nikodym_derivative(control_dist, self.dist_mu, a_n)
            # Compute running cost: f_tn + τ * ln p^θ
            running_cost_n = cost_n + self.tau * log_prob
            cost_opt.append(running_cost_n.unsqueeze(0))

            # drift = Hx + Ma
            drift = self.H @ x_tn + self.M @ a_n

            # noise = sigma dW
            noise = self.sigma @ dW[n]

            # explicit Euler scheme
            x_next = x_tn + drift * self.dt + noise
            x_tn = x_next
            x_traj.append(x_tn.numpy())
        
        # Terminal cost
        g_T = x_tn.T @ self.R @ x_tn
        cost_opt.append(g_T.unsqueeze(0))

        cost_opt = torch.stack(cost_opt)
        cost_opt_cum = torch.cumsum(cost_opt, dim = 0)

        return np.array(x_traj), cost_opt_cum

class ValueNN(nn.Module):
    def __init__(self, hidden_size = 512):
        """
        A neural network for approximating the value function V(t, x).
        This network only takes the time variable t as input and predicts:  
            - A symmetric positive semi-definite 2x2 matrix Q(t) representing the quadratic term x^T Q(t) x
            - A scalar offset term c(t)
        The final value function approximation is:
        v(t, x) ≈ x^T Q(t) x + c(t)

        Parameters:
            l1 (nn.Linear): First hidden layer mapping ℝ³ → ℝ^{hidden_size}.
            l2 (nn.Linear): Second hidden layer mapping ℝ^{hidden_size} → ℝ^{hidden_size}.
            matrix (nn.Linear): symmetric positive semi-definite matrix
            offset (nn.Linear): scalar offset
        """
        super().__init__()
        self.l1 = nn.Linear(1, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.matrix = nn.Linear(hidden_size,4) # =>2x2
        self.offset = nn.Linear(hidden_size,1)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        """
        Forward pass to get the value function.
        Returns a value function for the given t, x.

        Input:
            t: current time
            x: current x
        Output:
            quad_term + offset.squeeze(-1): a estimated value v(t, x) for the given t, x.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32)

        h = self.relu(self.l1(t))
        h = self.relu(self.l2(h))
        matrix = self.matrix(h)      # (batch,4)
        offset = self.offset(h)      # (batch,1)

        # reshape => 2x2
        matrix_2x2 = matrix.view(-1,2,2)
        matrix_sym = torch.bmm(matrix_2x2, matrix_2x2.transpose(1, 2))
        # Make the matrix positive definite(存疑)
        matrix_pd = matrix_sym + 1e-3 * torch.eye(2)
        
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 2, 1)
        quad_term = torch.bmm(torch.bmm(x_reshaped.transpose(1,2), matrix_pd), x_reshaped).view(-1)

        return quad_term + offset.squeeze(-1)

def OfflineCriticOnly(env, valueNN, epoch, n_episodes, lr):
    """
    Train a value function neural network (critic) using an offline critic-only algorithm.
    
    Input:
        env: A soft LQR environment
        valueNN: Value function neural network that approximates V(t, x).
        epoch: Number of training episodes.
        n_episodes: Number of episodes in one epoch.
        lr: Learning rate for the critic (valueNN). 
    Output:
        cost_history: Total running cost for every episode.
        criticloss: Critic loss for every episode.
    """
    # Create optimizers for valueNN
    critic_optim = optim.Adam(valueNN.parameters(), lr = lr)

    loss_list = []

    # Start to train
    for ep in range(epoch):
        # Sample x0
        x0_batch = torch.empty(n_episodes, 2).uniform_(-2.0, 2.0)
        loss = 0.0
        
        for i in range(n_episodes):
            x0 = x0_batch[i]
            # Simulate a trajectory
            x_tn = x0.clone()
            dW = torch.randn(env.N, 2) * np.sqrt(env.dt)

            t_list = []
            x_list = []
            cost_list = []

            for n in range(env.N):
                tn = n * env.dt
                S_tn = env.get_nearest_S(tn)
                S_tn = torch.tensor(S_tn, dtype = torch.float32)

                # mean
                mean= -torch.linalg.inv(env.D_eff) @ env.M.T @ S_tn @ x_tn 
                # covarian
                cov = env.tau * env.D_eff
                # distribution
                dist = MultivariateNormal(mean, cov)
                a_n = dist.sample()

                # Use explicit euler to get new state x_tn+1
                drift = env.H @ x_tn + env.M @ a_n
                noise = env.sigma @ dW[n]
                x_next = x_tn + drift * env.dt + noise
                
                # Compute log Radon-Nikodym derivative log p^θ
                log_prob = Radon_Nikodym_derivative(dist, env.dist_mu, a_n)
                # cost: f_tn
                f_n = x_tn.T @ env.C @ x_tn + a_n.T @ env.D @ a_n
                # running cost
                cost_n = f_n + env.tau * log_prob

                # Store data
                t_list.append(tn)
                x_list.append(x_tn)
                cost_list.append(cost_n)

                # Update
                x_tn = x_next
            x_T = x_tn
            # Terminal cost
            g_T = x_T.T @ env.R @ x_T

            # Compute target term: - (sum_{k=n}^{N-1} (f_tk + τ ln pθ(αtk |tk, Xtk ))(∆t) + gtN
            target = torch.zeros(env.N, dtype = torch.float32)
            total = 0.0
            for n in reversed(range(env.N)):
                total += cost_list[n] * env.dt
                target[n] = total
            
            # Compute L(η)
            t = torch.tensor(t_list).float().view(-1,1)
            x = torch.stack(x_list)
            # Use value function neural network to approximate v(t, x)
            v_n = valueNN(t, x)
            for n in range(env.N):
                loss += (v_n[n] - target[n] - g_T) ** 2

        critic_optim.zero_grad()
        loss.backward()
        critic_optim.step()
        loss_list.append(loss.item())
        print(f"[Epoch {ep}/{epoch}] CriticLoss = {loss.item():.6f}")

    return loss_list

def plot_criticloss(criticloss, average = True):
    """
    Plot the critic loss curve (in log10 scale) over training iterations or epochs.
    If average == True, the function averages the loss values over fixed-size chunks 
    (default: 500 steps per epoch) to reduce noise and visualize the overall trend more clearly.

    Input:
        criticloss: A list of critic loss values collected during training.
        average: Whether to average the loss values over epochs. 
                 When set to True, the loss values are grouped into chunks of 500 steps, 
                 and the mean of each chunk is plotted.
    """
    critic_loss = criticloss

    plt.figure(figsize = (12,5))
    plt.plot(np.log10(critic_loss), label='Critic log Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Critic Loss(log)')
    plt.title('Critic log of loss over Epoch')
    plt.grid()
    plt.legend()
    plt.show()

def plot_value3D(t_test, data_list):
    """
    Plot 3D surface plots for the learned value function and the theoretical value function.

    Input:
        t_test: List of time points 
        data_list: A list of datasets, where each dataset contains four columns(x1, x2, v_learned, v_theoretical)
    """
    fig = plt.figure(figsize = (10, 12))

    for i, data in enumerate(data_list):
        data = np.array(data)

        # Make the axes 3D
        ax1 = fig.add_subplot(4, 2, i*2+1, projection='3d')
        ax2 = fig.add_subplot(4, 2, i*2+2, projection='3d')

        # Plot the trisurf on each 3D axis
        ax1.plot_trisurf(data[:, 0], data[:, 1], data[:, 2], cmap='viridis')
        ax1.set_title(f'v_learn(t = {t_test[i]:.2f})')

        ax2.plot_trisurf(data[:, 0], data[:, 1], data[:, 3], cmap='plasma')
        ax2.set_title(f'v_theoretical(t = {t_test[i]:.2f})')

        # Optionally, set axis labels for clarity
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('value')

        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('value')
    
    plt.tight_layout()
    plt.show()

def find_maximum_error(env, valueNN, t_test, x_range):
    """
    Evaluate the maximum approximation error of a neural network value function over a grid of input states and times.

    Input:
        env: A soft LQR environment
        valueNN: Value function neural network that approximates V(t, x).
        t_test: List of time points at which to evaluate the value function.
        x_range: Range of values for each dimension of the state
    Output:
        data_list: A list containing tuples of the form (x1, x2, v_learned, v_theoretical), 
              storing both the neural network output and true value for each state.
    """
    max_error = 0.0
    data_list = []
    for t_value in t_test:
        S_t = env.get_nearest_S(t_value)
        data = []
        for x1 in x_range:
            for x2 in x_range:
                t_tensor = torch.tensor([[t_value]]).float()
                x_vector = torch.tensor([[x1, x2]], dtype = torch.float32)
                v_learn = valueNN.forward(t_tensor, x_vector).item()
                v_theoretical = env.value_function(t_value, x_vector.squeeze()).item()
                max_error = max(max_error, abs(v_learn - v_theoretical))
                data.append([x1, x2, v_learn, v_theoretical])
        data_list.append(data)
    print(f"Max error on specified grid: {max_error:.2f}")
    return data_list