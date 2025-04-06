import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

def Radon_Nikodym_derivative(dist_pi, dist_mu, a_n):
    RN = dist_pi.log_prob(a_n) - dist_mu.log_prob(a_n)
    return RN

class SoftLQREnvironment:
    def __init__(self, H, M, C, D, R, sigma, T, N, tau, gamma):
        """
        SoftLQR: 带熵正则的 LQR 问题
        状态方程： dX = (H X + M a )dt + sigma dW
        成本： ∫ (x^T C x + a^T D a + tau ln p(a|x)) dt + x_T^T R x_T
        控制策略： a ~ N(mean, cov)
        其中 mean = -D_eff^{-1} M^T S(t)x,  cov = tau * D_eff,
        D_eff = D + (tau/(2*gamma^2)) I.
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
        self.S_values = self.solve_riccati_ode()
    
    def riccati_ode(self, t, S_flat):
        """Riccati ODE 求解函数，转换为向量形式"""
        S = torch.tensor(S_flat, dtype=torch.float32).reshape(2,2) # 2x2 矩阵
        S_dot = S.T @ self.M @ torch.linalg.inv(self.D_eff) @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
        return S_dot.flatten()
    
    def solve_riccati_ode(self):
        """使用 solve_ivp 求解 Riccati ODE"""
        S_T = self.R.flatten()  # 终止条件 S(T) = R
        indices = torch.arange(self.time_grid.size(0) - 1, -1, -1)  # 生成倒序索引
        time_grid_re = torch.index_select(self.time_grid, 0, indices)
        sol = solve_ivp(self.riccati_ode, [self.T, 0], S_T, t_eval = time_grid_re, atol = 1e-10, rtol = 1e-10)  # 逆向求解
        S_matrices = sol.y.T[::-1].reshape(-1, 2, 2)  # 转换回矩阵格式
        return dict(zip(tuple(self.time_grid.tolist()), S_matrices))

    def get_nearest_S(self, t):
        """找到最近的 S(t)"""
        nearest_t = self.time_grid[torch.argmin(torch.abs(self.time_grid - t))]
        return self.S_values[nearest_t.tolist()]
    
    def value_function(self, t, x):
        """计算新的 v(t, x) = x^T S(t) x + ∫[t,T] tr(σσ^T S(r)) dr + (T-t)C_{D,tau, gamma}"""
        # 第一部分：x^T S(t) x
        S_t = self.get_nearest_S(t)
        S_t = torch.tensor(S_t, dtype = torch.float32)
        value = x.T @ S_t @ x
        
        # 第二部分：积分项 ∫[t,T] tr(σσ^T S(r)) dr
        def integrand(r):
            S_r = self.get_nearest_S(r)
            return torch.trace(self.sigma @ self.sigma.T @ S_r)
        
        integral, _ = quad(integrand, t, self.T)  # 使用数值积分计算积分项
        value += integral

        # 第三部分：(T-t)C_{D,tau, gamma}
        # C_{D,tau, gamma} = -tau ln(tau^{m/2}/gamma^{m} * det(∑)^{1/2}), ∑-1 = D+tau/(2*gamma^2)I
        inv_matrix = torch.linalg.inv(self.D_eff)
        det_matrix = torch.det(inv_matrix)
        C = - self.tau * torch.log((self.tau / self.gamma ** 2) * torch.sqrt(det_matrix))
        entropic = (self.T - t) * C
        value += entropic

        return value
    
    def optimal_control(self, t, x):
        """计算最优控制分布 pi(·|t, x) = N(-(D+tau/(2*gamma^2)I)^(-1) M^T S(t) x, tau(D+tau/(2*gamma^2)I))"""
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
        x0 = torch.tensor([torch.empty(1).uniform_(-2, 2), torch.empty(1).uniform_(-2, 2)], dtype = torch.float32)
        return x0
    
    def simulate_trajectory(self, x0, dW):
        """
        使用 Euler 方法模拟 soft LQR 轨迹
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
            control_a = control_dist.sample()

            # cost
            cost_n = x_tn.T @ self.C @ x_tn + control_a.T @ self.R @ control_a
            cost_opt.append(cost_n)
            # drift = Hx + Ma
            drift = self.H @ x_tn + self.M @ control_a

            # noise = sigma dW
            noise = self.sigma @ dW[n]

            # explicit Euler scheme
            x_next = x_tn + drift * self.dt + noise
            x_tn = x_next
            x_traj.append(x_tn.numpy())

        # terminal cost
        g_T = x_tn.T @ self.R @ x_tn
        cost_opt.append(g_T)
        cost_opt = torch.stack(cost_opt)
        cost_opt = torch.cumsum(cost_opt, dim=0)
        return np.array(x_traj), cost_opt

class PolicyNN(nn.Module):
    def __init__(self, hidden_size = 512):
        super(PolicyNN, self).__init__()

        self.hidden_layer1 = nn.Linear(1, hidden_size) 
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.dim = 2

        # Output for phi 
        self.phi = nn.Linear(hidden_size, 2 * 2)
        # Output for L matrix for Sigma 
        self.sigma_L = nn.Linear(hidden_size, 2 * (2 + 1) // 2)

        # precompute 
        self.tri_indices = torch.tril_indices(self.dim, self.dim)
    
    def forward(self, t, x):
        """
        Forward pass to get the action distribution.
        Returns a MultivariateNormal distribution.
        """
        # Forward pass 
        t = t.view(-1, 1)  # Ensure t is a column vector 
        hidden = torch.relu(self.hidden_layer1(t)) 
        hidden = torch.sigmoid(self.hidden_layer2(hidden))

        # Compute phi 
        phi_flat = self.phi(hidden) 
        phi = phi_flat.view(-1, self.dim, self.dim)

        # Compute Sigma 
        L_flat = self.sigma_L(hidden) 
        # Create a lower triangular matrix L where L_flat fills the lower triangle 
        L = torch.zeros(self.dim, self.dim) 
        L[self.tri_indices[0], self.tri_indices[1]] = L_flat 
        
        # Compute Sigma = LL^T to ensure positive semi-definiteness 
        Sigma = L @ L.T

        # mean
        mean = phi @ x
        # variance
        cov_matirx = Sigma

        return MultivariateNormal(mean, cov_matirx)

def OfflinePolicyGradient(env, PolicyNN, n_episodes, lr):
    """
    dostring
    """
    optimizer = optim.Adam(PolicyNN.parameters(), lr = lr)

    cost_history = []
    for ep in range(n_episodes):
        # Sample x0
        x0 = env.reset()
        x_tn = x0.clone()

        # dW = torch.randn(env.N, 2) * np.sqrt(env.dt)
        # 1) 采样一条轨迹
        """
        一次Episode: t=0..T, 共N步(与env.N一致)
        记录 { t_n, x_n, a_n, cost_n, logp_n } + x_N 用于 terminal cost
        其中 cost_n = x_n^T C x_n + a_n^T D a_n (不含 tau*logp,后面再加)
        """
        t_list = []
        x_list = []
        a_list = []
        v_list = []
        cost_list = []
        logp_list = []
        
        for n in range(env.N):
            # forward
            tn = n * env.dt
            # value
            v_n = env.value_function(tn, x_tn)

            tn = torch.tensor([tn], dtype=torch.float32)

            dist = PolicyNN.forward(tn, x_tn)
            a_n = dist.sample().squeeze()

            log_prob = Radon_Nikodym_derivative(dist, env.dist_mu, a_n)

            # cost
            cost_n = x_tn.T @ env.C @ x_tn + a_n.T @ env.D @ a_n

            # 储存
            t_list.append(tn)
            x_list.append(x_tn)
            a_list.append(a_n)
            v_list.append(v_n)
            cost_list.append(cost_n)
            logp_list.append(log_prob)

            # 显示Euler
            # drift = Hx + Ma
            drift = env.H @ x_tn + env.M @ a_n
            # noise = sigma dW
            # noise = env.sigma @ dW[n]
            noise = env.sigma @ (torch.randn(2) * np.sqrt(env.dt))
            # Euler–Maruyama 更新状态
            x_next = x_tn + drift * env.dt + noise

            x_tn = x_next
        x_T = x_tn
        v_T = env.value_function(env.T, x_T)
        v_list.append(v_T)

        # 2) Actor更新
        v = torch.stack(v_list)
        delta_v = torch.diff(v)
        cost = torch.stack(cost_list)
        logp = torch.stack(logp_list)
        cost_history.append(sum(cost))

        inside = (delta_v + (cost + env.tau * logp) * env.dt).unsqueeze(1)
        G_hat = (logp @ inside).sum()
        optimizer.zero_grad()
        G_hat.backward()
        optimizer.step()

        if ep % 100 == 0:
            print(f"Epoch {ep}: Cost = {sum(cost).item():.6f}")

    return cost_history

def x_learn_trajectory(env, actor, x0, dW):
    x_tn = x0.clone()
    x_traj_learn = []
    cost_learn = []
    for n in range(100):
        tn = n * env.dt
        tn = torch.tensor([tn], dtype=torch.float32)
        dist = actor.forward(tn, x_tn)
        a_n = dist.sample().squeeze()

        # cost
        cost_n = x_tn.T @ env.C @ x_tn + a_n.T @ env.D @ a_n
        cost_learn.append(cost_n)

        # drift = Hx + Ma
        drift = env.H @ x_tn + env.M @ a_n
        # noise = sigma dW
        noise = env.sigma @ dW[n]
        # Euler–Maruyama 更新状态
        x_next = x_tn + drift * env.dt + noise
        x_traj_learn.append(x_tn.numpy())
        x_tn = x_next
    # terminal cost
    g_T = x_tn.T @ env.R @ x_tn
    cost_learn.append(g_T)

    x_traj_learn = np.array(x_traj_learn)
    cost_learn = torch.stack(cost_learn)
    cost_learn = torch.cumsum(cost_learn, dim=0)
    return x_traj_learn, cost_learn

def plot_trajectory(x0, x_traj_learn, x_traj_optim):
    # 绘图
    plt.figure(figsize = (12,8))
    plt.plot(x_traj_learn[:,0], x_traj_learn[:,1], label="Learned Offline AC")
    plt.plot(x_traj_optim[:,0], x_traj_optim[:,1], label="Analytical Optimal")
    plt.scatter(x0[0].item(),x0[1].item(), color='red', s=40, label="Start")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Offline AC vs. Analytical Optimal, Start=({x0[0].item()},{x0[1].item()})")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cost(env, x0, cost_learn, cost_optim):
    # 绘图
    plt.figure(figsize = (12,8))
    plt.plot(env.time_grid, cost_learn, label="Learned Offline AC")
    plt.plot(env.time_grid, cost_optim, label="Analytical Optimal")
    plt.xlabel("Time")
    plt.ylabel("Cost")
    plt.title(f"Offline AC vs. Analytical Optimal, Start=({x0[0].item()},{x0[1].item()})")
    plt.legend()
    plt.grid(True)
    plt.show()