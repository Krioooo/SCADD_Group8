import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        self.S_values = self.solve_riccati_ode()
    
    def riccati_ode(self, t, S_flat):
        """Riccati ODE 求解函数，转换为向量形式"""
        S = torch.tensor(S_flat, dtype=torch.float32).reshape(2,2) # 2x2 矩阵
        D_eff_inv = torch.linalg.inv(self.D_eff)
        S_dot = S.T @ self.M @ torch.linalg.inv(D_eff_inv) @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
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

    def simulate_one_trajectory(self, x0, dW):
        """
        使用 Euler-Maruyama 模拟一条轨迹，
        返回 [(t, x, cost, x_next), ..., (t_T, x_T, None, None)]
        其中 cost = x^T C x + a^T D a + tau ln p(a|x) （未乘 dt)
        SoftLQR: 带熵正则的 LQR 问题
        状态方程： dX = (H X + M a )dt + sigma dW
        成本： ∫ (x^T C x + a^T D a + tau ln p(a|x)) dt + x_T^T R x_T
        控制策略： a ~ N(mean, cov)
        其中 mean = -D_eff^{-1} M^T S(t)x,  cov = tau * D_eff,
        D_eff = D + (tau/(2*gamma^2)) I.
        """
        data = []
        x_tn = x0.clone()

        for n in range(self.N):
            tn = n * self.dt
            S_tn = self.get_nearest_S(tn)
            S_tn = torch.tensor(S_tn, dtype = torch.float32)
            # mean
            mean= -torch.linalg.inv(self.D_eff) @ self.M.T @ S_tn @ x_tn 
            # covarian
            cov = self.tau * self.D_eff
            # distribution
            dist = MultivariateNormal(mean, cov)
            a_n = dist.sample()
            # log_prob
            log_prob = dist.log_prob(a_n).item()
            # cost
            cost_n = (x_tn.T @ self.C @ x_tn + a_n.T @ self.D @ a_n).item() + self.tau * log_prob

            # drift = Hx + Ma
            drift = self.H @ x_tn + self.M @ a_n
            # noise = sigma dW
            noise = self.sigma @ dW[n]
            # Euler–Maruyama 更新状态
            x_next = x_tn + drift * self.dt + noise

            data.append((tn, x_tn, cost_n, x_next))
            x_tn = x_next

        terminal_cost = (x_tn.T @ self.R @ x_tn).item()
        data.append((tn, x_tn, terminal_cost, None))
        return data
    
class CriticPolicy(nn.Module):
    def __init__(self, hidden_dim = 512):
        super(CriticPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 输入维度为 3：[t, x], x_dim = 2
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, t, x):
        if isinstance(t, torch.Tensor):
            input_para = torch.cat([t, x], dim = 1)
        else:
            input_para = torch.cat([torch.tensor([t], dtype=torch.float32), x])
        v_learn = self.net(input_para).squeeze()
        return v_learn
    
class ValueNN(nn.Module):
    def __init__(self, hidden_dim = 512):
        super(ValueNN, self).__init__()

        self.l1 = nn.Linear(1, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.matrix = nn.Linear(hidden_dim,4) # =>2x2
        self.offset = nn.Linear(hidden_dim,1)
        self.relu = nn.ReLU()
        # self.matrix = nn.Sequential(
        #     nn.Linear(1, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 4),       # =>2x2
        # )
        # self.offset = nn.Sequential(
        #     nn.Linear(1, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1),
        # )

    def forward(self, t, x):
        """
        t: (batch,1)
        x: (batch,2)
        => v(t,x) = x^T [Q(t)] x + offset(t)
        Q(t) = sym( net(t) ) + diag(1e-3)
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32)

        h = self.relu(self.l1(t))
        h = self.relu(self.l2(h))
        matrix = self.matrix(h)      # (batch,4)
        offset = self.offset(h)      # (batch,1)

        # matrix = self.matrix(t)      # (batch,4)
        # offset = self.offset(t)      # (batch,1)

        # reshape => 2x2
        matrix_2x2 = matrix.view(-1,2,2)
        matrix_sym = 0.5*(matrix_2x2 + matrix_2x2.transpose(1,2))
        # Make the matrix positive definite(存疑)
        matrix_pd = matrix_sym + 1e-3 * torch.eye(2)
        
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 2, 1)
        quad_term = torch.bmm(torch.bmm(x_reshaped.transpose(1,2), matrix_pd), x_reshaped).view(-1)

        return quad_term + offset.squeeze(-1)
    
    def value_function(self, t, x):
        t = torch.tensor([t], dtype=torch.float32)
        x = x.view(1,2).float()
        return self.forward(t, x).item()

def OfflineCriticAlgorithm_1(env, policy, n_episodes, lr):
    """
    离线训练价值网络：对每个时刻 n，计算累计成本 target_n，
    target_n = (∑_{k=n}^{N-1} (cost_k · dt)) + (x_T^T R x_T).
    损失： L(η) = ∑_{n=0}^{N-1} (Vθ(t_n,x_n) - target_n)^2.
    """
    optimizer = optim.Adam(policy.parameters(), lr = lr)
    mse_loss = nn.MSELoss(reduction = 'sum')

    # Sample x0
    x0 = env.reset()
    # torch.manual_seed(1234)
    dW = torch.randn(env.N, 2) * np.sqrt(env.dt)

    data = env.simulate_one_trajectory(x0, dW)

    # 计算每个时刻的累计成本
    cumulative_costs = []
    for n in range(env.N):
        cum = 0.0
        for k in range(n, env.N):
            cum += data[k][2] * env.dt
        cum += data[env.N][2]
        cumulative_costs.append(cum)
    target = torch.tensor(cumulative_costs, dtype = torch.float32)

    # 计算损失L(η)
    ts = [sample[0] for sample in data[:-1]]
    xs = [sample[1] for sample in data[:-1]]
    ts_tensor = torch.tensor(ts, dtype=torch.float32).unsqueeze(1)
    xs_tensor = torch.stack(xs).float()

    loss_history = []
    for ep in range (n_episodes):
        optimizer.zero_grad()
        v_pred = policy.forward(ts_tensor, xs_tensor)
        loss = mse_loss(v_pred, target)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if ep % 100 == 0:
            print(f"Epoch {ep}: Loss = {loss.item():.6f}")
        
    return loss_history

def OfflineCriticAlgorithm(env, ValueNN, n_episodes, lr):
    """
    离线训练价值网络：对每个时刻 n，计算累计成本 target_n，
    target_n = (∑_{k=n}^{N-1} (cost_k · dt)) + (x_T^T R x_T).
    损失： L(η) = ∑_{n=0}^{N-1} (Vθ(t_n,x_n) - target_n)^2.
    for ep in range(episodes):
          1) sample data from env
          2) build Lhat(eta)
          3) one gradient step
    """
    optimizer = optim.Adam(ValueNN.parameters(), lr = lr)
    mse_loss = nn.MSELoss(reduction = 'sum')

    loss_history = []
    for ep in range(n_episodes):
        # Sample x0
        x0 = env.reset()
        torch.manual_seed(1234)
        dW = torch.randn(env.N, 2) * np.sqrt(env.dt)

        data = env.simulate_one_trajectory(x0, dW)

        # 计算每个时刻的累计成本
        cumulative_costs = []
        for n in range(env.N):
            cum = 0.0
            for k in range(n, env.N):
                cum += data[k][2] * env.dt
            cum += data[env.N][2]
            cumulative_costs.append(cum)
        target = torch.tensor(cumulative_costs, dtype = torch.float32)

        # 计算损失L(η)
        ts = [sample[0] for sample in data[:-1]]
        xs = [sample[1] for sample in data[:-1]]
        ts_tensor = torch.tensor(ts, dtype=torch.float32).unsqueeze(1)
        xs_tensor = torch.stack(xs).float()

        optimizer.zero_grad()
        v_pred = ValueNN.forward(ts_tensor, xs_tensor)
        loss = mse_loss(v_pred, target)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if ep % 100 == 0:
            print(f"Epoch {ep}: Loss = {loss.item():.6f}")
                
    return loss_history


def plot_critic_loss(loss_history):
    """
    绘制训练损失曲线
    """
    plt.figure(figsize = (12,4))
    plt.plot(loss_history, label='Critic Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Critic Loss')
    plt.title('Offline Critic Training (Cumulative Temporal Difference Target)')
    plt.grid(True)
    plt.legend()
    plt.show()

def find_maximum_error(env, policy, t_test, x_range):
    """
    # 在指定的时间点 t in {0, 1/6, 2/6, 0.5} 
    # 与网格上比较离线 Critic 学到的 V(t,x) 与理论值
    """
    max_error = 0.0
    for t_value in t_test:
        S_t = env.get_nearest_S(t_value)
        for x1 in x_range:
            for x2 in x_range:
                x_vector = torch.tensor([x1, x2], dtype = torch.float32)
                # v_learn = policy.forward(t_value, x_vector)
                v_learn = policy.value_function(t_value, x_vector)
                v_theoretical = env.value_function(t_value, x_vector).item()
                max_error = max(max_error, abs(v_learn - v_theoretical))
    print(f"Max error on specified grid: {max_error:.2f}")

