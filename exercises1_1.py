from scipy.integrate import solve_ivp
from scipy.integrate import quad
import torch
import numpy as np

class LQR:
    def __init__(self, H, M, C, D, R, sigma, T, N):
        """
        初始化 LQR 类

        Parameters:
        H, M, C, D, R: 线性二次调节器的矩阵
        sigma: 噪声项
        T: 终止时间
        N: 时间步长
        # time_grid: 时间网格 (numpy array)
        """
        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.T = T
        self.N = N
        self.time_grid = torch.linspace(0, T, N+1)
        self.S_values = self.solve_riccati_ode()

    def riccati_ode(self, t, S_flat):
        """Riccati ODE 求解函数，转换为向量形式"""
        S = torch.tensor(S_flat, dtype = torch.float32).reshape(2,2) # 2x2 矩阵
        S_dot = S @ self.M @ torch.linalg.inv(self.D) @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
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
        return self.S_values[nearest_t.item()]
    
    def value_function(self, t, x):
        """计算新的 v(t, x) = x^T S(t) x + ∫[t,T] tr(σσ^T S(r)) dr"""
        # 第一部分：x^T S(t) x
        # print(self.S_values)
        S_t = self.get_nearest_S(t)
        S_t = torch.tensor(S_t, dtype = torch.float32)
        value = x.T @ S_t @ x
        
        # 第二部分：积分项 ∫[t,T] tr(σσ^T S(r)) dr
        def integrand(r):
            S_r = self.get_nearest_S(r)
            return torch.trace(self.sigma @ self.sigma.T @ S_r)
        
        integral, _ = quad(integrand, t, self.T)  # 使用数值积分计算积分项
        value += integral
        return value
    
    def optimal_control(self, t, x):
        """计算最优控制 a(t, x) = -D^(-1) M^T S(t) x"""
        S_t = self.get_nearest_S(t)
        S_t = torch.tensor(S_t, dtype = torch.float32)
        return -torch.linalg.inv(self.D) @ self.M.T @ S_t @ x
    
    def simulate_trajectory(self, x0, dW):
        """
        使用 Euler 方法模拟 LQR 轨迹
        """
        x_traj = [x0.numpy()]
        x_tn = x0
        dt = self.T / self.N
        for n in range(self.N):
            tn = n * dt
            S_tn = self.get_nearest_S(tn)
            S_tn = torch.tensor(S_tn, dtype = torch.float32)

            # a = -D^{-1} M^T S x
            control_a = -torch.linalg.inv(self.D) @self.M.T @ S_tn @ x_tn     # MC
            # print(control_a)

            # drift = Hx + Ma
            drift = self.H @ x_tn + self.M @ control_a
            # print(self.M @ control_a)

            # noise = sigma dW
            noise = self.sigma @ dW[n]

            # explicit Euler scheme
            x_next = x_tn + drift * dt + noise
            x_tn = x_next
            x_traj.append(x_tn.numpy())

        return np.array(x_traj)