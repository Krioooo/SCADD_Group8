from scipy.integrate import solve_ivp
from scipy.integrate import quad
import torch
import numpy as np
import utils.exercises1_1 as ex1_1
import matplotlib.pyplot as plt

class MonteCarloSDE:
    def __init__(self, H, M, C, D, R, sigma, T, N):
        """
        初始化蒙特卡罗模拟器

        Parameters:
        H, M, C, D, R: 线性二次调节器的矩阵
        sigma: 噪声项
        T: 终止时间
        tau: 时间步长
        N: 时间间隔数
        # M: 蒙特卡罗模拟次数(样本数)
        time_grid: 时间网格
        """
        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.T = T
        self.N = N
        self.tau = T/N
        self.time_grid = torch.linspace(0, T, N + 1)
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
    
    def cal_alpha_coe(self, t):
        """计算控制 a(t, x) = -D^(-1) M^T S(t) """
        S_t = self.get_nearest_S(t)
        S_t = torch.tensor(S_t, dtype = torch.float32)
        return -torch.linalg.inv(self.D) @ self.M.T @ S_t
    
    def simulate(self, x, M_samples):
        """
        使用隐式欧拉离散进行蒙特卡洛模拟 LQR 问题。

        Parameters:
        x: 初始状态x
        M: 蒙特卡洛样本数量
        """
        # 复制初始状态 x0_torch，得到所有样本的初值
        X = x.unsqueeze(0).repeat(M_samples, 1)
        torch.manual_seed(1919)
        dW = torch.randn(M_samples, self.N, 2) * np.sqrt(self.tau)
        eye_dim = torch.eye(2)

        # 价值函数
        V_values = torch.zeros(M_samples)

        for n in range(self.N):
            # print(X)
            tn = n * self.tau
            coe_alpha = self.cal_alpha_coe(tn)
            A = eye_dim - self.tau * (self.H + self.M @ coe_alpha)
            b = X + (dW[:, n, :] @ self.sigma.T)
            X_next = b @ torch.linalg.inv(A).T
            # 阶段成本 = (X_next^T C X_next + alpha^T R alpha) * tau
            alpha = X_next @ coe_alpha.T
            stage_cost_x = torch.sum(X_next @ self.C * X_next, dim = 1)
            stage_cost_a = torch.sum(alpha  @ self.D * alpha,  dim = 1)

            V_values += (stage_cost_x + stage_cost_a) * self.tau
            X = X_next
        
        # 末端成本
        terminal_cost = torch.sum(X @ self.R * X, dim = 1)
        V_values += terminal_cost

        return V_values.mean().item()
    

def error_on_N(H, M, C, D, R, sigma, T, M_samples_fixed, N_step_list, v, t0, x0):
    """
    算error函数(v_Mc - v)
    """
    err_N = []

    for N_step in N_step_list:
        # MClqr = ex1_2.MonteCarloSDE(H, M, C, D, R, sigma, T, N_step)
        MClqr = MonteCarloSDE(H, M, C, D, R, sigma, T, N_step)
        v_est = MClqr.simulate(x0, M_samples_fixed)
        err = abs(v_est - v)
        err_N.append(err)
    
    return err_N

def error_on_M(H, M, C, D, R, sigma, T, N_fixed, M_samples_list, v, t0, x0):
    err_M = []

    MClqr = MonteCarloSDE(H, M, C, D, R, sigma, T, N_fixed)

    for M_samples in M_samples_list:
        v_est = MClqr.simulate(x0, M_samples)
        err = abs(v_est - v)
        err_M.append(err)
    
    return err_M

def loglog_plot(x, y, fixed_para, change_para):
    plt.figure(figsize = (12, 4))
    plt.loglog(x, y, 'o-', label = f'Weak Error')
    # 添加参考线，斜率为 -1
    if change_para == 'N':
        C = y[0] * x[0]  # 选择合适的比例常数 C
        ref_line = C * 1/ np.array(x)  # 计算参考线的值
        plt.loglog(x, ref_line, '--', label = 'Slope -1')
    else:
        C = y[0] * x[0]  # 选择合适的比例常数 C
        ref_line = C * 1/np.sqrt(np.array(x))  # 计算参考线的值
        plt.loglog(x, ref_line, '--', label = 'Slope -1/2')
    plt.xlabel(f"{change_para} (log)")
    plt.ylabel("Error (log)")
    plt.title(f"Implicit Euler Scheme: error over {change_para} (fixed {fixed_para})")
    plt.legend()
    plt.grid()
    plt.show()

def compute_error_slope(x_values, y_values):
        """
        在 log-log 坐标下，对 (x_values, y_values) 做线性回归，返回斜率。
        """
        logx = np.log(x_values)
        logy = np.log(y_values)
        slope, _ = np.polyfit(logx, logy, 1)
        return slope