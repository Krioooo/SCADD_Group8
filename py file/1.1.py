import numpy as np
from scipy.integrate import solve_ivp

class LQR:
    def __init__(self, H, M, C, D, R, sigma, T, time_grid):
        """
        初始化 LQR 类

        Parameters:
        H, M, C, D, R: 线性二次调节器的矩阵
        sigma: 噪声项
        T: 终止时间
        time_grid: 时间网格 (numpy array)
        """
        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.T = T
        self.time_grid = time_grid
        self.S_values = self.solve_riccati_ode()

    def riccati_ode(self, t, S_flat):
        """Riccati ODE 求解函数，转换为向量形式"""
        S = S_flat.reshape(2, 2)  # 2x2 矩阵
        S_dot = S @ self.M @ np.linalg.inv(self.D) @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
        return S_dot.flatten()

    def solve_riccati_ode(self):
        """使用 solve_ivp 求解 Riccati ODE"""
        S_T = self.R.flatten()  # 终止条件 S(T) = R
        sol = solve_ivp(self.riccati_ode, [self.T, 0], S_T, t_eval=self.time_grid[::-1])  # 逆向求解
        S_matrices = sol.y.T[::-1].reshape(-1, 2, 2)  # 转换回矩阵格式
        return dict(zip(self.time_grid, S_matrices))

    def get_nearest_S(self, t):
        """找到最近的 S(t)"""
        nearest_t = self.time_grid[np.argmin(np.abs(self.time_grid - t))]
        return self.S_values[nearest_t]

    def value_function(self, t, x):
        """计算 v(t, x) = x^T S(t) x"""
        S_t = self.get_nearest_S(t)
        return np.dot(x.T, S_t @ x)

    def optimal_control(self, t, x):
        """计算最优控制 a(t, x) = -D^(-1) M^T S(t) x"""
        S_t = self.get_nearest_S(t)
        return -np.linalg.inv(self.D) @ self.M.T @ S_t @ x

# 示例参数
H = np.array([[0, 1], [-1, 0]])
M = np.array([[0], [1]])
C = np.eye(2)
D = np.array([[1]])
R = np.eye(2)
sigma = np.eye(2) * 0.1
T = 1.0
time_grid = np.linspace(0, T, 100)

# 实例化 LQR 类
lqr = LQR(H, M, C, D, R, sigma, T, time_grid)

# 计算示例
t_test = 0.5
x_test = np.array([1, 0])

print("Value function v(t, x):", lqr.value_function(t_test, x_test))
print("Optimal control a(t, x):", lqr.optimal_control(t_test, x_test))
