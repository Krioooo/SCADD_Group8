import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 添加此处

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============ 1. 问题常量设置 (使用torch定义，然后转为numpy) ============
# problem constants for SDE: dX = (HX + M alpha)dt + sigma dW

H_torch = torch.tensor([[1.0, 1.0],
                        [0.0, 1.0]]) * 0.5

M_torch = torch.tensor([[1.0, 1.0],
                        [0.0, 1.0]])

sigma_torch = torch.eye(2) * 0.5

C_torch = torch.tensor([[1.0, 0.1],
                        [0.1, 1.0]]) * 1.0

D_torch = torch.tensor([[1.0, 0.1],
                        [0.1, 1.0]]) * 0.1

R_torch = torch.tensor([[1.0, 0.3],
                        [0.3, 1.0]]) * 10.0

# 将这些torch张量转为numpy
H = H_torch.numpy()
M_ = M_torch.numpy()
sigma = sigma_torch.numpy()
C = C_torch.numpy()
D = D_torch.numpy()
R = R_torch.numpy()

T = 0.5  # 终止时间
dim = 2  # 状态维度

# 假设价值函数末端项为 G (若题意无此项，可换成零矩阵)
G = np.eye(dim)


# ============ 2. 解Riccati方程，得到 S(t) ============
#
# 方程形式:  -dS/dt = S*H + H^T*S - S*M*D^{-1}M^T*S + C,   S(T)=R (题中也有可能写 S(T)=R )
# 这里假设终端条件是 S(T) = R。若题目指定 S(T)=R，就设 G=R 即可。
#
# 代码演示时，假设 S(T)= G. 如题中要求 S(T)=R，则请自行将 G 换成 R.

def mat2vec(mat):
    return mat.reshape(-1)


def vec2mat(vec):
    return vec.reshape(dim, dim)


def riccati_ode(t, S_vec):
    """方程: -dS/dt = S*H + H^T*S - S*M*D^-1 M^T * S + C."""
    S_mat = vec2mat(S_vec)
    invD = np.linalg.inv(D)
    # 计算 (S H + H^T S - S M D^-1 M^T S + C)
    term = S_mat @ H + H.T @ S_mat - S_mat @ M_ @ invD @ M_.T @ S_mat + C
    # 因为 -dS/dt = term, 所以 dS/dt = -term
    return mat2vec(-term)


# 终端条件: S(T) = G
S_final = mat2vec(G)


def solve_riccati():
    # 用 solve_ivp 从 t=T 往 t=0 方向积分
    sol = solve_ivp(riccati_ode, [T, 0], S_final, t_eval=np.linspace(T, 0, 200))
    # sol.t: 倒序 [T->0], sol.y: shape=(4, len(t_eval)) if dim=2
    # 我们翻转顺序, 让 t_grid 从 0 -> T
    t_grid = sol.t[::-1]
    S_solutions = sol.y[:, ::-1]
    return t_grid, S_solutions


t_grid, S_solutions = solve_riccati()


def S_of_t(t):
    """
    简单查找离 t 最近的网格点, 返回该处的 S(t).
    如果要更高精度可改成插值(线性/样条等).
    """
    idx = (np.abs(t_grid - t)).argmin()
    return vec2mat(S_solutions[:, idx])


# ============ 3. 隐式欧拉的 Monte Carlo 模拟 ============

def simulate_path_implicit(x0, N, M_samples):
    """
    隐式欧拉离散: X_{n+1} = X_n
                   + tau[ H X_{n+1} - M D^-1 M^T S(t_{n+1}) X_{n+1} ]
                   + sigma (W_{n+1} - W_n)

    并在路径上累加代价 (x^T C x + alpha^T R alpha) dt + 末端项.

    x0: 初始状态 (dim=2)
    N:  时间步数
    M_samples: 蒙特卡洛样本数量
    返回: 该 x0 下的平均成本(价值).
    """
    tau = T / N
    invD = np.linalg.inv(D)

    cost_all = np.zeros(M_samples)

    for m in range(M_samples):
        X = np.array(x0)
        cost_path = 0.0

        for n in range(N):
            t_n1 = (n + 1) * tau  # t_{n+1}
            S_n1 = S_of_t(t_n1)

            # 生成随机增量 dW ~ Normal(0, sqrt(tau) * I)
            dW = np.random.randn(dim) * np.sqrt(tau)

            # 系数矩阵 A = I - tau [ H - M D^-1 M^T S(t_{n+1}) ]
            A = np.eye(dim) - tau * (H - M_ @ invD @ M_.T @ S_n1)

            # 右端项 b = X_n + sigma*dW
            b = X + sigma @ dW

            # 解 A X_{n+1} = b
            X_next = np.linalg.solve(A, b)

            # 最优控制 alpha_{n+1} = -D^-1 M^T S(t_{n+1}) X_{n+1}
            alpha = -invD @ M_.T @ S_n1 @ X_next

            # 累加阶段成本
            stage_cost = (X_next @ C @ X_next) + (alpha @ R @ alpha)
            cost_path += stage_cost * tau

            X = X_next

        # 末端成本
        terminal_cost = X @ G @ X
        cost_path += terminal_cost
        cost_all[m] = cost_path

    return np.mean(cost_all)


# ============ 4. 示范：对比一个较大规模参考值, 然后画误差 log-log 图 ============

def main():
    x0 = np.array([1.0, 1.0])  # 测试的初始状态

    # 先用较大 N_ref 和 M_ref 得到一个近似"真值"
    # 可根据机器性能改得更大, 提高参考值精度
    N_ref = 2000
    M_ref = 200000
    V_ref = simulate_path_implicit(x0, N_ref, M_ref)
    print("参考真值 V_ref =", V_ref)

    # (a) 固定 M, 增加 N, 观察误差随 N 的变化
    M_fixed = 10000
    N_list = [2 ** k for k in range(3, 9)]  # 2^3 到 2^8
    err_N = []
    for N_ in N_list:
        V_est = simulate_path_implicit(x0, N_, M_fixed)
        err = abs(V_est - V_ref)
        err_N.append(err)

    plt.figure(figsize=(6, 5))
    plt.loglog(N_list, err_N, 'o-', label='Error vs N')
    plt.xlabel("N (log scale)")
    plt.ylabel("Error (log scale)")
    plt.title("隐式欧拉: 误差随N (固定 M=10000)")
    plt.legend()
    plt.show()

    # (b) 固定 N, 增加 M, 观察误差随 M 的变化
    N_fixed = 1000
    M_list = [2 ** k for k in range(1, 12)]  # 2^1 到 2^11
    err_M = []
    for M_ in M_list:
        V_est = simulate_path_implicit(x0, N_fixed, M_)
        err = abs(V_est - V_ref)
        err_M.append(err)

    plt.figure(figsize=(6, 5))
    plt.loglog(M_list, err_M, 's-', label='Error vs M')
    plt.xlabel("M (log scale)")
    plt.ylabel("Error (log scale)")
    plt.title("隐式欧拉: 误差随M (固定 N=1000)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
