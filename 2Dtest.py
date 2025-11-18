import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ==============================================================================
# 0. 问题设定：高保真模型 (High-Fidelity Model - HFM)
# 在真实场景中，这将是一个昂贵的CFD求解器 (如Fluent, OpenFOAM)。
# 这里我们用一个确定的数学函数来模拟它，以演示ROM的原理。
# ==============================================================================

def generate_hfm_snapshot(alpha, nx=80, ny=40):
    """
    生成一个高保真模型的快照（二维速度场 U）。
    这个函数模拟一个CFD求解器在给定入口流速alpha下的稳态解。

    参数:
    alpha (float): 边界条件，这里代表入口的平均流速。
    nx (int): x方向的网格点数。
    ny (int): y方向的网格点数。

    返回:
    numpy.ndarray: 一个 (ny, nx) 的数组，代表U速度分量。
    """
    # 创建网格
    x = np.linspace(0, 2, nx)
    y = np.linspace(-0.5, 0.5, ny)
    X, Y = np.meshgrid(x, y)

    # 基础流：管道中的抛物线流
    # U(y) = U_max * (1 - (y/h)^2)，其中 U_max = 1.5 * alpha, h = 0.5
    base_flow = 1.5 * alpha * (1 - (Y / 0.5) ** 2)

    # 扰动流：一个依赖于alpha和空间位置的复杂扰动
    # 这使得流场不是简单的线性变化，为POD提供了提取特征的空间
    perturbation = 0.1 * alpha * np.sin(np.pi * X) * np.cos(1.5 * np.pi * Y) * np.exp(-X / 2)

    # 总流场 = 基础流 + 扰动流
    velocity_field = base_flow + perturbation

    return velocity_field


print("--- 降阶模型 (ROM) 演示 ---")

# ==============================================================================
# 1. 离线阶段 (Offline Stage): 数据生成与模型构建
# ==============================================================================
print("\n[1] 开始离线阶段...")

# 定义参数空间
# 我们将为这10个不同的入口流速生成快照
alphas_offline = np.linspace(1.0, 10.0, 10)
print(f"  - 将在以下alpha值处生成快照: {np.round(alphas_offline, 2)}")

# 收集快照
snapshots_list = []
nx, ny = 80, 40
for alpha in alphas_offline:
    snapshot = generate_hfm_snapshot(alpha, nx, ny)
    # 将二维快照展平为一维向量，并存入列表
    snapshots_list.append(snapshot.flatten())

# 将快照列表转换为一个大的“快照矩阵”S
# 矩阵的每一列是一个展平的快照
# 形状为 (n_points, n_snapshots)，即 (nx*ny, 10)
snapshot_matrix = np.array(snapshots_list).T
n_points, n_snapshots = snapshot_matrix.shape
print(f"  - 已生成 {n_snapshots} 个快照，快照矩阵形状: {snapshot_matrix.shape}")

# --- 执行主成分分析 (POD) ---
print("  - 正在执行主成分分析 (POD)...")

# 步骤 a: 计算平均流场
mean_flow = np.mean(snapshot_matrix, axis=1)

# 步骤 b: 计算波动场 (减去平均值)
fluctuation_matrix = snapshot_matrix - mean_flow[:, np.newaxis]

# 步骤 c: 计算协方差矩阵 K (小矩阵)
# K 的大小是 (n_snapshots, n_snapshots)，这里是 (10, 10)
# 远比计算 (n_points, n_points) 的协方差矩阵要快
K = fluctuation_matrix.T @ fluctuation_matrix

# 步骤 d: 求解协方差矩阵的特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eigh(K)

# 步骤 e: 排序，从大到小
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# 步骤 f: 计算POD模态 (基函数)
# 模态是空间函数，所以其维度应为 n_points
# 公式: phi_k = (1/sqrt(lambda_k)) * S' * v_k
# S' 是波动矩阵, v_k 是K的特征向量, lambda_k 是特征值
pod_bases = fluctuation_matrix @ eigenvectors
# 对模态进行归一化
pod_bases = pod_bases / np.linalg.norm(pod_bases, axis=0)

# 步骤 g: 投影原始快照到POD模态上，得到POD系数
# 系数矩阵的形状为 (n_snapshots, n_modes)
pod_coefficients = fluctuation_matrix.T @ pod_bases

# --- 降阶与模型选择 ---
# 决定保留多少个模态。我们可以通过分析特征值的能量来决定。
total_energy = np.sum(eigenvalues)
cumulative_energy = np.cumsum(eigenvalues) / total_energy
print(f"  - POD模态的累积能量: {np.round(cumulative_energy, 4)}")

# 选择保留99.99%能量所需的最少模态数
n_modes_to_keep = np.where(cumulative_energy >= 0.9999)[0][0] + 1
print(f"  - 选择保留 {n_modes_to_keep} 个模态 (能量占比 > 99.99%)")

# 截断POD模态和系数
rom_bases = pod_bases[:, :n_modes_to_keep]
rom_coefficients = pod_coefficients[:, :n_modes_to_keep]

print("... 离线阶段完成。")

# ==============================================================================
# 2. 在线阶段 (Online Stage): 快速预测
# ==============================================================================
print("\n[2] 开始在线阶段...")

# --- 构建插值模型 ---
# 对于每一个保留的模态，我们都创建一个插值函数
# 这个函数学习 alpha 和该模态系数之间的关系
interpolators = []
for i in range(n_modes_to_keep):
    # 使用三次样条插值
    interp_func = interp1d(alphas_offline, rom_coefficients[:, i], kind='cubic', fill_value="extrapolate")
    interpolators.append(interp_func)
print(f"  - 已为 {n_modes_to_keep} 个模态系数构建了插值模型。")


def predict_rom(alpha_new):
    """
    使用已构建的ROM快速预测新alpha下的流场。
    """
    # 步骤 a: 使用插值器预测新alpha下的POD系数
    predicted_coeffs = np.array([interp(alpha_new) for interp in interpolators])

    # 步骤 b: 重构流场
    # z_approx = mean_flow + sum(coeff_k * base_k)
    reconstructed_flat = mean_flow + rom_bases @ predicted_coeffs

    # 将一维向量重塑为二维流场
    return reconstructed_flat.reshape(ny, nx)


# ==============================================================================
# 3. 验证与可视化
# ==============================================================================
print("\n[3] 验证与可视化...")

# 选择一个 *新的* alpha值，它不在我们离线训练的集合中
alpha_new = 6.7
print(f"  - 目标: 预测 alpha = {alpha_new} 时的流场。")

# --- 获取ROM预测解 (非常快) ---
import time

start_time = time.time()
rom_prediction = predict_rom(alpha_new)
end_time = time.time()
print(f"  - ROM预测耗时: {1000 * (end_time - start_time):.4f} 毫秒")

# --- 获取HFM真实解 (用于对比，模拟昂贵计算) ---
start_time = time.time()
hfm_truth = generate_hfm_snapshot(alpha_new, nx, ny)
end_time = time.time()
print(f"  - HFM'真实'计算耗时 (模拟): {1000 * (end_time - start_time):.4f} 毫秒 (真实CFD会是数分钟到数小时)")

# --- 计算误差 ---
error_field = np.abs(hfm_truth - rom_prediction)
relative_error = np.linalg.norm(error_field) / np.linalg.norm(hfm_truth)
print(f"  - ROM预测的相对误差 (L2范数): {relative_error:.6%}")

# --- 可视化 ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f'ROM 预测 vs. HFM 真实解 (alpha = {alpha_new})', fontsize=16)

# 绘图范围
v_min = hfm_truth.min()
v_max = hfm_truth.max()

# 1. HFM 真实解
ax1 = axes[0]
im1 = ax1.imshow(hfm_truth, extent=[0, 2, -0.5, 0.5], origin='lower', cmap='viridis', vmin=v_min, vmax=v_max,
                 aspect='auto')
ax1.set_title('高保真模型 (HFM) "真实"解')
ax1.set_ylabel('Y')
fig.colorbar(im1, ax=ax1, label='U 速度')

# 2. ROM 预测解
ax2 = axes[1]
im2 = ax2.imshow(rom_prediction, extent=[0, 2, -0.5, 0.5], origin='lower', cmap='viridis', vmin=v_min, vmax=v_max,
                 aspect='auto')
ax2.set_title(f'降阶模型 (ROM) 预测解 (使用 {n_modes_to_keep} 个模态)')
ax2.set_ylabel('Y')
fig.colorbar(im2, ax=ax2, label='U 速度')

# 3. 绝对误差
ax3 = axes[2]
im3 = ax3.imshow(error_field, extent=[0, 2, -0.5, 0.5], origin='lower', cmap='Reds', aspect='auto')
ax3.set_title('绝对误差 |HFM - ROM|')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
fig.colorbar(im3, ax=ax3, label='误差大小')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()