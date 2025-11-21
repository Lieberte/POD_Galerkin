"""
POD分解核心模块
"""
import numpy as np
import pandas as pd
from pathlib import Path

class PODDecomposer:
    """POD分解器"""

    def __init__(self):
        self.Phi = None
        self.Sigma = None
        self.mean_field = None
        self.energy_ratio = None
        self.cumulative_energy = None

    def fit(self, snapshot_matrix, n_modes=None):
        """
        执行POD分解

        Parameters:
        -----------
        snapshot_matrix : ndarray, shape (n_cells, n_snapshots)
            快照矩阵
        n_modes : int, optional
            保留模态数，None表示全部

        Returns:
        --------
        self : PODDecomposer
        """
        print("开始POD分解...")
        print(f"快照矩阵形状: {snapshot_matrix.shape}")

        # 计算平均场
        self.mean_field = np.mean(snapshot_matrix, axis=1, keepdims=True)

        # 中心化
        X = snapshot_matrix - self.mean_field

        # SVD
        Phi, Sigma, VT = np.linalg.svd(X, full_matrices=False)

        # 能量分析
        energy = Sigma ** 2
        total_energy = np.sum(energy)
        self.energy_ratio = energy / total_energy
        self.cumulative_energy = np.cumsum(self.energy_ratio)

        # 截断
        if n_modes is None or n_modes > len(Sigma):
            n_modes = len(Sigma)

        self.Phi = Phi[:, :n_modes]
        self.Sigma = Sigma[:n_modes]

        print(f"保留{n_modes}个模态，累计能量: {self.cumulative_energy[n_modes-1]:.4%}")

        return self

    def transform(self, snapshot_matrix):
        """
        将快照投影到POD空间

        Parameters:
        -----------
        snapshot_matrix : ndarray, shape (n_cells, n_snapshots)

        Returns:
        --------
        coefficients : ndarray, shape (n_modes, n_snapshots)
            时间系数
        """
        X = snapshot_matrix - self.mean_field
        return self.Phi.T @ X

    def inverse_transform(self, coefficients):
        """
        从POD系数重构场

        Parameters:
        -----------
        coefficients : ndarray, shape (n_modes, n_snapshots)

        Returns:
        --------
        reconstructed : ndarray, shape (n_cells, n_snapshots)
        """
        return self.mean_field + self.Phi @ coefficients

    def save(self, save_dir):
        """保存POD模型"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # POD模态
        pd.DataFrame(
            self.Phi,
            columns=[f'mode_{i+1}' for i in range(self.Phi.shape[1])]
        ).to_csv(save_dir / 'POD_modes.csv', index_label='cell_id', float_format='%.8e')

        # 奇异值和能量
        pd.DataFrame({
            'mode': range(1, len(self.Sigma) + 1),
            'singular_value': self.Sigma,
            'energy_ratio': self.energy_ratio[:len(self.Sigma)],
            'cumulative_energy': self.cumulative_energy[:len(self.Sigma)]
        }).to_csv(save_dir / 'singular_values.csv', index=False, float_format='%.8e')

        # 平均场
        pd.DataFrame(
            {'mean_field': self.mean_field.flatten()}
        ).to_csv(save_dir / 'mean_field.csv', index_label='cell_id', float_format='%.8e')

        print(f"POD模型已保存到: {save_dir}")

    @classmethod
    def load(cls, save_dir):
        """加载POD模型"""
        save_dir = Path(save_dir)

        decomposer = cls()
        decomposer.Phi = pd.read_csv(save_dir / 'POD_modes.csv', index_col=0).values
        decomposer.mean_field = pd.read_csv(save_dir / 'mean_field.csv', index_col=0).values

        df_sigma = pd.read_csv(save_dir / 'singular_values.csv')
        decomposer.Sigma = df_sigma['singular_value'].values
        decomposer.energy_ratio = df_sigma['energy_ratio'].values
        decomposer.cumulative_energy = df_sigma['cumulative_energy'].values

        print(f"POD模型已加载，模态数: {decomposer.Phi.shape[1]}")
        return decomposer