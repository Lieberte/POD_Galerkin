"""
POD模型验证
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class PODValidator:
    """POD模型验证器"""

    @staticmethod
    def validate(pod_decomposer, surrogate, valid_matrix, valid_params, output_dir=None,
                 space_type='normalized', scalers=None, variable_indices=None):
        """
        验证POD+代理模型

        Parameters:
        -----------
        pod_decomposer : PODDecomposer
            训练好的POD分解器
        surrogate : PODSurrogate
            训练好的代理模型
        valid_matrix : ndarray, shape (n_cells, n_valid)
            验证集真实场
        valid_params : ndarray, shape (n_valid,)
            验证参数
        output_dir : Path, optional
            输出目录
        space_type : str
            'normalized' 或 'physical'，指定验证空间
        scalers : dict, optional
            如果space_type='physical'且POD在归一化空间训练，需要提供scalers进行反归一化
        variable_indices : dict, optional
            变量索引信息，用于多变量分离验证

        Returns:
        --------
        errors : DataFrame
            误差统计
        """
        print(f"开始验证 ({space_type} space)...")

        pred_coeffs = surrogate.predict(valid_params)
        pred_matrix = pod_decomposer.inverse_transform(pred_coeffs.T)

        # 如果需要反归一化到物理空间
        if space_type == 'physical' and scalers is not None and variable_indices is not None:
            pred_matrix_physical = np.zeros_like(pred_matrix)
            for var_name, idx_info in variable_indices.items():
                start = idx_info['start']
                end = idx_info['end']

                pred_var_flat = pred_matrix[start:end, :].flatten().reshape(-1, 1)
                pred_matrix_physical[start:end, :] = scalers[var_name].inverse_transform(
                    pred_var_flat
                ).reshape((end - start, -1))

            pred_matrix = pred_matrix_physical

        errors = []
        for i, param in enumerate(valid_params):
            true_field = valid_matrix[:, i]
            pred_field = pred_matrix[:, i]
            diff = true_field - pred_field

            errors.append({
                'param': param,
                'relative_L2': np.linalg.norm(diff) / np.linalg.norm(true_field),
                'max_absolute': np.max(np.abs(diff)),
                'mean_absolute': np.mean(np.abs(diff)),
                'max_relative': np.max(np.abs(diff / (true_field + 1e-10)))
            })

            print(f"  参数 {param}: L2误差 = {errors[-1]['relative_L2']*100:.4f}%")

        df_errors = pd.DataFrame(errors)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            df_errors.to_csv(output_dir / 'validation_errors.csv', index=False, float_format='%.8e')

            pd.DataFrame(
                pred_matrix,
                columns=[f'param_{p}' for p in valid_params]
            ).to_csv(output_dir / 'predicted_fields.csv', index_label='cell_id', float_format='%.8e')

            PODValidator.plot_errors(df_errors, output_dir / 'error_plot.png')

            PODValidator.plot_prediction_scatter(
                valid_matrix, pred_matrix, valid_params,
                output_dir / 'prediction_scatter.png',
                title_suffix=f"({space_type.capitalize()} Space)"
            )

        # 如果有变量索引信息，分变量验证
        if variable_indices is not None:
            PODValidator.validate_by_variable(
                valid_matrix, pred_matrix, valid_params,
                variable_indices, output_dir, space_type
            )

        return df_errors

    @staticmethod
    def validate_by_variable(valid_matrix, pred_matrix, valid_params,
                            variable_indices, output_dir, space_type):
        """
        按变量分离验证

        Parameters:
        -----------
        valid_matrix : ndarray, shape (n_cells, n_valid)
            真实场
        pred_matrix : ndarray, shape (n_cells, n_valid)
            预测场
        valid_params : ndarray
            验证参数
        variable_indices : dict
            变量索引信息
        output_dir : Path
            输出目录
        space_type : str
            空间类型标识
        """
        print(f"\n按变量分离验证 ({space_type} space)...")

        for var_name, idx_info in variable_indices.items():
            print(f"  验证变量: {var_name}")
            start = idx_info['start']
            end = idx_info['end']

            pred_var = pred_matrix[start:end, :]
            valid_var = valid_matrix[start:end, :]

            errors_var = []
            for i, param in enumerate(valid_params):
                true_field = valid_var[:, i]
                pred_field = pred_var[:, i]
                diff = true_field - pred_field

                errors_var.append({
                    'param': param,
                    'relative_L2': np.linalg.norm(diff) / np.linalg.norm(true_field),
                    'max_absolute': np.max(np.abs(diff)),
                    'mean_absolute': np.mean(np.abs(diff)),
                    'max_relative': np.max(np.abs(diff / (true_field + 1e-10)))
                })

            var_dir = output_dir / var_name
            var_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(errors_var).to_csv(
                var_dir / 'validation_errors.csv',
                index=False,
                float_format='%.8e'
            )

            pd.DataFrame(
                pred_var,
                columns=[f'param_{p}' for p in valid_params]
            ).to_csv(
                var_dir / 'predicted_fields.csv',
                index_label='cell_id',
                float_format='%.8e'
            )

            PODValidator.plot_errors(
                pd.DataFrame(errors_var),
                var_dir / 'error_plot.png'
            )

            PODValidator.plot_prediction_scatter(
                valid_var, pred_var, valid_params,
                var_dir / 'prediction_scatter.png',
                title_suffix=f"({space_type.capitalize()} Space - {var_name})"
            )

    @staticmethod
    def plot_errors(df_errors, save_path):
        """绘制误差图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(df_errors['param'], df_errors['relative_L2']*100, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_ylabel('Relative L2 Error (%)', fontsize=12)
        axes[0, 0].set_title('Relative L2 Error', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(df_errors['param'], df_errors['max_absolute'], 's-', color='red', linewidth=2, markersize=8)
        axes[0, 1].set_ylabel('Max Absolute Error', fontsize=12)
        axes[0, 1].set_title('Maximum Absolute Error', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(df_errors['param'], df_errors['mean_absolute'], '^-', color='green', linewidth=2, markersize=8)
        axes[1, 0].set_ylabel('Mean Absolute Error', fontsize=12)
        axes[1, 0].set_xlabel('Parameter', fontsize=12)
        axes[1, 0].set_title('Mean Absolute Error', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(df_errors['param'], df_errors['max_relative']*100, 'd-', color='purple', linewidth=2, markersize=8)
        axes[1, 1].set_ylabel('Max Relative Error (%)', fontsize=12)
        axes[1, 1].set_xlabel('Parameter', fontsize=12)
        axes[1, 1].set_title('Maximum Relative Error', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_prediction_scatter(true_matrix, pred_matrix, valid_params, save_path, title_suffix=""):
        """绘制预测值 vs 真实值散点图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        true_flat = true_matrix.flatten()
        pred_flat = pred_matrix.flatten()

        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        n_points = len(true_flat)
        if n_points > 50000:
            indices = np.random.choice(n_points, 50000, replace=False)
            true_sample = true_flat[indices]
            pred_sample = pred_flat[indices]
        else:
            true_sample = true_flat
            pred_sample = pred_flat

        axes[0].scatter(true_sample, pred_sample, alpha=0.3, s=1, color='steelblue')

        min_val = min(true_flat.min(), pred_flat.min())
        max_val = max(true_flat.max(), pred_flat.max())
        axes[0].plot([min_val, max_val], [min_val, max_val],
                     'r--', linewidth=2, label='Perfect Prediction (y=x)')

        x_range = np.linspace(min_val, max_val, 100)
        axes[0].plot(x_range, x_range * 1.05, 'g--', linewidth=1.5, alpha=0.7, label='±5% Error')
        axes[0].plot(x_range, x_range * 0.95, 'g--', linewidth=1.5, alpha=0.7)
        axes[0].plot(x_range, x_range * 1.10, 'orange', linestyle='--', linewidth=1.5, alpha=0.7, label='±10% Error')
        axes[0].plot(x_range, x_range * 0.90, 'orange', linestyle='--', linewidth=1.5, alpha=0.7)

        axes[0].set_xlabel('True Values', fontsize=14)
        axes[0].set_ylabel('Predicted Values', fontsize=14)
        axes[0].set_title(f'Prediction vs Truth {title_suffix}\nR² = {r2:.4f}', fontsize=16)
        axes[0].legend(fontsize=11, loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal', adjustable='box')

        errors_by_case = []
        for i in range(len(valid_params)):
            relative_errors = (pred_matrix[:, i] - true_matrix[:, i]) / (true_matrix[:, i] + 1e-10) * 100
            errors_by_case.append(relative_errors)

        bp = axes[1].boxplot(errors_by_case, labels=[f'{p}' for p in valid_params],
                             patch_artist=True, showmeans=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[1].axhline(y=5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='±5%')
        axes[1].axhline(y=-5, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[1].axhline(y=10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='±10%')
        axes[1].axhline(y=-10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

        axes[1].set_xlabel('Validation Parameter', fontsize=14)
        axes[1].set_ylabel('Relative Error (%)', fontsize=14)
        axes[1].set_title(f'Error Distribution by Case {title_suffix}', fontsize=16)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()