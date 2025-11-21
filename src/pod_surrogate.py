"""
POD系数代理模型
"""
import numpy as np
import pickle
from pathlib import Path
from scipy.interpolate import Rbf, CubicSpline, interp1d

class PODSurrogate:
    """POD系数插值代理模型"""

    def __init__(self, method='rbf', rbf_function='cubic'):
        """
        Parameters:
        -----------
        method : str
            插值方法: 'rbf', 'cubic', 'linear'
        rbf_function : str
            RBF核函数: 'multiquadric', 'cubic', 'thin_plate', 'gaussian'
        """
        self.method = method
        self.rbf_function = rbf_function
        self.surrogates = []
        self.train_params = None
        self.n_modes = None

    def fit(self, train_params, train_coefficients):
        """
        训练代理模型

        Parameters:
        -----------
        train_params : ndarray, shape (n_train,)
            训练参数（如流量）
        train_coefficients : ndarray, shape (n_train, n_modes)
            POD系数

        Returns:
        --------
        self : PODSurrogate
        """
        self.train_params = train_params
        self.n_modes = train_coefficients.shape[1]

        print(f"构建{self.n_modes}个模态的代理模型 (方法: {self.method})")

        self.surrogates = []
        for i in range(self.n_modes):
            if self.method == 'rbf':
                surrogate = Rbf(train_params, train_coefficients[:, i],
                               function=self.rbf_function)
            elif self.method == 'cubic':
                surrogate = CubicSpline(train_params, train_coefficients[:, i])
            elif self.method == 'linear':
                surrogate = interp1d(train_params, train_coefficients[:, i],
                                    kind='linear', fill_value='extrapolate')
            else:
                raise ValueError(f"不支持的方法: {self.method}")

            self.surrogates.append(surrogate)

        print("代理模型训练完成")
        return self

    def predict(self, params):
        """
        预测POD系数

        Parameters:
        -----------
        params : float or ndarray
            新参数值

        Returns:
        --------
        coefficients : ndarray, shape (n_cases, n_modes)
        """
        params = np.atleast_1d(params)
        coeffs = np.zeros((len(params), self.n_modes))

        for i, surrogate in enumerate(self.surrogates):
            coeffs[:, i] = surrogate(params)

        return coeffs

    def save(self, save_path):
        """保存代理模型"""
        with open(save_path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'rbf_function': self.rbf_function,
                'train_params': self.train_params,
                'n_modes': self.n_modes,
                'surrogates': self.surrogates
            }, f)
        print(f"代理模型已保存: {save_path}")

    @classmethod
    def load(cls, save_path):
        """加载代理模型"""
        with open(save_path, 'rb') as f:
            data = pickle.load(f)

        surrogate = cls(method=data['method'], rbf_function=data['rbf_function'])
        surrogate.train_params = data['train_params']
        surrogate.n_modes = data['n_modes']
        surrogate.surrogates = data['surrogates']

        print(f"代理模型已加载: {save_path}")
        return surrogate