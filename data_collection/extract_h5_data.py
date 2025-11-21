import h5py
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler

"""
trans h5 data into csv 
- make an norm(z-score normalization) optional which can be used when variable types extremely different (P and uvw)
- suggestion dataset <= 50g as this program stores all h5 to ache first 
"""

class H5DataExtractor:
    """Extract and normalize multiple variables from h5 files"""

    def __init__(self, h5_folder: str, variables: List[str]):
        self.h5_folder = Path(h5_folder)
        self.variables = variables
        self.h5_files = sorted(self.h5_folder.glob('*.h5'),
                               key=lambda x: int(x.stem))
        self.flow_rates = [int(f.stem) for f in self.h5_files]
        self.scalers = {var: StandardScaler() for var in variables}

    def extract(self, normalize: bool = False, scaler_path: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Extract all h5 files, optionally normalize with global statistics

        Parameters:
        -----------
        normalize : bool
            Whether to normalize data using global mean/std across all cells and cases
        scaler_path : str, optional
            Path to load pre-fitted scalers for validation/test sets

        Returns:
        --------
        stacked_matrix : np.ndarray
            Shape (n_vars * n_cells, n_cases)
            Each variable normalized with single global mean/std
        metadata : dict
            Contains variables, flow_rates, n_cells, normalized flag, and scalers
        """
        if scaler_path is not None:
            self._load_scalers(scaler_path)

        buffer = {var: [] for var in self.variables}

        for h5_file in self.h5_files:
            with h5py.File(h5_file, 'r') as f:
                for var in self.variables:
                    data = f['results']['1']['phase-1']['cells'][var]['1'][:]
                    buffer[var].append(data.flatten())

        var_matrices = {}
        for var in self.variables:
            var_matrices[var] = np.column_stack(buffer[var])

            if normalize:
                if scaler_path is None:
                    # Flatten to 1D, fit, then reshape back
                    shape = var_matrices[var].shape
                    flat_data = var_matrices[var].flatten().reshape(-1, 1)
                    normalized_flat = self.scalers[var].fit_transform(flat_data)
                    var_matrices[var] = normalized_flat.reshape(shape)
                else:
                    shape = var_matrices[var].shape
                    flat_data = var_matrices[var].flatten().reshape(-1, 1)
                    normalized_flat = self.scalers[var].transform(flat_data)
                    var_matrices[var] = normalized_flat.reshape(shape)

        stacked_matrix = np.vstack([var_matrices[var] for var in self.variables])

        metadata = {
            'variables': self.variables,
            'flow_rates': self.flow_rates,
            'n_cells': var_matrices[self.variables[0]].shape[0],
            'normalized': normalize,
            'scalers': self.scalers if normalize else None
        }

        return stacked_matrix, metadata

    def save_results(self, stacked_matrix: np.ndarray, metadata: Dict, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        n_cells = metadata['n_cells']

        row_labels = []
        for var in self.variables:
            row_labels.extend([f'{var}_cell{i}' for i in range(n_cells)])

        col_labels = [f'flowrate_{fr}' for fr in metadata['flow_rates']]

        df = pd.DataFrame(stacked_matrix, index=row_labels, columns=col_labels)
        output_file = save_dir / 'snapshot_matrix.csv'
        df.to_csv(output_file, float_format='%.8e')

        # ====== 新增：保存变量索引信息 ======
        variable_indices = {}
        for i, var in enumerate(self.variables):
            variable_indices[var] = {
                'start': i * n_cells,
                'end': (i + 1) * n_cells,
                'n_cells': n_cells
            }

        import json
        with open(save_dir / 'variable_indices.json', 'w') as f:
            json.dump(variable_indices, f, indent=2)
        # ===================================

        if metadata['normalized']:
            scaler_file = save_dir / 'scalers.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scalers, f)

            norm_params = {}
            for var in self.variables:
                norm_params[f'{var}_mean'] = self.scalers[var].mean_
                norm_params[f'{var}_std'] = self.scalers[var].scale_

            df_norm = pd.DataFrame(norm_params)
            df_norm.to_csv(save_dir / 'normalization_params.csv', index=False)

        df_meta = pd.DataFrame({
            'case_id': range(len(metadata['flow_rates'])),
            'flow_rate': metadata['flow_rates']
        })
        df_meta.to_csv(save_dir / 'flow_rates.csv', index=False)

    def _load_scalers(self, scaler_path: str):
        """Load pre-fitted scalers from pickle file"""
        with open(scaler_path, 'rb') as f:
            self.scalers = pickle.load(f)


if __name__ == "__main__":
    variables = ['SV_U','SV_V','SV_W','SV_P']

    train_extractor = H5DataExtractor("../database/h5/train", variables)
    train_matrix, train_metadata = train_extractor.extract(normalize=True)
    train_extractor.save_results(train_matrix, train_metadata, "../database/csv/uvwp_norm")
