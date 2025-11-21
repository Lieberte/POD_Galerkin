import numpy as np
import pandas as pd
import matplotlib
import h5py
import pickle
import json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from src.pod_decomposition import PODDecomposer
from src.pod_surrogate import PODSurrogate
from src.pod_validation import PODValidator

METHOD_OPTIONS = {
    'surrogate_method': 'rbf',  # Options: 'rbf', 'gpr', 'poly'
    'rbf_function': 'cubic',
    # Options: 'multiquadric', 'inverse_multiquadric', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'
}

PATH_CONFIG = {
    'csv_dir': Path("./database/csv/p_nonorm"),
    'valid_h5_folder': Path("./database/h5/valid"),
    'results_dir': Path("./results/p_nonorm_5"),
}

POD_CONFIG = {
    'n_modes': 5,
    'variables': ['SV_P'],
    'use_normalization': False,
}


def load_validation_h5(h5_folder, variables, normalize=False, scalers=None):
    """Load validation data from H5 files with multi-variable stacking."""
    h5_folder = Path(h5_folder)
    h5_files = sorted(h5_folder.glob("*.h5"), key=lambda x: int(x.stem))

    buffer = {var: [] for var in variables}
    flow_rates = []

    for h5_file in h5_files:
        flow_rate = int(h5_file.stem)
        with h5py.File(h5_file, 'r') as f:
            for var in variables:
                data = f['results']['1']['phase-1']['cells'][var]['1'][:]
                buffer[var].append(data.flatten())
        flow_rates.append(flow_rate)

    var_matrices = {}
    for var in variables:
        var_matrices[var] = np.column_stack(buffer[var])

        if normalize and scalers is not None and var in scalers:
            shape = var_matrices[var].shape
            flat_data = var_matrices[var].flatten().reshape(-1, 1)
            normalized_flat = scalers[var].transform(flat_data)
            var_matrices[var] = normalized_flat.reshape(shape)

    data_matrix = np.vstack([var_matrices[var] for var in variables])
    flow_rates = np.array(flow_rates)

    return data_matrix, flow_rates


if __name__ == "__main__":
    variables = POD_CONFIG['variables']
    use_norm = POD_CONFIG['use_normalization']
    n_modes = POD_CONFIG['n_modes']

    data_dir = PATH_CONFIG['csv_dir']
    valid_h5_folder = PATH_CONFIG['valid_h5_folder']

    if use_norm:
        results_dir = PATH_CONFIG['results_dir'] / 'normalized'
    else:
        results_dir = PATH_CONFIG['results_dir'] / 'physical'

    pod_save_dir = results_dir / 'POD'
    validation_dir = results_dir / 'validation'
    surrogate_dir = results_dir / 'surrogate'

    pod_save_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    surrogate_dir.mkdir(parents=True, exist_ok=True)

    if use_norm:
        scaler_path = data_dir / 'scalers.pkl'
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        with open(data_dir / 'variable_indices.json', 'r') as f:
            variable_indices = json.load(f)
    else:
        scalers = None
        variable_indices = None

    df_snapshot = pd.read_csv(data_dir / 'snapshot_matrix.csv', index_col=0)
    df_flowrates = pd.read_csv(data_dir / 'flow_rates.csv')

    train_matrix = df_snapshot.values
    train_flow_rates = df_flowrates['flow_rate'].values

    pod = PODDecomposer()
    pod.fit(train_matrix, n_modes=n_modes)
    pod.save(pod_save_dir)

    metadata = {
        'normalized': use_norm,
        'scaler_path': str(scaler_path) if use_norm else None,
        'variables': variables
    }
    with open(pod_save_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    modes = np.arange(1, len(pod.energy_ratio) + 1)
    ax1.bar(modes, pod.energy_ratio * 100, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Mode Number', fontsize=12)
    ax1.set_ylabel('Energy Ratio (%)', fontsize=12)
    ax1.set_title('Individual Mode Energy Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(modes, pod.cumulative_energy * 100, marker='o', linewidth=2, markersize=6, color='darkred')
    ax2.axhline(y=99, color='green', linestyle='--', linewidth=1.5, label='99% Energy')
    ax2.set_xlabel('Mode Number', fontsize=12)
    ax2.set_ylabel('Cumulative Energy (%)', fontsize=12)
    ax2.set_title('Cumulative Energy Distribution', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(pod_save_dir / 'energy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    train_coeffs = pod.transform(train_matrix).T

    surrogate = PODSurrogate(
        method=METHOD_OPTIONS['surrogate_method'],
        rbf_function=METHOD_OPTIONS['rbf_function']
    )
    surrogate.fit(train_flow_rates, train_coeffs)
    surrogate.save(surrogate_dir / 'surrogate_model.pkl')

    valid_matrix_norm, valid_flow_rates = load_validation_h5(
        valid_h5_folder,
        variables=variables,
        normalize=use_norm,
        scalers=scalers
    )

    errors_norm = PODValidator.validate(
        pod_decomposer=pod,
        surrogate=surrogate,
        valid_matrix=valid_matrix_norm,
        valid_params=valid_flow_rates,
        output_dir=validation_dir / 'normalized_space',
        space_type='normalized',
        variable_indices=variable_indices if use_norm else None
    )

    if use_norm:
        valid_matrix_physical, _ = load_validation_h5(
            valid_h5_folder,
            variables=variables,
            normalize=False,
            scalers=None
        )

        errors_physical = PODValidator.validate(
            pod_decomposer=pod,
            surrogate=surrogate,
            valid_matrix=valid_matrix_physical,
            valid_params=valid_flow_rates,
            output_dir=validation_dir / 'physical_space',
            space_type='physical',
            scalers=scalers,
            variable_indices=variable_indices
        )

    test_flow_rates = np.linspace(train_flow_rates.min(), train_flow_rates.max(), 10)
    pred_coeffs = surrogate.predict(test_flow_rates)
    pred_fields = pod.inverse_transform(pred_coeffs.T)

    pd.DataFrame(
        pred_fields,
        columns=[f'flowrate_{fr:.1f}' for fr in test_flow_rates]
    ).to_csv(surrogate_dir / 'predictions_normalized.csv', float_format='%.8e')

    if use_norm:
        pred_fields_physical = np.zeros_like(pred_fields)
        for var_name, idx_info in variable_indices.items():
            start = idx_info['start']
            end = idx_info['end']

            flat_pred = pred_fields[start:end, :].flatten().reshape(-1, 1)
            pred_fields_physical[start:end, :] = scalers[var_name].inverse_transform(
                flat_pred
            ).reshape((end - start, -1))

        pd.DataFrame(
            pred_fields_physical,
            columns=[f'flowrate_{fr:.1f}' for fr in test_flow_rates]
        ).to_csv(surrogate_dir / 'predictions_physical.csv', float_format='%.8e')

    print(f"Completed! Results saved in: {results_dir}")