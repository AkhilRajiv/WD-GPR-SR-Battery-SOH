import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Any

class BatteryDataPipeline:
    """
    Leak-free data pipeline for CALCE battery state-of-health (SOH) prediction.
    Enforces strict Leave-One-Cell-Out (LOCO) cross-validation and nested
    Out-Of-Fold (OOF) splitting for residual generation.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.cells = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
        self.features = [
            'Discharge_Time(s)',
            'CC_Charge_Time(s)',
            'Avg_Voltage(V)',
            'Avg_Resistance(Ohm)'
        ]
        self.feature_names = [
            'Discharge_Time',
            'CC_Charge_Time',
            'Avg_Voltage',
            'Avg_Resistance'
        ]
        self.target = 'SOH(%)'
        self.datasets: Dict[str, pd.DataFrame] = {}
        self._load_data()

    def _load_data(self):
        """Loads and sorts the 4 processed CALCE datasets."""
        for cell in self.cells:
            file_path = os.path.join(self.data_dir, f'{cell}_features.csv')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Processed dataset not found: {file_path}")

            df = pd.read_csv(file_path).sort_values('Global_Cycle').reset_index(drop=True)
            # Standard treatment of non-positive CC charge durations as missing
            df.loc[df['CC_Charge_Time(s)'] <= 0, 'CC_Charge_Time(s)'] = np.nan
            self.datasets[cell] = df

    def get_loco_fold(self, test_cell: str) -> Dict[str, Any]:
        """
        Generates outer Leave-One-Cell-Out train and test data.
        Imputer, scaler, and target statistics are strictly fitted on train cells only.
        """
        if test_cell not in self.cells:
            raise ValueError(f"Unknown test cell: {test_cell}. Expected one of {self.cells}")

        train_cells = [c for c in self.cells if c != test_cell]
        df_train = pd.concat([self.datasets[c] for c in train_cells], ignore_index=True)
        df_test = self.datasets[test_cell].copy()

        # Extract numpy arrays
        X_train_raw = df_train[self.features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        X_test_raw = df_test[self.features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        y_train = df_train[self.target].to_numpy(dtype=float)
        y_test = df_test[self.target].to_numpy(dtype=float)

        # Preprocessing fitted strictly on training data
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # Target scaling parameters for numerical stability in GPR
        y_train_mean = float(y_train.mean())
        y_train_std = float(y_train.std())
        y_train_scaled = (y_train - y_train_mean) / y_train_std

        return {
            'test_cell': test_cell,
            'train_cells': train_cells,
            'X_train_raw': X_train_raw,
            'X_test_raw': X_test_raw,
            'X_train_scaled': X_train_scaled,
            'y_train': y_train,
            'y_train_scaled': y_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'y_train_mean': y_train_mean,
            'y_train_std': y_train_std,
            'scaler': scaler,
            'imputer': imputer,
            'features': self.features,
            'feature_names': self.feature_names
        }

    def get_nested_oof_folds(self, train_cells: List[str]) -> List[Dict[str, Any]]:
        """
        Generates 3 inner LOCO splits across the 3 training cells.
        Used to generate leak-free Out-Of-Fold (OOF) residuals and predictive variances
        for symbolic regression without fitting on the GPR training set.
        """
        inner_folds = []
        for inner_val_cell in train_cells:
            inner_train_cells = [c for c in train_cells if c != inner_val_cell]
            df_inner_train = pd.concat([self.datasets[c] for c in inner_train_cells], ignore_index=True)
            df_inner_val = self.datasets[inner_val_cell].copy()

            X_tr_raw = df_inner_train[self.features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            X_val_raw = df_inner_val[self.features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            y_tr = df_inner_train[self.target].to_numpy(dtype=float)
            y_val = df_inner_val[self.target].to_numpy(dtype=float)

            imputer = SimpleImputer(strategy='median')
            X_tr_imp = imputer.fit_transform(X_tr_raw)
            X_val_imp = imputer.transform(X_val_raw)

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_imp)
            X_val_scaled = scaler.transform(X_val_imp)

            y_tr_mean = float(y_tr.mean())
            y_tr_std = float(y_tr.std())
            y_tr_scaled = (y_tr - y_tr_mean) / y_tr_std

            inner_folds.append({
                'val_cell': inner_val_cell,
                'train_cells': inner_train_cells,
                'X_train_scaled': X_tr_scaled,
                'y_train': y_tr,
                'y_train_scaled': y_tr_scaled,
                'X_val_scaled': X_val_scaled,
                'y_val': y_val,
                'y_train_mean': y_tr_mean,
                'y_train_std': y_tr_std,
                'scaler': scaler,
                'imputer': imputer
            })
        return inner_folds
