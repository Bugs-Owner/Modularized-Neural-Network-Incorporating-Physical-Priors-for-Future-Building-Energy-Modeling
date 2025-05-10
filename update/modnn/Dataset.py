from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
import os
import torch
import pickle


def _get_ModNN_input(start_time, timestep_minutes, occupancy=None, hvac=None,
                         temp_amb=None, solar=None, temp_room=None, path_to_save=None):
    """
    Generate dataframe that can be used for modnn.

    Args:
        start_time (str): Start timestamp (e.g., '2023-07-01 00:00')
        timestep_minutes (int): Time resolution (e.g., 15 for 15-minute steps)
        occupancy (list or np.array): Occupancy schedule
        hvac (list or np.array): HVAC power values (in W)
        temp_amb (list or np.array): Ambient temperature [°F]
        solar (list or np.array): Solar radiation [W/m²]
        temp_room (list or np.array): Room temp [°F]

    Returns:
        pd.DataFrame: Formatted and time-indexed input DataFrame
    """

    n = len(hvac)
    index = pd.date_range(start=start_time, periods=n, freq=f"{timestep_minutes}min")

    def fill_or_default(arr, default):
        return arr if arr is not None else np.full(n, default)

    df = pd.DataFrame({
        "temp_room": fill_or_default(temp_room, 72),
        "temp_amb": fill_or_default(temp_amb, 85),
        "solar": fill_or_default(solar, 0),
        "occ": fill_or_default(occupancy, 0),
        "phvac": hvac
    }, index=index)
    df=df.resample('15T').mean()
    df.to_csv(path_to_save)

    return df


class MyData(Dataset):
    """
    Generate sequence-to-sequence pairs for learning.
    """
    def __init__(self, X_seq, y_seq):
        self.X = np.array(X_seq)
        self.y = np.array(y_seq)[:, :, [0]]  # Index 0 is Tzone

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), \
               torch.tensor(self.y[index], dtype=torch.float32)


class DataCook:
    """
    Full data pipeline for modnn:
    - Load CSV
    - Add time-based features
    - Normalize inputs
    - Slice into training/testing sequences
    - Create PyTorch DataLoaders
    """

    def __init__(self, args):
        """
        Args:
            args (dict): Configuration arguments from config.py
        """
        self.args = args
        self.user_defined_minmax = args["user_defined_minmax"]
        self.scaler_save_name = args["scaler_save_name"]
        self.scaler_load = args["scaler_load"]
        self.df = None
        self.processed_data = None
        self.scalers = None
        folder_name = "../Scaler/{}".format(self.args['save_name'])
        self.scaler_path = os.path.join(folder_name, self.scaler_save_name)

    def load_data(self):
        """Load raw CSV data and preprocess it."""
        self.df = pd.read_csv(self.args["datapath"], index_col=[0])
        self._parse_time_index()
        self._generate_time_features()
        self._scale_features()

    def load_scaler(self, path):
        """Load scaler dictionary from pickle file."""
        with open(path, "rb") as f:
            self.scalers = pickle.load(f)

    def _parse_time_index(self):
        """Convert index to datetime format (auto-detect)."""
        try:
            self.df.index = pd.to_datetime(self.df.index, format="%m/%d/%Y %H:%M")
        except:
            try:
                self.df.index = pd.to_datetime(self.df.index, format="%Y-%m-%d %H:%M:%S")
            except:
                raise ValueError("Unsupported datetime format in index.")

    def _generate_time_features(self):
        """Add normalized time-of-day features (sin and cos scaled to [0, 1], not [-1, 1] like people normally did)."""
        if "day_sin" not in self.df.columns:
            time_hours = self.df.index.hour + self.df.index.minute / 60
            self.df["day_sin"] = np.sin(2 * np.pi * time_hours / 24) / 2 + 0.5
            self.df["day_cos"] = np.cos(2 * np.pi * time_hours / 24) / 2 + 0.5

    def _scale_features(self):
        """Apply MinMax scaling to each feature and save scalers."""
        features = ["temp_room", "temp_amb", "solar", "occ", "phvac"]

        # Load existing scaler if provided
        if self.scaler_load:
            self.load_scaler(self.scaler_path)
            temp_scaler = self.scalers["temp"]
            flux_scaler = self.scalers["flux"]
        else:
        # Or fit a new sacler
            scalers = {}
            # Temperature scaler
            if self.user_defined_minmax["temp"]:
                temp_min, temp_max = self.user_defined_minmax["temp"]
                temp_scaler = MinMaxScaler(feature_range=(0, 1))
                temp_scaler.fit(np.array([[temp_min], [temp_max]]))
            else:
                temp_scaler = MinMaxScaler(feature_range=(0, 1))
                temp_scaler.fit(self.df[["temp_room", "temp_amb"]].values.flatten().reshape(-1, 1))
            scalers["temp"] = temp_scaler

            # Flux scaler
            if self.user_defined_minmax["flux"]:
                flux_min, flux_max = self.user_defined_minmax["flux"]
                flux_scaler = MinMaxScaler(feature_range=(-1, 1))
                flux_scaler.fit(np.array([[flux_min], [flux_max]]))
            else:
                flux_scaler = MinMaxScaler(feature_range=(-1, 1))
                flux_scaler.fit(self.df[["phvac"]].values.flatten().reshape(-1, 1))
            scalers["flux"] = flux_scaler

            # Other feature scalers
            for f in features:
                if f in ["temp_room", "temp_amb", "phvac"]:
                    continue
                else:
                    scalers[f] = MinMaxScaler(feature_range=(0, 1))
                    scalers[f].fit(self.df[f].values.flatten().reshape(-1, 1))
            self.scalers = scalers

            # Save newly created scalers
            folder_name = "../Scaler/{}".format(self.args['save_name'])
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            scaler_path = os.path.join(folder_name, self.scaler_save_name)
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scalers, f)

        # Transform using scalers
        scaled_temp_room = self.scalers["temp"].transform(self.df[["temp_room"]].values)
        scaled_temp_amb = self.scalers["temp"].transform(self.df[["temp_amb"]].values)
        scaled_solar = self.scalers["solar"].transform(self.df[["solar"]].values)
        scaled_phvac = self.scalers["flux"].transform(self.df[["phvac"]].values)
        scaled_occ = self.scalers["occ"].fit_transform(self.df[["occ"]].values)

        # Combine
        self.processed_data = np.hstack([
            scaled_temp_room,
            scaled_temp_amb,
            scaled_solar,
            self.df[["day_sin", "day_cos"]].values,
            scaled_occ,
            scaled_phvac,
        ])

    def prepare_data_splits(self):
        """Split into training and testing datasets."""
        res = int(1440 / self.args["resolution"])
        start, train, test = self.args["startday"], self.args["trainday"], self.args["testday"]
        en_len, de_len = self.args["enLen"], self.args["deLen"]

        self.trainingdf = self.processed_data[res * start : res * (start + train)]
        # offset an encoder, since the prediction needs to start from 12:00
        self.testingdf = self.processed_data[res * (start + train) - en_len :
                                             res * (start + train + test) + de_len]

        self.test_raw_df = self.df.iloc[res * (start + train): res * (start + train + test + 1)]
        self.test_start = self.test_raw_df.index[0].strftime("%m-%d")
        self.test_end = self.test_raw_df.index[-1].strftime("%m-%d")

    def create_dataloaders(self):
        """Generate PyTorch dataloaders."""
        self.TrainLoader, self.ValidLoader = self._create_dataloader(self.trainingdf,
                                                                     self.args["training_batch"],
                                                                     shuffle=True, split=0.3)
        self.TestLoader = self._create_dataloader(self.testingdf, batch_size=len(self.testingdf), shuffle=False)

    def _create_dataloader(self, data, batch_size, shuffle, split=None):
        """Internal function to create dataset + optional split."""
        X, y = self._generate_sequences(data)
        dataset = MyData(X, y)

        if split:
            train_size = int((1 - split) * len(dataset))
            valid_size = len(dataset) - train_size
            train_ds, valid_ds = random_split(dataset, [train_size, valid_size])
            return DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle), \
                   DataLoader(valid_ds, batch_size=batch_size, shuffle=shuffle)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _generate_sequences(self, data):
        """Slice data into overlapping sequences of encoder+decoder length."""
        en_len, de_len = self.args["enLen"], self.args["deLen"]
        X, y = [], []
        for i in range(len(data) - (en_len + de_len)):
            seq = data[i: i + en_len + de_len]
            # Don't be surprise why X and Y are suing same index
            # The offset was considered in model itself
            X.append(seq)
            y.append(seq)
        return X, y

    def cook(self):
        """Run the full pipeline."""
        self.load_data()
        self.prepare_data_splits()
        self.create_dataloaders()
