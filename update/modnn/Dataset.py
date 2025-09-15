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

    def __init__(self, args, df):
        """
        Args:
            args (dict): Configuration arguments from config.py
        """
        self.args = args
        self.user_defined_minmax = args["user_defined_minmax"]
        self.scaler_save_name = args["scaler_save_name"]
        self.scaler_load = args["scaler_load"]
        self.df = df
        self.processed_data = None
        self.scalers = None
        folder_name = "../Scaler/{}".format(self.args['save_name'])
        self.scaler_path = os.path.join(folder_name, self.scaler_save_name)

    def load_data(self):
        """Load raw CSV data and preprocess it."""
        if self.df is not None:
            pass
        else:
            try:
                self.df = pd.read_csv(self.args["datapath"], index_col=[0])
                # Temperature unit convert
                if self.args["temp_unit"] == "F":
                    self.df['temp_amb'] = (self.df['temp_amb'] - 32) * 5 / 9
                    self.df['temp_room'] = (self.df['temp_room'] - 32) * 5 / 9

            except:
                print("Input error")

        self._parse_time_index()
        self._generate_time_features()
        self._check_missing_features()
        self._scale_features()

    def load_scaler(self, path):
        """Load scaler dictionary from pickle file."""
        with open(path, "rb") as f:
            self.scalers = pickle.load(f)

    def _check_missing_features(self):
        """Check dataset for required features and fill in defaults for missing ones."""
        required_features = ["temp_room", "temp_amb", "solar", "occ", "phvac", "setpt_cool", "setpt_heat", "price"]
        missing_features = [f for f in required_features if f not in self.df.columns]

        if missing_features:
            print(f"Missing features: {missing_features}")

        # Generate setpt_cool and setpt_heat if missing
        occupied = (self.df.index.hour <= 8) | (self.df.index.hour >= 17)
        if "setpt_cool" not in self.df.columns:
            self.df["setpt_cool"] = 25  # default occupied cooling setpoint
            self.df.loc[~occupied, "setpt_cool"] += 4  # apply 4°C setback when unoccupied

        if "setpt_heat" not in self.df.columns:
            self.df["setpt_heat"] = 20  # default occupied heating setpoint
            self.df.loc[~occupied, "setpt_heat"] -= 4  # apply 4°C setback when unoccupied

        # Generate price if missing
        if "price" not in self.df.columns:
            self.df["price"] = 1  # normal price
            peak_hours = (self.df.index.hour >= 17) & (self.df.index.hour < 20)  # 5pm to 8pm
            self.df.loc[peak_hours, "price"] = 5

        print("Missing features filled by default values")


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
        """Add normalized time-of-day features."""
        if "day_sin" not in self.df.columns:
            time_hours = self.df.index.hour + self.df.index.minute / 60
            self.df["day_sin"] = np.sin(2 * np.pi * time_hours / 24) / 2 + 0.5
            self.df["day_cos"] = np.cos(2 * np.pi * time_hours / 24) / 2 + 0.5

    def _scale_features(self):
        """Apply MinMax scaling to each feature and save scalers."""
        features = ["temp_room", "temp_amb", "solar", "occ", "phvac", "setpt_cool", "setpt_heat", "price"]

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
                temp_scaler = MinMaxScaler(feature_range=(-1, 1))
                temp_scaler.fit(np.array([[temp_min], [temp_max]]))
            else:
                temp_scaler = MinMaxScaler(feature_range=(-1, 1))
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
                if f in ["temp_room", "temp_amb", "phvac", "setpt_cool", "setpt_heat"]:
                    continue
                else:
                    scalers[f] = MinMaxScaler(feature_range=(-1, 1))
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
        scaled_setpt_cool = self.scalers["temp"].transform(self.df[["setpt_cool"]].values)
        scaled_setpt_heat = self.scalers["temp"].transform(self.df[["setpt_heat"]].values)
        scaled_price = self.scalers["price"].fit_transform(self.df[["price"]].values)

        # Combine
        self.processed_data = np.hstack([
            scaled_temp_room,
            scaled_temp_amb,
            scaled_solar,
            self.df[["day_sin", "day_cos"]].values,
            scaled_occ,
            scaled_phvac,
            scaled_setpt_cool,
            scaled_setpt_heat,
            scaled_price
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

    def clean_training_data(self):
        """
        Clean training data by removing periods where data remains constant for too long.
        Split data into clean segments and regenerate training/validation datasets.

        Args:
            tolerance_hours (float): Hours of constant data to consider as breaking point
        """
        tolerance_hours = self.args["tolerance_hours"]
        res = int(1440 / self.args["resolution"])  # timesteps per day
        tolerance_steps = int(tolerance_hours * 60 / self.args["resolution"])  # convert hours to timesteps

        # 1) Identify breaking points
        breaking_points = []

        # Check for constant periods in key features (temp_room is index 0)
        temp_room_data = self.trainingdf[:, 0]  # First column is temp_room

        i = 0
        while i < len(temp_room_data) - tolerance_steps:
            # Check if data is constant for tolerance_steps
            window = temp_room_data[i:i + tolerance_steps]
            if np.all(np.abs(window - window[0]) < 1e-6):  # Using small epsilon for floating point comparison
                # Found constant period, mark as breaking point
                breaking_points.append(i)
                # Skip to end of constant period
                j = i + tolerance_steps
                while j < len(temp_room_data) and np.abs(temp_room_data[j] - temp_room_data[i]) < 1e-6:
                    j += 1
                breaking_points.append(j)
                i = j
            else:
                i += 1

        # 2) Split data into clean segments
        clean_segments = []
        start_idx = 0

        for i in range(0, len(breaking_points), 2):
            if i < len(breaking_points):
                # Add segment before breaking point
                if breaking_points[i] > start_idx:
                    clean_segments.append(self.trainingdf[start_idx:breaking_points[i]])

                # Update start index to after the breaking point
                if i + 1 < len(breaking_points):
                    start_idx = breaking_points[i + 1]
                else:
                    start_idx = breaking_points[i] + tolerance_steps

        # Add final segment if exists
        if start_idx < len(self.trainingdf):
            clean_segments.append(self.trainingdf[start_idx:])

        # Filter out segments that are too short for sequence generation
        min_length = self.args["enLen"] + max(self.args.get("multi_deLen", [self.args["deLen"]]))
        clean_segments = [seg for seg in clean_segments if len(seg) >= min_length]

        print(f"Original training data length: {len(self.trainingdf)}")
        print(f"Found {len(breaking_points) // 2} breaking points")
        print(f"Split into {len(clean_segments)} clean segments")
        print(f"Segment lengths: {[len(seg) for seg in clean_segments]}")

        # 3) Generate training and validation datasets from clean segments
        decoder_lengths = self.args.get("multi_deLen", [self.args["deLen"]])
        self.TrainLoader = []
        self.ValidLoader = []

        for dlen in decoder_lengths:
            all_train_datasets = []
            all_valid_datasets = []

            # Process each clean segment
            for segment in clean_segments:
                if len(segment) > self.args["enLen"] + dlen:
                    train_loader, valid_loader = self._create_dataloader(segment,
                                                                         self.args["training_batch"],
                                                                         shuffle=True,
                                                                         split=0.3,
                                                                         en_len=self.args["enLen"],
                                                                         de_len=dlen)
                    if train_loader.dataset:
                        all_train_datasets.append(train_loader.dataset)
                    if valid_loader and valid_loader.dataset:
                        all_valid_datasets.append(valid_loader.dataset)

            # Combine all segments
            if all_train_datasets:
                from torch.utils.data import ConcatDataset
                combined_train_dataset = ConcatDataset(all_train_datasets)
                combined_valid_dataset = ConcatDataset(all_valid_datasets) if all_valid_datasets else None

                self.TrainLoader.append(DataLoader(combined_train_dataset,
                                                   batch_size=self.args["training_batch"],
                                                   shuffle=True))
                if combined_valid_dataset:
                    self.ValidLoader.append(DataLoader(combined_valid_dataset,
                                                       batch_size=self.args["training_batch"],
                                                       shuffle=True))
                else:
                    self.ValidLoader.append(None)

        # Generate control loader from clean segments
        all_control_datasets = []
        for segment in clean_segments:
            if len(segment) >= self.args["enLen"] + self.args["deLen"]:
                control_loader = self._create_dataloader(segment,
                                                         self.args["training_batch"],
                                                         shuffle=True,
                                                         split=None,
                                                         en_len=self.args["enLen"],
                                                         de_len=self.args["deLen"])[0]
                if control_loader.dataset:
                    all_control_datasets.append(control_loader.dataset)

        if all_control_datasets:
            from torch.utils.data import ConcatDataset
            combined_control_dataset = ConcatDataset(all_control_datasets)
            self.ControlLoader = DataLoader(combined_control_dataset,
                                            batch_size=self.args["training_batch"],
                                            shuffle=True)

        # Generate TestLoader (keep same as original method - no cleaning applied to test data)
        self.TestLoader = self._create_dataloader(self.testingdf,
                                                  batch_size=len(self.testingdf),
                                                  shuffle=False,
                                                  en_len=self.args["enLen"],
                                                  de_len=self.args["deLen"])[0]

    def create_dataloaders(self):
        """Generate PyTorch dataloaders with multiple decoder lengths."""
        decoder_lengths = self.args.get("multi_deLen", [self.args["deLen"]])  # e.g., [8, 16, 24, 48, 96]
        self.TrainLoader = []
        self.ValidLoader = []

        for dlen in decoder_lengths:
            TrainLoader_, ValidLoader_ = self._create_dataloader(self.trainingdf,
                                                               self.args["training_batch"],
                                                               shuffle=True,
                                                               split=0.3,
                                                               en_len=self.args["enLen"],
                                                               de_len=dlen)
            self.TrainLoader.append(TrainLoader_)
            self.ValidLoader.append(ValidLoader_)

        # For testing, keep it fixed with the original decoder length
        self.TestLoader = self._create_dataloader(self.testingdf,
                                                  batch_size=len(self.testingdf),
                                                  shuffle=False,
                                                  en_len=self.args["enLen"],
                                                  de_len=self.args["deLen"])[0]

        self.ControlLoader = self._create_dataloader(self.trainingdf,
                                                  self.args["training_batch"],
                                                  shuffle=True,
                                                  split=None,
                                                  en_len=self.args["enLen"],
                                                  de_len=self.args["deLen"])[0]

    def _create_dataloader(self, data, batch_size, shuffle, split=None, en_len=48, de_len=96):
        X, y = self._generate_sequences(data, en_len, de_len)
        dataset = MyData(X, y)
        if split:
            train_size = int((1 - split) * len(dataset))
            valid_size = len(dataset) - train_size
            train_ds, valid_ds = random_split(dataset, [train_size, valid_size])
            return DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle), \
                DataLoader(valid_ds, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), None

    def _generate_sequences(self, data, en_len, de_len):
        """Slice data into overlapping sequences of encoder+decoder length."""
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
        if self.args["use_data_cleaning"]:
            self.clean_training_data()
        else:
            self.create_dataloaders()
