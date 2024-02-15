from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch


class MyData(Dataset):
    def __init__(self, X_seq, y_seq, tar):
        if tar=='load':
            self.X = np.array(X_seq)
            self.y = np.array(y_seq)[:, :, [10,11,12,13,14]]
        if tar=='single_temp':
            self.X = np.array(X_seq)
            self.y = np.array(y_seq)[:, :, [0]]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        features = self.X[index]
        targets = self.y[index]
        return torch.from_numpy(features).float(), torch.from_numpy(targets).float()


class DataCook:
    def __init__(self):
        # Data Pre-process
        self.df = None
        self.tar = None
        self.T_out = None
        self.Solar = None
        self.day_sin, self.day_cos = None, None
        self.adj_data = None
        self.processed_data = None

        # Data_roll
        self.num_zone = None
        self.resolution = None
        self.startday = None
        self.trainday = None
        self.testday = None
        self.enLen = None
        self.deLen = None
        self.trainingdf = None
        self.testingdf = None
        self.test_raw_df = None

        # Data_loader
        self.training_batch = None
        self.TrainLoader = None
        self.TestLoader = None
        self.CheckLoader = None

    def normalize_adjacency_matrix(self, adj_matrix):
        # Add self-loops
        adj_matrix_with_loops = adj_matrix + np.eye(adj_matrix.shape[0])
        # Calculate the Degree Matrix
        degree_matrix = np.diag(np.sum(adj_matrix_with_loops, axis=1))
        # Normalizing the adjacency matrix
        degree_matrix_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
        normalized_adj_matrix = degree_matrix_inv_sqrt @ adj_matrix_with_loops @ degree_matrix_inv_sqrt

        return normalized_adj_matrix

    def data_preprocess(self, datapath, num_zone):
        self.df = pd.read_csv(datapath, index_col=[0])
        self.df.index = pd.to_datetime(self.df.index, format="%Y-%m-%d %H:%M:%S")
        self.num_zone = num_zone

        # Split data into zones
        processed_data = {}
        for zone in range(num_zone):
            # Data
            Tout = self.df['temp_outdoor'].to_numpy().reshape(-1, 1)
            Solar = self.df['solar'].to_numpy().reshape(-1, 1)
            Day_sin = self.df['day_sin'].to_numpy().reshape(-1, 1)
            Day_cos = self.df['day_cos'].to_numpy().reshape(-1, 1)
            Occ = self.df['occ_{}'.format(zone)].to_numpy().reshape(-1, 1)
            Tzone = self.df['temp_zone_{}'.format(zone)].to_numpy().reshape(-1, 1)
            Tset = self.df['Cset_zone_{}'.format(zone)].to_numpy().reshape(-1, 1)
            Phvac = self.df['Cphvac_{}'.format(zone)].to_numpy().reshape(-1, 1)
            # Scaler
            ToutScaler= MinMaxScaler(feature_range=(0, 1))
            SolarScaler= MinMaxScaler(feature_range=(0, 1))
            OccScaler= MinMaxScaler(feature_range=(0, 1))
            TzoneScaler = MinMaxScaler(feature_range=(0, 1))
            TsetScaler = MinMaxScaler(feature_range=(0, 1))
            PhvacScaler = MinMaxScaler(feature_range=(-1, 0))
            # Normalization
            Toutscaled = ToutScaler.fit_transform(Tout).reshape(-1, 1)
            Solarscaled = SolarScaler.fit_transform(Solar).reshape(-1, 1)
            Occscaled = OccScaler.fit_transform(Occ).reshape(-1, 1)
            Tzonescaled = TzoneScaler.fit_transform(Tzone).reshape(-1, 1)
            Tsetscaled = TsetScaler.fit_transform(Tset).reshape(-1, 1)
            Phvacscaled = PhvacScaler.fit_transform(Phvac).reshape(-1, 1)
            # Summary
            Summary = np.concatenate((Tzonescaled, Toutscaled, Solarscaled, Day_sin, Day_cos, Occscaled, Tsetscaled, Phvacscaled), axis=1)
            # Collect
            space = {}
            space['Day_sin'], space['Day_cos'] = Day_sin, Day_cos
            space['Tout'], space['Solar'] = Tout, Solar
            space['Occ'], space['Tzone'], space['Tset'], space['Phvac'] = Occ, Tzone, Tset, Phvac
            space['ToutScaler'], space['SolarScaler'], space['OccScaler'] = ToutScaler, SolarScaler, OccScaler
            space['TzoneScaler'], space['TsetScaler'], space['PhvacScaler'] = TzoneScaler, TsetScaler, PhvacScaler
            space['Toutscaled'], space['Solarscaled'], space['Occscaled'] = Toutscaled, Solarscaled, Occscaled
            space['Tzonescaled'], space['Tsetscaled'], space['Phvacscaled'] = Tzonescaled, Tsetscaled, Phvacscaled
            space['Summary'] = Summary
            processed_data[zone] = space

        self.processed_data = processed_data

    def data_roll(self, enLen, deLen, startday, trainday, testday, resolution):
        self.resolution = resolution
        self.startday = startday
        self.trainday = trainday
        self.testday = testday
        self.enLen = enLen
        self.deLen = deLen
        res = int(1440 / self.resolution)
        trainingdf, testingdf = {},{}
        for zone in range(self.num_zone):
            trainingdf[zone] = self.processed_data[zone]["Summary"][res * (self.startday):res * (self.startday + self.trainday)]
            testingdf[zone] = self.processed_data[zone]["Summary"][res * (self.startday + self.trainday) - self.enLen - 1:
                                                                   res * (self.startday + self.trainday + self.testday) + self.deLen]
        self.trainingdf = trainingdf
        self.testingdf = testingdf
        self.test_raw_df = self.df[res * (self.startday + self.trainday):res * (self.startday + self.trainday + self.testday + 1)]
        self.test_start = self.test_raw_df.index[0].strftime("%m-%d")
        self.test_end = self.test_raw_df.index[-1].strftime("%m-%d")

    def data_loader(self, training_batch, tar):
        self.tar = tar
        self.training_batch = training_batch
        # Training loader
        TrainLoader = {}
        for zone in range(self.num_zone):
            L = len(self.trainingdf[zone])
            X, y = [], []
            for i in range(L - self.enLen - self.deLen):
                train_seq = self.trainingdf[zone][i: i + self.enLen + self.deLen]
                train_label = self.trainingdf[zone][i + 1: i + self.enLen + self.deLen + 1]
                X.append(train_seq)
                y.append(train_label)
            myset = MyData(X, y, tar)
            params = {'batch_size': self.training_batch,
                      'shuffle': True}
            TrainLoader[zone] = DataLoader(myset, **params)

        # Testing loader
        TestLoader = {}
        for zone in range(self.num_zone):
            L = len(self.testingdf[zone])
            X, y = [], []
            for i in range(L - self.enLen - self.deLen - 1):
                test_seq = self.testingdf[zone][i: i + self.enLen + self.deLen]
                test_label = self.testingdf[zone][i + 1: i + self.enLen + self.deLen + 1]
                X.append(test_seq)
                y.append(test_label)
            myset = MyData(X, y, tar)
            params = {'batch_size': L - self.enLen - self.deLen - 1,
                      'shuffle': False}
            TestLoader[zone] = DataLoader(myset, **params)

        self.TrainLoader = TrainLoader
        self.TestLoader = TestLoader

