from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch


class MyData(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X = np.array(X_seq)
        self.y = np.array(y_seq)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        features = self.X[index]
        targets = self.y[index]
        return torch.from_numpy(features).float(), torch.from_numpy(targets).float()


class DataCook:
    def __init__(self):
        # Data Pre-process
        self.test_end = None
        self.test_start = None
        self.df = None
        self.tar = None
        self.T_out = None
        self.Solar = None
        self.day_sin, self.day_cos = None, None
        self.adj_data = None
        self.processed_data = None
        self.normalized_matrix = None

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

    def data_preprocess(self, datapath, num_zone, adj_matrix, mode):
        self.df = pd.read_csv(datapath.replace("\\","/"), index_col=[0])
        self.df.index = pd.to_datetime(self.df.index, format="%Y-%m-%d %H:%M:%S")
        self.num_zone = num_zone
        adj_data = []
        for zone in range(num_zone):
            Tzone = self.df['temp_zone_{}'.format(zone)].to_numpy().reshape(-1, 1)
            TzoneScaler = MinMaxScaler(feature_range=(0, 1))
            Tzonescaled = TzoneScaler.fit_transform(Tzone).reshape(-1, 1)
            adj_data.append(Tzonescaled)
        self.adj_data = np.array(adj_data).squeeze().T
        self.normalized_matrix = self.normalize_adjacency_matrix(adj_matrix)

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
            CTset = self.df['Cset_zone_{}'.format(zone)].to_numpy().reshape(-1, 1)
            HTset = self.df['Hset_zone_{}'.format(zone)].to_numpy().reshape(-1, 1)
            CPhvac = self.df['Cphvac_{}'.format(zone)].to_numpy().reshape(-1, 1)
            HPhvac = self.df['Hphvac_{}'.format(zone)].to_numpy().reshape(-1, 1)
            Ehvac = self.df['HVAC_{}'.format(zone)].to_numpy().reshape(-1, 1)
            Total = self.df['Etotal_{}'.format(zone)].to_numpy().reshape(-1, 1)
            Ecool = self.df['Ecool_{}'.format(zone)].to_numpy().reshape(-1, 1)
            Eheat = self.df['Eheat_{}'.format(zone)].to_numpy().reshape(-1, 1)

            # Scaler
            ToutScaler= MinMaxScaler(feature_range=(0, 1))
            SolarScaler= MinMaxScaler(feature_range=(0, 1))
            OccScaler= MinMaxScaler(feature_range=(0, 1))
            EhvacScaler = MinMaxScaler(feature_range=(0, 1))
            EcoolScaler = MinMaxScaler(feature_range=(0, 1))
            EheatScaler = MinMaxScaler(feature_range=(0, 1))
            TotalScaler = MinMaxScaler(feature_range=(0, 1))
            # Normalization
            Toutscaled = ToutScaler.fit_transform(Tout).reshape(-1, 1)
            Solarscaled = SolarScaler.fit_transform(Solar).reshape(-1, 1)
            Occscaled = OccScaler.fit_transform(Occ).reshape(-1, 1)
            Ehvacscaled = EhvacScaler.fit_transform(Ehvac).reshape(-1, 1)
            Ecoolscaled = EcoolScaler.fit_transform(Ecool).reshape(-1, 1)
            Eheatscaled = EheatScaler.fit_transform(Eheat).reshape(-1, 1)
            Totalscaled = TotalScaler.fit_transform(Total).reshape(-1, 1)
            if zone == 0:
                TzoneScaler = MinMaxScaler(feature_range=(0, 1))
                CPhvacScaler = MinMaxScaler(feature_range=(-1, 0))
                HPhvacScaler = MinMaxScaler(feature_range=(0, 1))
                TzoneScaler.fit(Tzone)
                CPhvacScaler.fit(CPhvac)
                HPhvacScaler.fit(HPhvac)
                Tzonescaled = TzoneScaler.transform(Tzone).reshape(-1, 1)
                CTsetscaled = TzoneScaler.transform(CTset).reshape(-1, 1)
                HTsetscaled = TzoneScaler.transform(HTset).reshape(-1, 1)
                CPhvacscaled = CPhvacScaler.transform(CPhvac).reshape(-1, 1)
                HPhvacscaled = HPhvacScaler.transform(HPhvac).reshape(-1, 1)
            else:
                Tzonescaled = TzoneScaler.transform(Tzone).reshape(-1, 1)
                CTsetscaled = TzoneScaler.transform(CTset).reshape(-1, 1)
                HTsetscaled = TzoneScaler.transform(HTset).reshape(-1, 1)
                CPhvacscaled = CPhvacScaler.transform(CPhvac).reshape(-1, 1)
                HPhvacscaled = HPhvacScaler.transform(HPhvac).reshape(-1, 1)

            # Summary
            # remove self.adj_data.reshape(-1, 1), not sure
            if num_zone == 1:
                self.adj_data = self.adj_data.reshape(-1, 1)
            else:
                self.adj_data = self.adj_data
            if mode == "Cooling":
                Tsetscaled = CTsetscaled
                Phvacscaled = CPhvacscaled
            else:
                Tsetscaled = HTsetscaled
                Phvacscaled = HPhvacscaled

            Summary = np.concatenate((Tzonescaled, Toutscaled, Solarscaled, Day_sin, Day_cos,
                                      Occscaled, Tsetscaled, Phvacscaled, self.adj_data,
                                      Ehvacscaled, Totalscaled, Ecoolscaled, Eheatscaled), axis=1)
            # Collect
            space = {}
            space['Day_sin'], space['Day_cos'] = Day_sin, Day_cos
            space['Tout'], space['Solar'] = Tout, Solar
            space['Occ'], space['Tzone'] = Occ, Tzone
            space['ToutScaler'], space['SolarScaler'], space['OccScaler'] = ToutScaler, SolarScaler, OccScaler
            space['TzoneScaler'], space['EhvacScaler'] = TzoneScaler, EhvacScaler
            space['TotalScaler'] = TotalScaler
            space['EcoolScaler'], space['EheatScaler'] = EcoolScaler, EheatScaler
            space['Toutscaled'], space['Solarscaled'], space['Occscaled'] = Toutscaled, Solarscaled, Occscaled
            space['Tzonescaled'], space['Tsetscaled'], space['Phvacscaled'] = Tzonescaled, Tsetscaled, Phvacscaled
            space['Ehvacscaled'], space['Totalscaled'] = Ehvacscaled, Totalscaled
            space['Ecoolscaled'], space['Eheatscaled'] = Ecoolscaled, Eheatscaled
            space['Summary'] = Summary
            if mode == "Cooling":
                space['Tset'] = CTset
                space['Phvac'] = CPhvac
                space['PhvacScaler'] = CPhvacScaler
            else:
                space['Tset'] = HTset
                space['Phvac'] = HPhvac
                space['PhvacScaler'] = HPhvacScaler
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
        self.train_raw_df = self.df[res * self.startday : res * (self.startday + self.trainday)]
        self.test_start = self.test_raw_df.index[0].strftime("%m-%d")
        self.test_end = self.test_raw_df.index[-1].strftime("%m-%d")
        self.train_start = self.train_raw_df.index[0].strftime("%m-%d")
        self.train_end = self.train_raw_df.index[-1].strftime("%m-%d")

    def data_loader(self, training_batch):
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
            myset = MyData(X, y)
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
            myset = MyData(X, y)
            params = {'batch_size': L - self.enLen - self.deLen - 1,
                      'shuffle': False}
            TestLoader[zone] = DataLoader(myset, **params)

        self.TrainLoader = TrainLoader
        self.TestLoader = TestLoader

