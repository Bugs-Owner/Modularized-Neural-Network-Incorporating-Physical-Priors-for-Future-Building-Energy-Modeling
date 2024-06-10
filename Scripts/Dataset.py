from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
import os
import torch
import joblib


class MyData(Dataset):
    def __init__(self, X_seq, y_seq):
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
        self.day_sin = None
        self.day_cos = None
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

    def data_preprocess(self, datapath, num_zone, args):
        self.df = pd.read_csv(datapath, index_col=[0])
        try:
            try:
                self.df.index = pd.to_datetime(self.df.index, format="%m/%d/%Y %H:%M")
            except:
                self.df.index = pd.to_datetime(self.df.index, format="%Y-%m-%d %H:%M:%S")
        except:
            print("Error, please change your time format to '%m/%d/%Y %H:%M' or '%Y-%m-%d %H:%M:%S'")
        self.num_zone = num_zone
        # Split data into zones
        processed_data = {}
        for zone in range(num_zone):
            # Data
            Tout = self.df['temp_outdoor'].to_numpy().reshape(-1, 1)

            Solar = self.df['solar'].to_numpy().reshape(-1, 1)
            if 'day_sin' in self.df.columns:
                0
            else:
                self.df["Time"] = self.df.index
                h = self.df["Time"].dt.hour
                m = self.df["Time"].dt.minute
                ti = h + m / 60
                self.df["day_sin"] = np.sin(ti * (2. * np.pi / 24))
                self.df["day_cos"] = np.cos(ti * (2. * np.pi / 24))
            Day_sin = self.df['day_sin'].to_numpy().reshape(-1, 1)
            Day_cos = self.df['day_cos'].to_numpy().reshape(-1, 1)
            Occ = self.df['occ_{}'.format(zone)].to_numpy().reshape(-1, 1)
            Tzone = self.df['temp_zone_{}'.format(zone)].to_numpy().reshape(-1, 1)
            Phvac = self.df['phvac_{}'.format(zone)].to_numpy().reshape(-1, 1)
            try:
                Tset = self.df['setpt_{}'.format(zone)].to_numpy().reshape(-1, 1)
            except:
                print("no setpoint detected, use Zone temperature instead")
                Tset = Tzone

            upper = np.ones(96) * 80
            lower = np.ones(96) * 65
            upper[6 * 4:18 * 4] = 75
            lower[6 * 4:18 * 4] = 70
            weight = np.ones(96) * 1
            weight[6 * 4:18 * 4] = 2
            price = np.ones(96) * 1
            price[15 * 4: 18 * 4] = 5
            daylen = int(Tzone.shape[0] / 96)
            upper = np.tile(upper, daylen).reshape(-1, 1)
            lower = np.tile(lower, daylen).reshape(-1, 1)
            weight = np.tile(weight, daylen).reshape(-1, 1)
            price = np.tile(price, daylen).reshape(-1, 1)
            # Scaler
            ToutScaler = MinMaxScaler(feature_range=(-1, 1))
            SolarScaler = MinMaxScaler(feature_range=(-1, 1))
            OccScaler = MinMaxScaler(feature_range=(-1, 1))
            TzoneScaler = MinMaxScaler(feature_range=(-1, 1))

            if np.all(Phvac<0):
                PhvacScaler = MinMaxScaler(feature_range=(-1*args.scale, 0))
            else:
                PhvacScaler = MinMaxScaler(feature_range=(-1*args.scale, 0))
            TsetScaler = MinMaxScaler(feature_range=(-1, 1))

            # Normalization
            Toutscaled = ToutScaler.fit_transform(Tout).reshape(-1, 1)
            Solarscaled = SolarScaler.fit_transform(Solar).reshape(-1, 1)
            Occscaled = OccScaler.fit_transform(Occ).reshape(-1, 1)
            Tzonescaled = TzoneScaler.fit_transform(Tzone).reshape(-1, 1)
            Phvacscaled = PhvacScaler.fit_transform(Phvac).reshape(-1, 1)
            Tsetscaled = TsetScaler.fit_transform(Tset).reshape(-1, 1)
            upperscaled = TzoneScaler.transform(upper).reshape(-1, 1)
            lowerscaled = TzoneScaler.transform(lower).reshape(-1, 1)

            # Summary
            Summary = np.concatenate((Tzonescaled, Toutscaled, Solarscaled, Day_sin, Day_cos, Occscaled,
                                      Tsetscaled, Phvacscaled, upperscaled, lowerscaled, weight, price), axis=1)
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

            saved_scalers = {'ToutScaler': ToutScaler, 'SolarScaler': SolarScaler, 'OccScaler': OccScaler,
                             'TzoneScaler': TzoneScaler, 'TsetScaler': TsetScaler, 'PhvacScaler': PhvacScaler}
            joblib.dump(saved_scalers, '../Checkpoint/saved_scalers.pkl')

        self.processed_data = processed_data

    def data_roll(self, args):
        self.resolution = args.resolution
        self.startday = args.startday
        self.trainday = args.trainday
        self.testday = args.testday
        self.enLen = args.enco
        self.deLen = args.deco
        res = int(1440 / self.resolution)
        trainingdf, testingdf = {},{}
        for zone in range(self.num_zone):
            trainingdf[zone] = self.processed_data[zone]["Summary"][res * (self.startday):res * (self.startday + self.trainday)]
            testingdf[zone] = self.processed_data[zone]["Summary"][res * (self.startday + self.trainday) - (self.enLen+1):
                                                                   res * (self.startday + self.trainday + self.testday) + self.deLen + 1]
        self.trainingdf = trainingdf
        self.testingdf = testingdf
        self.test_raw_df = self.df[res * (self.startday + self.trainday):res * (self.startday + self.trainday + self.testday + 1)]
        self.test_start = self.test_raw_df.index[0].strftime("%m-%d")
        self.test_end = self.test_raw_df.index[-1].strftime("%m-%d")

    def data_loader(self, training_batch):
        """
        1) Index is important
        2) Check my instruction to better understand
        """
        self.training_batch = training_batch
        # Training loader
        TrainLoader, ValidLoader = {}, {}
        for zone in range(self.num_zone):
            L = len(self.trainingdf[zone])
            X, y = [], []
            for i in range(L - 1 - (self.enLen + 1 + self.deLen)):
                train_seq = self.trainingdf[zone][i: i + (self.enLen + 1 + self.deLen - 1)]
                train_label = self.trainingdf[zone][i + 1: i + (self.enLen + 1 + self.deLen - 1) + 1]
                X.append(train_seq)
                y.append(train_label)
            myset = MyData(X, y)
            train_size = int(0.7 * len(myset))
            valid_size = len(myset) - train_size
            train_dataset, valid_dataset = random_split(myset, [train_size, valid_size])
            train_params = {'batch_size': self.training_batch,
                            'shuffle': True}
            vali_params= {'batch_size': self.training_batch,
                          'shuffle': True}
            TrainLoader[zone] = DataLoader(train_dataset, **train_params)
            ValidLoader[zone] = DataLoader(valid_dataset, **vali_params)
        # Testing loader
        TestLoader = {}
        for zone in range(self.num_zone):
            L = len(self.testingdf[zone])
            X, y = [], []
            for i in range(L - (self.enLen + 1 + self.deLen) - 1):
                test_seq = self.testingdf[zone][i: i + (self.enLen + 1 + self.deLen - 1)]
                test_label = self.testingdf[zone][i + 1: i + (self.enLen + 1 + self.deLen - 1) + 1]
                X.append(test_seq)
                y.append(test_label)
            myset = MyData(X, y)
            params = {'batch_size': L - (self.enLen + 1 + self.deLen) - 1,
                      'shuffle': False}
            TestLoader[zone] = DataLoader(myset, **params)

        self.TrainLoader = TrainLoader
        self.ValidLoader = ValidLoader
        self.TestLoader = TestLoader

    def eplus_data_manager(self, args):
        # Historical data
        folder = '../Dataset/Eplus_Generator/Eplus_realtime/{}_{}_{}_{}'.format(args.Eplusmdl, args.city, args.term, args.sce)
        csv_name = '{}_{}_{}.csv'.format(args.controllertype, args.Begin_Month, args.Begin_Day_of_Month)
        read_csv = os.path.join(folder, csv_name)
        Historical = pd.read_csv(read_csv)[96*2:] #It also includes current measurement
        # Future Data
        folder = '../Dataset/Eplus_Generator/Eplus_train/{}_{}_{}_{}'.format(args.Eplusmdl, args.city, args.term, args.sce)
        csv_name = '{}_{}_{}.csv'.format(args.controllertype, args.Begin_Month, args.Begin_Day_of_Month)
        read_csv = os.path.join(folder, csv_name)
        future = pd.read_csv(read_csv)
        # Scalers
        saved_scalers = joblib.load('../Checkpoint/saved_scalers.pkl')
        #
        import warnings
        warnings.simplefilter(action='ignore')
        # Data replace
        future.cooling[:args.timestep+1] = Historical.Phvac
        future.temp_zone_0[:args.timestep+1] = (Historical.Tzone * 9/5) + 32
        future.phvac_0[:args.timestep+1] = Historical.Phvac * -1

        # from matplotlib import pyplot as plt
        # plt.plot(Historical.Phvac.values[:96])
        # plt.plot(future.cooling[:args.timestep].values)
        # plt.show()
        self.df = future
        processed_data = {}
        for zone in range(1):
            # Data
            Tout = self.df['temp_outdoor'].to_numpy().reshape(-1, 1)
            Solar = self.df['solar'].to_numpy().reshape(-1, 1)
            Day_sin = self.df['day_sin'].to_numpy().reshape(-1, 1)
            Day_cos = self.df['day_cos'].to_numpy().reshape(-1, 1)
            Occ = self.df['occ_{}'.format(zone)].to_numpy().reshape(-1, 1)
            Tzone = self.df['temp_zone_{}'.format(zone)].to_numpy().reshape(-1, 1)
            Phvac = self.df['phvac_{}'.format(zone)].to_numpy().reshape(-1, 1)
            try:
                Tset = self.df['setpt_{}'.format(zone)].to_numpy().reshape(-1, 1)
            except:
                print("no setpoint detected, use Zone temperature instead")
                Tset = Tzone

            upper = (1 - Occ / 3) * 80 + (Occ / 3) * 75
            lower = (1 - Occ / 3) * 65 + (Occ / 3) * 70

            weight = np.ones(96) * 1
            weight[6 * 4:18 * 4] = 2
            price = np.ones(96) * 10
            price[15 * 4: 18 * 4] = 50
            daylen = int(Tzone.shape[0] / 96)
            weight = np.tile(weight, daylen).reshape(-1, 1)
            price = np.tile(price, daylen).reshape(-1, 1)
            # Scaler
            # Normalization
            Toutscaled = saved_scalers['ToutScaler'].transform(Tout).reshape(-1, 1)
            Solarscaled = saved_scalers['SolarScaler'].transform(Solar).reshape(-1, 1)
            Occscaled = saved_scalers['OccScaler'].transform(Occ).reshape(-1, 1)
            Tzonescaled = saved_scalers['TzoneScaler'].transform(Tzone).reshape(-1, 1)
            Phvacscaled = saved_scalers['PhvacScaler'].transform(Occ).reshape(-1, 1)
            Tsetscaled = saved_scalers['TsetScaler'].transform(Tzone).reshape(-1, 1)
            upperscaled = saved_scalers['TzoneScaler'].transform(upper).reshape(-1, 1)
            lowerscaled = saved_scalers['TzoneScaler'].transform(lower).reshape(-1, 1)
            # Summary
            Summary = np.concatenate((Tzonescaled, Toutscaled, Solarscaled, Day_sin, Day_cos, Occscaled,
                                      Tsetscaled, Phvacscaled, upperscaled, lowerscaled, weight, price), axis=1)
            # Collect
            space = {}
            space['Day_sin'], space['Day_cos'] = Day_sin, Day_cos
            space['Tout'], space['Solar'] = Tout, Solar
            space['Occ'], space['Tzone'], space['Tset'], space['Phvac'] = Occ, Tzone, Tset, Phvac

            space['Toutscaled'], space['Solarscaled'], space['Occscaled'] = Toutscaled, Solarscaled, Occscaled
            space['Tzonescaled'], space['Tsetscaled'], space['Phvacscaled'] = Tzonescaled, Tsetscaled, Phvacscaled
            space['Summary'] = Summary
            processed_data[zone] = space
        self.processed_data = processed_data
        inputX = []
        inputX.append(self.processed_data[zone]["Summary"][args.timestep - args.enco - 1 : args.timestep + args.deco - 1, :])
        myset = MyData(inputX, inputX)
        params = {'batch_size': 1,
                  'shuffle': False}
        self.EplusLoader = DataLoader(myset, **params)




