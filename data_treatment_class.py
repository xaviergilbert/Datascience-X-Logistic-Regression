# from describe import features
import pandas as pd
import numpy as np

class Data_treatment():
    def __init__(self, file_name):
        self.data_brut = pd.read_csv(file_name)
        self.data = self.data_brut.to_numpy()[:,1:]
        self.features = self.get_features()
        self.num_data, self.num_features = self.get_numeric_data()
        self.clean_data = self.get_clean_data(self.num_data)
        self.normalize_data = self.get_normalize_data(self.clean_data)

    def get_features(self):
        columns = self.data_brut.columns[1:]
        features = []
        for column in columns:
            features.append(column)
        return features

    def get_numeric_data(self):
        data = self.data
        features = self.features
        col = 0
        while col < data.shape[1]:
            if isinstance(data[0][col], str):
                data = np.delete(data, col, 1)
                del features[col]
                col -= 1
            col += 1
        return data, features

    def get_clean_data(self, data):
        for col in range(data.shape[1]):
            row = 0
            while row < data.shape[0]:
                if not isinstance(data[row][col], float):
                    print("La donnee n'est pas numerique: " + str(data[row][col]))
                if np.isnan(data[row][col]):
                    data = np.delete(data, row, axis=0)
                    row -= 1
                row += 1
        return data

    def get_normalize_data(self, data):
        
        def normalize(data, max, min):
            return (data - min) / (max - min) * 100

        for column in range(data.shape[1]):
            min = np.min(data[:, column])
            max = np.max(data[:, column])
            for row in range(data.shape[0]):
                data[row][column] = normalize(data[row][column], max, min)
        return data