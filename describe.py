import sys
import numpy as np
from data_treatment_class import Data_treatment

def count_func(summary, data, features):
    summary['count'] = {}
    for col in range(data.shape[1]):
        summary['count'][features[col]] = data.shape[0]
    return summary

def mean_func(summary, data, features):
    summary['mean'] = {}
    for col in range(data.shape[1]):
        sum = 0
        for row in range(data.shape[0]):
            sum += data[row][col]
        mean = sum / data.shape[0]
        # summary['count'][features[col]]
        summary['mean'][features[col]] = mean
    return summary

def std_func(summary, data, features):
    summary['std'] = {}
    for col in range(data.shape[1]):
        sum = 0
        for row in range(data.shape[0]):
            sum += abs(data[row][col] - summary['mean'][features[col]]) ** 2
        res = sum / data.shape[0]
        std = np.sqrt(res)
        summary['std'][features[col]] = std
    return summary

def quartile_func(summary, data, features):
    summary['min'] = {}
    summary['25%'] = {}
    summary['50%'] = {}
    summary['75%'] = {}
    summary['max'] = {}
    for col in range(data.shape[1]):
        data_col = data[:, col]
        data_col = np.sort(data_col, 0)
        summary['min'][features[col]] = data_col[0]
        summary['25%'][features[col]] = data_col[int(data_col.shape[0] / 4)]
        summary['50%'][features[col]] = data_col[int(data_col.shape[0] / 2)]
        summary['75%'][features[col]] = data_col[int(data_col.shape[0] * 3 / 4)]
        summary['max'][features[col]] = data_col[-1]
    return summary

def print_func(summary, features):
    case = " " * 14
    print(case, end='')
    for feature in features:
        chaine = feature
        print(" " * (len(case) - len(chaine[:10])) + chaine[:10], end='')
    print("")
    for key, values in summary.items():
        print(key + " " * (len(case) - len(key)), end='')
        for key2, values2 in values.items():
            chaine = values2
            print(" " * (len(case) - len(str(round(chaine, 6)))) + str(round(chaine, 6)), end='')
        print("")

if __name__ == "__main__":
    file = sys.argv[1]
    df = Data_treatment(file)

    summary = {}
    summary = count_func(summary, df.normalize_data, df.features)
    summary = mean_func(summary, df.normalize_data, df.features)
    summary = std_func(summary, df.normalize_data, df.features)
    summary = quartile_func(summary, df.normalize_data, df.features)
    print_func(summary, df.num_features)
    # print(summary)
