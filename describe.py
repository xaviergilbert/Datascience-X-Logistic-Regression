import sys
import pandas as pd
import numpy as np


def get_numeric_data(data, features):
    col = 0
    while col < data.shape[1]:
        if isinstance(data[0][col], str):
            data = np.delete(data, col, 1)
            del features[col]
            col -= 1
        col += 1
    return data, features

def clean_data(data):
    for col in range(data.shape[1]):
        row = 0
        while row < data.shape[0]:
            if np.isnan(data[row][col]):
                data = np.delete(data, row, axis=0)
                row -= 1
            row += 1
    return data

def count_func(summary, data):
    summary['count'] = {}
    for col in range(data.shape[1]):
        summary['count']['Feature ' + str(col)] = data.shape[0]
    return summary

def mean_func(summary, data):
    summary['mean'] = {}
    for col in range(data.shape[1]):
        sum = 0
        for row in range(data.shape[0]):
            sum += data[row][col]
        mean = sum / summary['count']['Feature ' + str(col)]
        summary['mean']['Feature ' + str(col)] = mean
    return summary

def std_func(summary, data):
    summary['std'] = {}
    for col in range(data.shape[1]):
        sum = 0
        for row in range(data.shape[0]):
            sum += abs(data[row][col] - summary['mean']['Feature ' + str(col)]) ** 2
        res = sum / data.shape[0]
        std = np.sqrt(res)
        summary['std']['Feature ' + str(col)] = std
    return summary

def quartile_func(summary, data):
    summary['min'] = {}
    summary['25%'] = {}
    summary['50%'] = {}
    summary['75%'] = {}
    summary['max'] = {}
    for col in range(data.shape[1]):
        data_col = data[:, col]
        data_col = np.sort(data_col, 0)
        summary['min']['Feature ' + str(col)] = data_col[0]
        summary['25%']['Feature ' + str(col)] = data_col[int(data_col.shape[0] / 4)]
        summary['50%']['Feature ' + str(col)] = data_col[int(data_col.shape[0] / 2)]
        summary['75%']['Feature ' + str(col)] = data_col[int(data_col.shape[0] * 3 / 4)]
        summary['max']['Feature ' + str(col)] = data_col[-1]
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
    data_brut = pd.read_csv(file)
    data = data_brut.to_numpy()[:,1:]

    columns = data_brut.columns[1:]
    features = []
    for column in columns:
        features.append(column)

    data, features = get_numeric_data(data, features)
    data = clean_data(data)

    summary = {}
    summary = count_func(summary, data)
    summary = mean_func(summary, data)
    summary = std_func(summary, data)
    summary = quartile_func(summary, data)
    print_func(summary, features)
    # print(summary)

# count 
# mean
# std sqrt(mean(abs(x - x.mean())**2))
# min 
# 25%
# 50%
# 75%
# max