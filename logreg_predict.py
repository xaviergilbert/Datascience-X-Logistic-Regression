#!/usr/bin/env python 

import os
import sys
import csv
import argparse
import numpy as np
from data_treatment_class import Data_treatment

def sigmoid_(x):
    x = np.array(x, dtype=np.float32)
    return 1 / (1 + np.exp(-x))

def predict(X, thetas):
    tmp = np.dot(X, thetas[1:]) + thetas[:1]
    Y_pred = np.array(sigmoid_(tmp))
    return Y_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="csv datas file")
    parser.add_argument("-a", "--accuracy", \
        help="Allow to split the dataset_train to get accuracy", \
        action="store_true")
    args = parser.parse_args()
    file_name = args.file

    try:
        df = Data_treatment(file_name)
    except Exception as e:
        print("Error", e)
        exit()

    X = df.normalize_data[:, 1:]
    Y = df.clean_data[:, :1]

    #pour pouvoir calculer un accuracy correcte
    if args.accuracy:
        perc = ""
        while not perc.isdigit():
            perc = input("How much of the dataset did you use for the training ? (%) : ")
        perc = int(perc) if int(perc) < 100 else 99
        X = X[int(len(Y) * perc / 100):, :]
        Y = Y[int(len(Y) * perc / 100):, :]

    if os.path.exists("thetas.txt"):
        f = open("thetas.txt", "r")
    else:
        print("No thetas.txt found. Please train the model before testing")
        exit()

    thetas = ""
    lines = f.readlines()
    for line in lines:
        thetas += line
    thetas = thetas.replace("\n", "").split("\\")
    house = {0: 'Gryffindor', 1: 'Hufflepuff', 2: 'Ravenclaw', 3: 'Slytherin'}
    thetas = {'Gryffindor': thetas[0], 'Hufflepuff': thetas[1], 'Ravenclaw': thetas[2], 'Slytherin': thetas[3]}
    f.close()

    # Preparation des resultats
    results = [("Index", "Hogward House")]
    for i, ligne in enumerate(X):
        tmp_res = []
        for key, value in thetas.items():
            tmp_res.append(predict(ligne, np.fromstring(value, dtype=float, sep=' ')))
        results.append((i, house[np.argmax(tmp_res)]))

    # MIse des resultats dans fichiers csv
    f = open("houses.csv", "w")
    with f:
        writer = csv.writer(f)
        writer.writerows(results)
    f.close()

    # check Accuracy
    result = []
    for i in range(len(results)):
        if i == 0:
            continue
        result.append(results[i][1])
    true = 0
    false = 0
    for i in range(len(result)):
        if result[i] == Y[i]:
            true += 1
        else:
            false += 1
    print("Accuracy : ", true / (true + false) * 100)


if __name__ == "__main__":
    main()
