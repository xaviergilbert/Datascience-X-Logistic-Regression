import os
import sys
import csv
import numpy as np
from data_treatment_class import Data_treatment

def sigmoid_(x):
    x = np.array(x, dtype=np.float32)
    return 1 / (1 + np.exp(-x))

def predict(X, thetas):
    """
        Predict class labels for samples in x_train.
        Arg:
            x_train: a 1d or 2d numpy ndarray for the samples
        Returns:
            y_pred, the predicted class label per sample.
            None on any error.
        Raises:
            This method should not raise any Exception.
    """
    tmp = np.dot(X, thetas[1:]) + thetas[:1]
    Y_pred = np.array(sigmoid_(tmp))
    return Y_pred

def main():
    file = sys.argv[1]
    df = Data_treatment(file)

    X = df.normalize_data[:, 1:]
    Y = df.clean_data[:, :1]

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
