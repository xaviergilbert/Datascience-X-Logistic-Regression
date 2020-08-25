import sys
import numpy as np
import matplotlib.pyplot as plt
from data_treatment_class import Data_treatment


# from describe import get_numeric_data, clean_data




def get_numeric_data_by_house(datas, features, houses):
    col = 1
    while col < datas.shape[1]:
        if isinstance(datas[0][col], str):
            datas = np.delete(datas, col, 1)
            del features[col]
            col -= 1
        col += 1
    datas = {houses[0]: datas, houses[1]: datas, houses[2]: datas, houses[3]: datas}
    for house in houses:
        data = datas[house]
        row = 0
        while row < data.shape[0]:
            if data[row][0] != house:
                data = np.delete(data, row, 0)
                row -= 1
            row += 1
        col = 0
        while col < data.shape[1]:
            if isinstance(data[0][col], str):
                data = np.delete(data, col, 1)
                col -= 1
            col += 1
        datas[house] = data
    return datas, features

if __name__ == "__main__":
    file = sys.argv[1]
    df = Data_treatment(file)

    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']

    data, features = get_numeric_data_by_house(df.data, df.features, houses)
    for house in houses:
        data[house] = df.get_clean_data(data[house])
        data[house] = df.get_normalize_data(data[house])

    features = features[1:]
    means = []
    for house in houses:
        means.append(np.mean(data[house], axis=0))

    means = np.array(means, dtype=np.float64)
    print(means)
    # On calcul l'ecart-type pour chaque matiere

    i = 0
    disparite_houses = {}
    for feature in features:
        disparite_houses[feature] = np.std(means, axis=0)[i]
        i += 1

    print("disparite_houses : ")
    print(disparite_houses)

    plt.bar(list(disparite_houses.keys()), disparite_houses.values(), color='g')
    plt.title('Difference in treatment regarding the house \nin the different Hogwards courses')
    plt.xlabel('Courses')
    plt.ylabel('Standard deviation')
    plt.setp(plt.xticks()[1], rotation=90)
    plt.tight_layout()
    # plt.show()
    plt.savefig("hystogram.png")
