import sys
import numpy as np
import matplotlib.pyplot as plt
from data_treatment_class import Data_treatment


# from describe import get_numeric_data, clean_data

def get_numeric_data_by_house(datas, houses):
    min_max = []
    col = 1
    while col < datas.shape[1]:
        if isinstance(datas[0][col], str):
            datas = np.delete(datas, col, 1)
            col -= 1
        else:
            min_max.append((np.min(datas[:, col]), np.max(datas[:, col])))
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
        data = np.delete(data, 0, 1)
        datas[house] = data
    return datas, min_max

def get_disparite_per_house(df):
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    data, min_max = get_numeric_data_by_house(df.data, houses)
    features = df.features
    for house in houses:
        # On clean les donnees 
        data[house] = df.get_clean_data(data[house])
        # On normalise les donnees
        for column in range(data[house].shape[1]):
            min = min_max[column][0]
            max = min_max[column][1]
            for row in range(data[house].shape[0]):
                data[house][row][column] = (data[house][row][column] - min) / (max - min) * 100

    means = []
    for house in houses:
        means.append(np.mean(data[house], axis=0))
    means = np.array(means, dtype=np.float64)
    # print(means)

    i = 0
    disparite_houses = {}
    for feature in features:
        disparite_houses[feature] = np.std(means, axis=0)[i]
        i += 1
    return disparite_houses

if __name__ == "__main__":
    file = sys.argv[1]
    df = Data_treatment(file)
    disparite_houses = get_disparite_per_house(df)


    print("disparite_houses : ")
    print(disparite_houses)

    plt.bar(list(disparite_houses.keys()), disparite_houses.values(), color='g')
    plt.title('Difference in treatment regarding the house \nin the different Hogwards courses')
    plt.xlabel('Courses')
    plt.ylabel('Standard deviation \nregarding the mean of the houses')
    plt.setp(plt.xticks()[1], rotation=90)
    plt.tight_layout()
    # plt.show()
    plt.savefig("histogram.png")
