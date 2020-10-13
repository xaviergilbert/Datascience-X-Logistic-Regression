
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_treatment_class import Data_treatment




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="csv datas file")
    parser.add_argument("-d", "--drop", 
        help="Allow the user to drop some features", 
        action="store_true")
    args = parser.parse_args()
    file_name = args.file

    try:
        df = Data_treatment(file_name)
    except Exception as e:
        print("Error", e)
        exit()
        
    df.data_brut = df.data_brut.drop(['Index', 'First Name', 'Last Name', 'Best Hand', 'Birthday'], axis=1)
    if args.drop:
        while True:
            print("Features you can drop : ")
            features = df.data_brut.columns.values[1:]
            for i, feature in enumerate(features):
                print(i, ":", feature)
            index = input("\nEnter a wrong value when finished\nWhich one do you want to drop ? 0-" + str(len(features)) + ": ")
            try:
                df.data_brut = df.data_brut.drop([features[int(index)]], axis=1)
            except Exception as e:
                print(e, "\n")
                break

    print("Building the plot...")
    sns.pairplot(df.data_brut, hue="Hogwarts House")
    plt.tight_layout()

    plt.savefig("pair_plot.png")