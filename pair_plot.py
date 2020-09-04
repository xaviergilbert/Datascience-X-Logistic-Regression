import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_treatment_class import Data_treatment




if __name__ == "__main__":
    file = sys.argv[1]
    df = Data_treatment(file)
    df.data_brut = df.data_brut.drop(['Index', 'First Name', 'Last Name', 'Best Hand', ], axis=1)
    sns.pairplot(df.data_brut, hue="Hogwarts House")
    plt.tight_layout()

    plt.savefig("pair_plot.png")

    # print(df.data_brut)