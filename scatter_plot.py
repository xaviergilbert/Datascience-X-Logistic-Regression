import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_treatment_class import Data_treatment
from histogram import get_disparite_per_house, get_numeric_data_by_house


#Quelles sont les deux features qui sont semblables ?
# etape 1 : normaliser les donnees
# etape 2 : comparaison des matieres sur :
#   - std
#   - std moy inter maison
#   - min
#   - mean
#   - max

if __name__ == "__main__":
    file = sys.argv[1]
    df = Data_treatment(file)

    print(df.normalize_data)

    exit()