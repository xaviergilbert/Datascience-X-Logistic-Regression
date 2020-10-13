import os
import sys
import argparse
import matplotlib.pyplot as plt
from data_treatment_class import Data_treatment
from describe import mean_func, std_func, quartile_func
from histogram import get_disparite_per_house


#Quelles sont les deux features qui sont semblables ?

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="csv datas file")
    parser.add_argument("-s", "--show", 
        help="Show normalized notes of the students", 
        action="store_true")
    parser.add_argument("-g", "--graph",
        help="Save all compared graph",
        action="store_true")
    args = parser.parse_args()
    file_name = args.file

    try:
        df = Data_treatment(file_name)
    except Exception as e:
        print("Error", e)
        exit()

    summary = {}
    summary = mean_func(summary, df.normalize_data, df.num_features)
    summary = std_func(summary, df.normalize_data, df.num_features)
    summary = quartile_func(summary, df.normalize_data, df.num_features)
    summary['disparite_houses'] = get_disparite_per_house(df)

    tableau = {}
    indicateurs = []
    lst_matieres = []
    for indicateur, matieres in summary.items():
        indicateurs.append(indicateur)
        for matiere, valeur in matieres.items():
            if not matiere in tableau:
                tableau[matiere] = []
                lst_matieres.append(matiere)
            tableau[matiere].append(valeur)

    # Sauvegarde une panoplie de scatter plot dans le dossier scatter_plot
    if args.graph:
        path = os.getcwd()
        try:
            os.mkdir(path + "/scatter_plots")
        except:
            pass
        for index, matiere in enumerate(lst_matieres):
            if index == 0:
                continue
            for index2, matiere2 in enumerate(lst_matieres):
                if index2 <= index:
                    continue
                plt.scatter(tableau[matiere],
                            tableau[matiere2],
                            edgecolor='black', linewidth=1, alpha=0.3)
                plt.title('Ressemblance entre les matieres \n' + matiere + ' et ' + matiere2 + '\n ' + str(indicateurs))
                plt.xlabel(matiere)
                plt.ylabel(matiere2)
                plt.savefig("scatter_plots/" + matiere + "_vs_" + matiere2 + ".png")
                plt.clf()

    plt.scatter(tableau['Astronomy'],
                tableau['Defense Against the Dark Arts'],
                edgecolor='black', linewidth=1, alpha=0.3)

    if args.show:
        plt.scatter(df.normalize_data[:, 1], df.normalize_data[:, 3], edgecolor='black', linewidth=1, alpha=0.75)

    plt.title('Ressemblance entre les matieres \n Astronomy et Defense Against the Dark Arts \n ' + str(indicateurs))
    plt.xlabel('Astronomy')
    plt.ylabel('Defense Against the Dark Arts')
    plt.tight_layout()

    plt.savefig("scatter.png")
    exit()
