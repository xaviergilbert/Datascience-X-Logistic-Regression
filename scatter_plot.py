import sys
import matplotlib.pyplot as plt
from data_treatment_class import Data_treatment
from describe import mean_func, std_func, quartile_func
from histogram import get_disparite_per_house


#Quelles sont les deux features qui sont semblables ?

if __name__ == "__main__":
    file = sys.argv[1]
    df = Data_treatment(file)

    summary = {}
    summary = mean_func(summary, df.normalize_data, df.features)
    summary = std_func(summary, df.normalize_data, df.features)
    summary = quartile_func(summary, df.normalize_data, df.features)
    summary['disparite_houses'] = get_disparite_per_house(df)
    # print(summary['mean'])

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
    # print(tableau)
    # print(indicateurs)

    # Sauvegarde une panoplie de scatter plot dans le dossier scatter_plot
    ######################################################################
    # for index, matiere in enumerate(lst_matieres):
    #     for index2, matiere2 in enumerate(lst_matieres):
    #         if index2 <= index:
    #             if lst_matieres[index] == lst_matieres[-1]:
    #                 break
    #             matiere2 = lst_matieres[index + 1]
    #         plt.scatter(tableau[matiere],
    #                     tableau[matiere2],
    #                     edgecolor='black', linewidth=1, alpha=0.3)
    #         plt.title('Ressemblance entre les matieres \n' + matiere + ' et ' + matiere2 + '\n ' + str(indicateurs))
    #         plt.xlabel(matiere)
    #         plt.ylabel(matiere2)
    #         plt.savefig("scatter_plots/" + matiere + "_vs_" + matiere2 + ".png")
    #         plt.clf()

    plt.scatter(tableau['Arithmancy'],
                tableau['Care of Magical Creatures'],
                edgecolor='black', linewidth=1, alpha=0.3)

    # plt.scatter(df.normalize_data[:, 1], df.normalize_data[:, 5], edgecolor='black', linewidth=1, alpha=0.75)

    plt.title('Ressemblance entre les matieres \n Arithmancy et Care of Magical Creatures \n ' + str(indicateurs))
    plt.xlabel('Arithmancy')
    plt.ylabel('Care of Magical Creatures')
    plt.tight_layout()

    plt.savefig("scatter.png")
    exit()
