import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from matplotlib.ticker import MaxNLocator


if __name__ == "__main__":

    runs = [1,2,3,4,5]
    NoPOT = [[],[],[],[],[]]
    NoPOT_err = [[],[],[],[],[]]

    E_range = ["All energies","Low energy"]
    selections = ["ZPBDT", "ZPLOOSESEL","ZP presel","NPBDT", "NPL", "NP presel"]
    indices = [14,16,18,2,4,6]

    for run in runs:
        with open('CSV_Runs/Results_rundep_run{}.csv'.format(run), mode ='r')as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                #print(lines[3])
                nb = int(run)-1
                NoPOT[nb].append(lines[2])
                NoPOT_err[nb].append(lines[3])

    

    for i in range(0,2):
        Ename = E_range[i]
        for j in range(len(selections)):
            Avg = 0
            for run in runs:
                #print(indices[j]+i*6)
                plt.errorbar(int(run),float(NoPOT[run-1][indices[j]+i*6]),yerr=float(NoPOT_err[run-1][indices[j]+i*6]),xerr=None,marker='o',markeredgecolor='royalblue', linestyle='solid',color='royalblue' , markerfacecolor='royalblue', markersize=10) 
                Avg += float(NoPOT[run-1][indices[j]+i*6])
            plt.xlabel('Run')
            ax = plt.gca()
            ax.set_xticks(ax.get_xticks()[::2])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.ylabel('N/POT')
            plt.title('{}  {}'.format(selections[j],E_range[i]))
            plt.axhline(y=Avg/5, color='r', linestyle='-')
            plt.savefig('Plots/Rundep_{}_{}.pdf'.format(selections[j],E_range[i]),dpi = 1000,bbox_inches = "tight")
            plt.show()
    
    