import os
import uproot
import numpy as np
import time
import csv

class NuMILEEWeights:
    def __init__(self,leeVarFileName="/Users/elenag/Desktop/PlotterLEE/NuMIFlux/biggest_variation_LEEApperanceNuMI_FHC.csv", current="FHC"):
        self.nameDictionary = {14: 'numu', -14: 'numubar', 12: 'nue', -12:'nuebar'}
        self.fullDictionary = None 
        self.energyEdges    = None
        self.angleEdges     = None 
        self.readWeigthsDictionary(current, leeVarFileName)

        
    def readWeigthsDictionary(self, current="FHC",leeVarFileName="/Users/elenag/Desktop/PlotterLEE/NuMIFlux/biggest_variation_LEEApperanceNuMI_FHC.csv"):
    mydict = {}
    with open(leeVarFileName, mode='r') as infile:
        reader = csv.reader(infile)
    mydictNue = {rows[0]:rows[1] for rows in reader}
    print(mydictNue)    


    def calculateLEEWeightKey(self, neutrino, energy):
        key  = self.nameDictionary[neutrino]
        en_v = self.energyEdges[key]
        itemindexE  = np.where(en_v[:-1] <= energy)
        try:
            energyBin   = np.max(itemindexE)
        except:
            print(key,energy)
            print(itemindexE)
        key += "_"+str(en_v[energyBin])
        return(key)

    def calculateGeoWeight(self, neutrino, energy, ):
        key  = self.calculateLEEWeightKey(neutrino, energy)
        w    = self.fullDictionary[key]
        return(w)
