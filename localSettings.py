import os

def get_specifics(sample):
    ntuple_path = ""
    append = ""
    if sample == '0304':
        ntuple_path = "E:\\HEPPA\\Data\\PeLEE\\0304\\"
        append = ""
    if sample == "0304_numupresel":
        ntuple_path = "E:\\HEPPA\\Data\\PeLEE\\0304_numupresel\\"
        append = "_numupresel"
    if sample == "0320_systematics":
        ntuple_path = "E:\\HEPPA\\Data\\PeLEE\\0320_systematics\\"
        append = ""
        
    detsys_pickle = 'reco_nu_e_range_04062020_fullsel_df.pickle'
    return ntuple_path, append, detsys_pickle
        

main_path = "C:\\Users\\Ryan\\python-workspace\\PeLEE-newmcfilteredsamples\\"
RUN1 = "Run1\\"
RUN2 = "Run2\\"
RUN3 = "Run3\\"
G1 = "E:\\HEPPA\\Data\\PeLEE\\G1\\neutrinoselection_filt_G1.root"
G1_POT = 1.58e20
pickle_path = "pickles\\"
plots_path = "plots\\"
#fold = "searchingfornues"
fold = "nuselection"

SAMPLE = '0304_numupresel' ###########CHANGE THIS####################

ntuple_path,APPEND,detsys_pickle = get_specifics(SAMPLE)
    
#----------------------------------------------------
#trimmed sample (faster loading)
#ntuple_path = "E:\\HEPPA\\Data\\PeLEE\\0304_numupresel\\"
#SAMPLE = "0304_numupresel"
#APPEND = "_numupresel"


#---------------------------------------------
#full samples
#ntuple_path = "E:\\HEPPA\\Data\\PeLEE\\0304\\"
#SAMPLE = "0304"
#APPEND = ""