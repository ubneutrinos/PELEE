"""
This script is a collection of redundant bits of code that don't exactly belong in the Plotter-NUMU notebook
Lots of these functions modify parameters that save files and modify plots
"""

import os


def get_scaling(ISRUN3, ISG1, scaling=1):
    if ISRUN3:
        if not ISG1: 
            weights = {
                "data": 1 * scaling,
                "mc": 5.70e-03 * scaling,
                "nue": 5.70e-3 * scaling,#should be identical to numu weight, since parsed from same sample
                #"nue": 1.21e-04 * scaling, #weight when using exclusive nue sa3mple
                #"ext": 3.02E-02 * scaling, #for the combined EXT sample
                "ext": 2.52E-02 * scaling, #G1
                "dirt": 2.35e-02 * scaling,
                #"lee": 1.21e-04 * scaling,
            }
            pot = 0.763e19*scaling
        if ISG1:
            weights = {
                "data": 1 * scaling,
                "mc": 0.118 * scaling,
                "nue": 0.118 * scaling,#should be identical to numu weight, since parsed from same sample
                #"nue": 1.21e-04 * scaling, #weight when using exclusive nue sa3mple
                #"ext": 3.02E-02 * scaling, #for the combined EXT sample
                "ext": .520 * scaling, #G1
                "dirt": .486 * scaling,
                #"lee": 1.21e-04 * scaling,
            }
            pot = 1.58E+20*scaling
        else:            
            weights = {
                "data": 1 * scaling,
                "mc": 5.70e-03 * scaling,
                "nue": 5.70e-3 * scaling,#should be identical to numu weight, since parsed from same sample
                #"nue": 1.21e-04 * scaling, #weight when using exclusive nue sa3mple
                #"ext": 3.02E-02 * scaling, #for the combined EXT sample
                "ext": 2.52E-02 * scaling, #G1
                "dirt": 2.35e-02 * scaling,
                #"lee": 1.21e-04 * scaling,
            }
            pot = 0.763e19*scaling
    else:
        weights = {
            "mc": 3.12e-02 * scaling,
            "nue": 7.73e-04 * scaling,
            "ext": 2.69E-01 * scaling, #C+D+E #C only: 1.40e-01
            "dirt": 1.26e-01 * scaling,
            #"lee": 7.73e-04 * scaling,
        }
        pot = 4.08e19*scaling
        
    return weights, pot