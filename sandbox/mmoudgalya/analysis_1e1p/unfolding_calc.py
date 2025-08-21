import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IntFlux = 104712197684.87015 #for new NuWro nue sample
Ntargets = 40 * 8.77583943e+29 # = 3.51e31

# # delta pt
# binning = [0, 0.3, 1.7]
# bw1 = 0.3
# bw2 = 1.4
# bw = np.array([bw1, bw2])

# # for NuWro FD
# genie_bc = np.array([9809.316256791428,3496.4063603644154])
# nuwro_xs = np.array([35.0752e-38,3.39848e-38]) #from generator study (xsec units is per Ar nucelus, not nucelon)

# # additional smearing matrix for WSVD 2nd derivative
# Ac = np.array([
#     [0.989, 0.007905],
#     [0.01103, 0.9859]
# ])

# alpha3d
binning = [0, 90, 180]
bw1 = 90
bw2 = 90
bw = np.array([bw1, bw2])

# for NuWro FD
genie_bc = np.array([5298.06016919002,8007.6624479661505])
nuwro_xs = np.array([0.0609098e-38, 0.0827589e-38]) #from generator study (xsec units is per Ar nucelus, not nucelon)

# additional smearing matrix for WSVD 2nd derivative
Ac = np.array([
    [0.8913, 0.106],
    [0.1491, 0.8482]
])
# Ac = np.array([
#     [0.9714, 0.02572],
#     [0.04892, 0.9482]
# ])

# Calculation by hand

# convert nuwro xsec to bin counts first

#nuwro_bc = nuwro_xs * bw * Ntargets * IntFlux   # = [386782.97110259, 174887.4673355 ]
nuwro_bc = nuwro_xs * bw * Ntargets/40 * IntFlux   # = [9669.57427756, 4372.18668339]
print("genie bin counts: \n", genie_bc)
print("nuwro bin counts: \n", nuwro_bc)
print()

# apply the additional smearing matrix
genie_smear_bc = Ac.dot(genie_bc)
nuwro_smear_bc = Ac.dot(nuwro_bc)
print("Smeared bin counts:")
print("genie: \n", genie_smear_bc)
print("nuwro: \n", nuwro_smear_bc)
print()

# convert back to xsec (in units that are per nucleon)
genie_smear_xs = (genie_smear_bc / (bw * Ntargets * IntFlux)) * 1e39
nuwro_smear_xs = (nuwro_smear_bc / (bw * Ntargets * IntFlux)) * 1e39

print("Smeared xsec:")
print("genie: \n", genie_smear_xs)
print("nuwro: \n", nuwro_smear_xs)
print()


