from ROOT import *
def calculateErrorBar (num, den):
    if den:
        return TMath.Sqrt(num/(den*den)*(1-num/den)   )
    else:
        return 0



B0 = [1280., 544.]
A0 = [40., 139.]

B1 = [125., 59.] 
A1 = [3., 22.]

for i in xrange(2):
    print calculateErrorBar(A0[i],B0[i]),A0[i]/B0[i], A0[i]/B0[i] - calculateErrorBar(A0[i],B0[i]), A0[i]/B0[i] + calculateErrorBar(A0[i],B0[i])
    print calculateErrorBar(A1[i],B1[i]),A1[i]/B1[i], A1[i]/B1[i] - calculateErrorBar(A1[i],B1[i]), A1[i]/B1[i] + calculateErrorBar(A1[i],B1[i])
    print
