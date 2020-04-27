# nue preselection
PRESQ = 'nslice == 1'
PRESQ += ' and selected == 1'
PRESQ += ' and shr_energy_tot_cali > 0.07'
PRESQ += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'

# 1eNp preselection
NPPRESQ = PRESQ
NPPRESQ += ' and n_tracks_contained > 0'
NPPRESEQ_one_shower = NPPRESQ + ' and n_showers_contained == 1'

# 2+ showers preselection
PRESQ_twoplus_showers = PRESQ + ' and n_showers_contained >= 2'
NPPRESQ_twoplus_showers = NPPRESQ + ' and n_showers_contained >= 2'

# very loose box cuts
NPVLCUTQ = NPPRESQ
NPVLCUTQ += ' and CosmicIPAll3D > 10.'
NPVLCUTQ += ' and trkpid < 0.25'
NPVLCUTQ += ' and hits_ratio > 0.5'
NPVLCUTQ += ' and trkfit < 0.90'
# NPVLCUTQ += ' and n_showers_contained == 1'
NPVLCUTQ += ' and tksh_distance < 10.0'
NPVLCUTQ += ' and tksh_angle > -0.9'

# loose box cuts
NPLCUTQ = NPPRESQ
NPLCUTQ += ' and CosmicIPAll3D > 10.'
NPLCUTQ += ' and trkpid < 0.02'
NPLCUTQ += ' and hits_ratio > 0.50'
NPLCUTQ += ' and shrmoliereavg < 9'
NPLCUTQ += ' and subcluster > 4'
NPLCUTQ += ' and trkfit < 0.65'
# NPLCUTQ += ' and n_showers_contained == 1'
NPLCUTQ += ' and tksh_distance < 6.0'
NPLCUTQ += ' and (shr_tkfit_nhits_tot > 1 and shr_tkfit_dedx_max > 0.5 and shr_tkfit_dedx_max < 5.5)'
NPLCUTQ += ' and secondshower_Y_nhit < 50'
NPLCUTQ += ' and tksh_angle > -0.9'

# tight box cuts
NPTCUTQ = NPLCUTQ
NPTCUTQ += ' and CosmicIPAll3D > 30.'
NPTCUTQ += ' and CosmicDirAll3D > -0.98 and CosmicDirAll3D < 0.98'
NPTCUTQ += ' and trkpid < 0.02'
NPTCUTQ += ' and hits_ratio > 0.65'
NPTCUTQ += ' and shr_score < 0.25'
NPTCUTQ += ' and shrmoliereavg > 2 and shrmoliereavg < 10'
NPTCUTQ += ' and subcluster > 7'
NPTCUTQ += ' and trkfit < 0.70'
NPTCUTQ += ' and n_showers_contained == 1'
NPTCUTQ += ' and tksh_distance < 4.0'
NPTCUTQ += ' and trkshrhitdist2 < 1.5'
NPTCUTQ += ' and (shr_tkfit_nhits_tot > 1 and shr_tkfit_dedx_max > 1.0 and shr_tkfit_dedx_max < 3.8)'
NPTCUTQ += ' and (secondshower_Y_nhit<=8 or secondshower_Y_dot<=0.8 or anglediff_Y<=40 or secondshower_Y_vtxdist>=100)'
NPTCUTQ += ' and secondshower_Y_nhit < 30'
NPTCUTQ += ' and tksh_angle > -0.9 and tksh_angle < 0.70'

# BDT cuts
# 0304 extnumi, pi0 and nonpi0
BDTCQ = NPLCUTQ
BDTCQ += ' and pi0_score > 0.67 and nonpi0_score > 0.70'

# SIDEBANDS CUTS
LOW_PID = '(0.0 < pi0_score < 1.0) and (0.0 < nonpi0_score < 1.0) and ~((pi0_score > 0.1) and (nonpi0_score > 0.1))'
MEDIUM_PID = '(0.1 < pi0_score < 1.0) and (0.1 < nonpi0_score < 1.0) and ~((pi0_score > 0.67) and (nonpi0_score > 0.7))'
LOW_ENERGY = '(0.05 < reco_e < 0.75)'
MEDIUM_ENERGY = '(0.75 < reco_e < 1.05)'
HIGH_ENERGY = '(1.05 < reco_e < 2.05)'

stages_queries = {
    1 : ' and '.join([HIGH_ENERGY, NPPRESEQ_one_shower]),
    2 : ' and '.join([LOW_PID, NPPRESEQ_one_shower]),
    3 : ' and '.join([HIGH_ENERGY, NPPRESEQ_one_shower, NPVLCUTQ]),
    4 : ' and '.join([HIGH_ENERGY, NPPRESEQ_one_shower, NPLCUTQ]),
    5 : ' and '.join([HIGH_ENERGY, NPPRESEQ_one_shower, BDTCQ]),
}

stages_queries_two_plus_showers = {
    1 : ' and '.join([HIGH_ENERGY, NPPRESQ_twoplus_showers]),
    2 : ' and '.join([LOW_PID, NPPRESQ_twoplus_showers]),
    3 : ' and '.join([HIGH_ENERGY, NPPRESQ_twoplus_showers, NPVLCUTQ]),
    4 : ' and '.join([HIGH_ENERGY, NPPRESQ_twoplus_showers, NPLCUTQ]),
    5 : ' and '.join([HIGH_ENERGY, NPPRESQ_twoplus_showers, BDTCQ]),
}

stages_titles = {
    1 : 'Stage 1\n1.05 GeV < Reco energy < 2.05 GeV and Np preselection cuts\nN showers contained == 1',
    2 : 'Stage 2\nLow PID and Np preselection cuts\nN showers contained == 1',
    3 : 'Stage 3\n1.05 GeV < Reco energy < 2.05 GeV and Np very loose box cuts',
    4 : 'Stage 4\n1.05 GeV < Reco energy < 2.05 GeV and Np loose box cuts',
    5 : 'Stage 5\n1.05 GeV < Reco energy < 2.05 GeV and high PID',
}

stages_titles_two_plus_showers = {
    1 : 'Stage 1\n1.05 GeV < Reco energy < 2.05 GeV and Np preselection cuts',
    2 : 'Stage 2\nLow PID and Np preselection cuts',
    3 : 'Stage 3\n1.05 GeV < Reco energy < 2.05 GeV and Np very loose box cuts',
    4 : 'Stage 4\n1.05 GeV < Reco energy < 2.05 GeV and Np loose box cuts',
    5 : 'Stage 5\n1.05 GeV < Reco energy < 2.05 GeV and high PID',
}


bdt_scan = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]

plot_variables = {
    'insensitive_variables': [
        ('n_showers_contained',10,(-0.5, 9.5),"n showers contained"),
        ('n_tracks_contained',6,(-0.5, 5.5),"n tracks contained"),
        ('trk_score',20,(0.5,1.0),"trk score"),
        ('slclustfrac',20,(0,1),"slice clustered fraction"),
        ('reco_nu_vtx_x',20,(0,260),"x"),
        ('reco_nu_vtx_y',20,(-120,120),"y"),
        ('reco_nu_vtx_z',20,(0,1100),"z"),
    ],
    'input_bdt': [
        ('tksh_angle',20,(-1,1),"cos(tksh angle)"),
        ('trkfit',9,(0,1.0),"Fraction of Track-fitted points"),
        ('shrmoliereavg',20,(0,50),"average Moliere angle [degrees]"),
#         ('shr_score',20,(0,0.5),"shr score"),
         ('hits_ratio',20,(0,1),"shower hits/all hits"),
#         ('trkshrhitdist2',20,(0,10),"2D trk-shr distance (Y)"),
#         ('subcluster',20,(0,20),"N sub-clusters in shower"),
#         ('secondshower_Y_nhit',20,(0,200),"Nhit 2nd shower (Y)"),
#         ('secondshower_Y_dot',20,(-1,1),"cos(2nd shower direction wrt vtx) (Y)"),
#         ('anglediff_Y',20,(0,350),"angle diff 1st-2nd shower (Y) [degrees]"),
#         ('secondshower_Y_vtxdist',20,(0.,200),"vtx dist 2nd shower (Y)"),
#         ('CosmicIPAll3D',20,(0,200),"CosmicIPAll3D [cm]"),
#         ('CosmicDirAll3D',20,(-1,1),"cos(CosmicDirAll3D)"),
        ('tksh_distance',20,(0,30),"tksh distance [cm]"),
        ('shr_tkfit_dedx_max',15,(0,10),"shr tkfit dE/dx (max, 0-4 cm) [MeV/cm]"),
        ('trkpid',21,(-1,1),"track LLR PID"),
        ('shr_energy_tot_cali',20,(0,1),"shr energy (calibrated) [GeV]"),
#         ('hits_y',20,(0.,1000),"N hits Y plane"),
#         ('shr_tkfit_dedx_Y',20,(0,10),"shr tkfit dE/dx (Y, 0-4 cm) [MeV/cm]"),
#         ('shr_tkfit_dedx_U',20,(0,10),"shr tkfit dE/dx (U, 0-4 cm) [MeV/cm]"),
#         ('shr_tkfit_dedx_V',20,(0,10),"shr tkfit dE/dx (V, 0-4 cm) [MeV/cm]"),
        ('shr_tkfit_nhits_tot',20,(0,20),"shr tkfit nhits (tot, 0-4 cm) [MeV/cm]"),
        
    ],
    'bdt_scores': [
#         ('nonpi0_score',5,(0.,0.5),"BDT non-$\pi^0$ score"),
#         ('nonpi0_score',5,(0.5,1.0),"BDT non-$\pi^0$ score"),
        ('nonpi0_score',10,(0,1.0),"BDT non-$\pi^0$ score"),
#         ('pi0_score',5,(0.,0.5),"BDT $\pi^0$ score"),
#         ('pi0_score',5,(0.5,1.0),"BDT $\pi^0$ score"),
        ('pi0_score',10,(0,1.0),"BDT $\pi^0$ score"),
        ('reco_e',22,(-0.05,2.15),r"Reconstructed Energy [GeV]"),
    ]
}
