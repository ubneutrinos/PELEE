# nue preselection
PRESQ = 'nslice == 1'
PRESQ += ' and selected == 1'
PRESQ += ' and shr_energy_tot_cali > 0.07'
PRESQ += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'

# 1eNp preselection
NPPRESQ = PRESQ
NPPRESQ += ' and n_tracks_contained > 0'
NPPRESEQ_one_shower = NPPRESQ + ' and n_showers_contained == 1'
NPPRESEQ_one_shower_one_track = NPPRESEQ_one_shower + ' and n_tracks_contained == 1'
NPPRESEQ_one_shower_twoplus_tracks = NPPRESEQ_one_shower + ' and n_tracks_contained > 1'

# 2+ showers preselection
PRESQ_twoplus_showers = PRESQ + ' and n_showers_contained >= 2'
NPPRESQ_twoplus_showers = NPPRESQ + ' and n_showers_contained >= 2'

# very loose box cuts
NPVLCUTQ = NPPRESQ
NPVLCUTQ += ' and CosmicIPAll3D > 10.'
NPVLCUTQ += ' and trkpid < 0.25'
NPVLCUTQ += ' and hits_ratio > 0.5'
NPVLCUTQ += ' and trkfit < 0.90'
#NPVLCUTQ += ' and n_showers_contained == 1'
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
#NPLCUTQ += ' and secondshower_Y_nhit < 50'
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
#NPTCUTQ += ' and n_showers_contained == 1'
NPTCUTQ += ' and tksh_distance < 4.0'
NPTCUTQ += ' and trkshrhitdist2 < 1.5'
NPTCUTQ += ' and (shr_tkfit_nhits_tot > 1 and shr_tkfit_dedx_max > 1.0 and shr_tkfit_dedx_max < 3.8)'
NPTCUTQ += ' and (secondshower_Y_nhit<=8 or secondshower_Y_dot<=0.8 or anglediff_Y<=40 or secondshower_Y_vtxdist>=100)'
#NPTCUTQ += ' and secondshower_Y_nhit < 30'
NPTCUTQ += ' and tksh_angle > -0.9 and tksh_angle < 0.70'

# BDT cuts
# 0304 extnumi, pi0 and nonpi0
BDTCQ = NPLCUTQ
BDTCQ += ' and pi0_score > 0.67 and nonpi0_score > 0.70'

#1e0p selection
ZPPRESEL = PRESQ
ZPPRESEL += ' and n_tracks_contained == 0'
ZPPRESEL += ' and n_showers_contained > 0'
ZPBOXCUTS = ZPPRESEL
ZPBOXCUTS += ' and n_showers_contained == 1'
ZPBOXCUTS += ' and shrmoliereavg > 1 and shrmoliereavg < 8'
ZPBOXCUTS += ' and shr_score < 0.05'
ZPBOXCUTS += ' and CosmicIPAll3D > 20. '
ZPBOXCUTS += ' and (CosmicDirAll3D<0.75 and CosmicDirAll3D>-0.75)'
ZPBOXCUTS += ' and trkfit < 0.4'
ZPBOXCUTS += ' and subcluster > 6'
ZPBOXCUTS += " and (shr_tkfit_gap10_dedx_Y>1.5 & shr_tkfit_gap10_dedx_Y<2.5)"
ZPBOXCUTS += " and (shr_tkfit_gap10_dedx_U>1.5 & shr_tkfit_gap10_dedx_U<3.75)"
ZPBOXCUTS += " and (shr_tkfit_gap10_dedx_V>1.5 & shr_tkfit_gap10_dedx_V<3.75)"
ZPBOXCUTS += " and shr_tkfit_2cm_dedx_max>1. and shr_tkfit_2cm_dedx_max<4."
ZPLOOSESEL = ZPPRESEL
ZPLOOSESEL += ' and n_showers_contained == 1'
ZPLOOSESEL += ' and CosmicIPAll3D > 10.'
ZPLOOSESEL += ' and CosmicDirAll3D > -0.9 and CosmicDirAll3D < 0.9'
ZPLOOSESEL += ' and shrmoliereavg < 15'
ZPLOOSESEL += ' and subcluster > 4'
ZPLOOSESEL += ' and trkfit < 0.65'
ZPLOOSESEL += ' and secondshower_Y_nhit < 50'
ZPLOOSESEL += ' and shr_trk_sce_start_y > -100 and shr_trk_sce_start_y < 100'
ZPLOOSESEL += ' and shr_trk_sce_end_y > -100 and shr_trk_sce_end_y < 100 '
ZPBDTVLOOSE = ZPLOOSESEL
ZPBDTVLOOSE += ' and bkg_0p_score >0.5'
ZPBDTLOOSE = ZPLOOSESEL
ZPBDTLOOSE += ' and bkg_0p_score >0.72'
ZPBDT = ZPLOOSESEL
ZPBDT += ' and bkg_0p_score >0.85'

# SIDEBANDS CUTS
LOW_PID = '(0.0 < pi0_score < 1.0) and (0.0 < nonpi0_score < 1.0) and ~((pi0_score > 0.1) and (nonpi0_score > 0.1))'
MEDIUM_PID = '(0.1 < pi0_score < 1.0) and (0.1 < nonpi0_score < 1.0) and ~((pi0_score > 0.67) and (nonpi0_score > 0.7))'
LOW_ENERGY = '(0.05 < reco_e < 0.75)'
MEDIUM_ENERGY = '(0.75 < reco_e < 1.05)'
LOW_MEDIUM_ENERGY = '(0.05 < reco_e < 1.05)'
HIGH_ENERGY = '(1.05 < reco_e < 2.05)'
ALL_ENERGY = '(reco_e > 0.)'

# pi0 selection
SCORECUT = 0.5 # 0.75 #75 # max track score
DVTX = 3.0 # 3. # distance from vertex of each shower
VTXDOT = 0.8 # dot product between each shower's direction and the vtx -> shr start vector
EMIN1 =  60 #60 # leading photon min energy
EMIN2 =  40 #40 #20. # 20. # subleading photon min energy
GAMMADOT = 0.94 # max dot product between showres
DEDXCUT = 1.0 # MeV/cm cut on leading shower only
PI0SEL = 'nslice == 1'
PI0SEL += ' & pi0_shrscore1 < %f & pi0_shrscore2 < %f'%(SCORECUT,SCORECUT)
PI0SEL += '& pi0_dot1  > %f & pi0_dot2 > %f '%(VTXDOT,VTXDOT)
PI0SEL += ' & pi0_radlen1 > %f & pi0_radlen2 > %f & pi0_gammadot < %f '%(DVTX,DVTX,GAMMADOT)
PI0SEL += ' & pi0_energy1_Y > %f & pi0_energy2_Y > %f'%(EMIN1,EMIN2)
#PI0SEL += ' and (filter_pi0 == 1 or bnbdata==1 or extdata==1)'
#PI0SEL += ' and (filter_pi0 == 1)'
PI0SEL += ' and pi0_dedx1_fit_Y >= %f'%DEDXCUT

stages_queries = {
    1 : ' and '.join([HIGH_ENERGY, NPPRESEQ_one_shower]),
    2 : ' and '.join([LOW_PID, NPPRESEQ_one_shower]),
    3 : ' and '.join([HIGH_ENERGY, NPPRESEQ_one_shower, NPVLCUTQ]),
    4 : ' and '.join([HIGH_ENERGY, NPPRESEQ_one_shower, NPLCUTQ]),
    5 : ' and '.join([HIGH_ENERGY, NPPRESEQ_one_shower, ZPBDTVLOOSE]),
    6 : ' and '.join([HIGH_ENERGY, NPPRESEQ_one_shower, BDTCQ]),
}

stages_titles = {
    1 : 'Stage 1\n1.05 GeV < Reco energy < 2.05 GeV and Np preselection cuts\nN showers contained == 1',
    2 : 'Stage 2\nLow PID and Np preselection cuts\nN showers contained == 1',
    3 : 'Stage 3\n1.05 GeV < Reco energy < 2.05 GeV and Np very loose box cuts',
    4 : 'Stage 4\n1.05 GeV < Reco energy < 2.05 GeV and Np loose box cuts',
    5 : 'Stage 5\n1.05 GeV < Reco energy < 2.05 GeV and high PID',
    6 : 'Stage 6\n1.05 GeV < Reco energy < 2.05 GeV and 0p BDT>0.5',
}

stages_queries_two_plus_showers = {
    0 : PRESQ_twoplus_showers,
    1 : ' and '.join([HIGH_ENERGY, NPPRESQ_twoplus_showers]),
    2 : ' and '.join([LOW_PID, NPPRESQ_twoplus_showers]),
    3 : ' and '.join([HIGH_ENERGY, NPPRESQ_twoplus_showers, NPVLCUTQ]),
    4 : ' and '.join([HIGH_ENERGY, NPPRESQ_twoplus_showers, NPLCUTQ]),
    5 : ' and '.join([HIGH_ENERGY, NPPRESQ_twoplus_showers, BDTCQ]),
    6 : ' and '.join([ALL_ENERGY, NPPRESQ_twoplus_showers]),
    7 : ' and '.join([ALL_ENERGY, NPPRESQ_twoplus_showers, NPVLCUTQ]),
    8 : ' and '.join([ALL_ENERGY, NPPRESQ_twoplus_showers, NPLCUTQ]),
    9 : ' and '.join([ALL_ENERGY, NPPRESQ_twoplus_showers, BDTCQ]),
    10: ' and '.join([ALL_ENERGY, NPPRESQ_twoplus_showers, NPTCUTQ]),
}

stages_titles_two_plus_showers = {
    0 : 'Stage 0\n' + r'$\nu_e$ preselection cuts',
    1 : 'Stage 1\n1.05 GeV < Reco energy < 2.05 GeV and Np preselection cuts',
    2 : 'Stage 2\nLow PID and Np preselection cuts',
    3 : 'Stage 3\n1.05 GeV < Reco energy < 2.05 GeV and Np very loose box cuts',
    4 : 'Stage 4\n1.05 GeV < Reco energy < 2.05 GeV and Np loose box cuts',
    5 : 'Stage 5\n1.05 GeV < Reco energy < 2.05 GeV and high PID',
    6 : 'Stage 6\nNp preselection cuts',
    7 : 'Stage 7\nNp very loose box cuts',
    8 : 'Stage 8\nNp loose box cuts',
    9 : 'Stage 9\nhigh PID',
    10: 'Stage10\nNp tight box cuts',
}

stages_queries_two_plus_showers_low_medium_energy = {
    1 : ' and '.join([LOW_MEDIUM_ENERGY, NPPRESQ_twoplus_showers]),
    3 : ' and '.join([LOW_MEDIUM_ENERGY, NPPRESQ_twoplus_showers, NPVLCUTQ]),
    4 : ' and '.join([LOW_MEDIUM_ENERGY, NPPRESQ_twoplus_showers, NPLCUTQ]),
    5 : ' and '.join([LOW_MEDIUM_ENERGY, NPPRESQ_twoplus_showers, BDTCQ]),
}

stages_titles_two_plus_showers_low_medium_energy = {
    1 : 'Stage 1\n0.05 GeV < Reco energy < 1.05 GeV and Np preselection cuts',
    3 : 'Stage 3\n0.05 GeV < Reco energy < 1.05 GeV and Np very loose box cuts',
    4 : 'Stage 4\n0.05 GeV < Reco energy < 1.05 GeV and Np loose box cuts',
    5 : 'Stage 5\n0.05 GeV < Reco energy < 1.05 GeV and high PID',
}

stages_queries_two_plus_showers_no_high_en_cut = {
    1 : ' and '.join([NPPRESQ_twoplus_showers]),
    3 : ' and '.join([NPPRESQ_twoplus_showers, NPVLCUTQ]),
    4 : ' and '.join([NPPRESQ_twoplus_showers, NPLCUTQ]),
    5 : ' and '.join([NPPRESQ_twoplus_showers, BDTCQ]),
}

stages_titles_two_plus_showers_no_high_en_cut = {
    1 : 'Stage 1\nNp preselection cuts',
    3 : 'Stage 3\nNp very loose box cuts',
    4 : 'Stage 4\nNp loose box cuts',
    5 : 'Stage 5\nhigh PID',
}

bdt_scan = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]

plot_variables = [
        ('n_showers_contained',1,(-0.5, 9.5),"normalization","onebin"),
        ('n_showers_contained',10,(-0.5, 9.5),"n showers contained"),
        ('n_tracks_contained',6,(-0.5, 5.5),"n tracks contained"),
        ('trk_score',20,(0.5,1.0),"trk score"),
        ('slclustfrac',20,(0,1),"slice clustered fraction"),
        ('reco_nu_vtx_x',20,(0,260),"x"),
        ('reco_nu_vtx_y',20,(-120,120),"y"),
        ('reco_nu_vtx_z',20,(0,1100),"z"),
        ('tksh_angle',20,(-1,1),"cos(tksh angle)"),
        ('trkfit',10,(0,1.0),"Fraction of Track-fitted points"),
        ('shrmoliereavg',20,(0,50),"average Moliere angle [degrees]"),
        ('shr_score',20,(0,0.5),"shr score"),
        ('hits_ratio',20,(0,1),"shower hits/all hits"),
        ('trkshrhitdist2',20,(0,10),"2D trk-shr distance (Y)"),
        ('subcluster',20,(0,40),"N sub-clusters in shower"),
        ('secondshower_Y_nhit',20,(0,200),"Nhit 2nd shower (Y)"),
        ('secondshower_Y_dot',20,(-1,1),"cos(2nd shower direction wrt vtx) (Y)"),
        ('anglediff_Y',20,(0,350),"angle diff 1st-2nd shower (Y) [degrees]"),
        ('secondshower_Y_vtxdist',20,(0.,200),"vtx dist 2nd shower (Y)"),
        ('CosmicIPAll3D',20,(0,200),"CosmicIPAll3D [cm]"),
        ('CosmicDirAll3D',20,(-1,1),"cos(CosmicDirAll3D)"),
        ('tksh_distance',20,(0,40),"tksh distance [cm]"),
        ('shr_tkfit_dedx_max',15,(0,10),"shr tkfit dE/dx (max, 0-4 cm) [MeV/cm]"),
        ('trkpid',21,(-1,1),"track LLR PID"),
        ('trkpid',2,(-1,1),"track LLR PID", 'twobins'),
        ('trk_energy_tot',20,(0,1),"trk energy (range, P) [GeV]"),
        ('shr_energy_tot_cali',20,(0,2),"shr energy (calibrated) [GeV]"),
        ('shr_tkfit_nhits_tot',20,(0,20),"shr tkfit nhits (tot, 0-4 cm) [MeV/cm]"),
        ('protonenergy',12,(0,0.6),"proton kinetic energy [GeV]"),
        #('NeutrinoEnergy0', 20, (0,2000), r"Reconstructed Calorimetric Energy U [MeV]"),
        #('NeutrinoEnergy1', 20, (0,2000), r"Reconstructed Calorimetric Energy V [MeV]"),
        ('NeutrinoEnergy2', 20, (0,2000), r"Reconstructed Calorimetric Energy Y [MeV]"),
        #('slnhits',20,(0.,5000),"N total slice hits"),
        #('hits_u',20,(0.,1000),"N clustered hits U plane"),
        #('hits_v',20,(0.,1000),"N clustered hits V plane"),
        #('hits_y',20,(0.,1000),"N clustered hits Y plane"),
        ('shr_trk_sce_start_y',20,(-120,120),"shr_trk_sce_start y"),
        ('shr_trk_sce_end_y',20,(-120,120),"shr_trk_sce_end y"),
        ('pt',20,(0,2),"pt [GeV]"),
        ('ptOverP',20,(0,1),"pt/p"),
        ('phi1MinusPhi2',13,(-6.5,6.5),"shr phi - trk phi"),
        ('theta1PlusTheta2',13,(0,6.5),"shr theta + trk theta"),
        #('trk_hits_tot',20,(0.,2000),"Total N hits in tracks"),
        #('trk_hits_u_tot',20,(0.,700),"Total N hits in tracks (U)"),
        #('trk_hits_v_tot',20,(0.,700),"Total N hits in tracks (V)"),
        #('trk_hits_y_tot',20,(0.,700),"Total N hits in tracks (Y)"),
        #('shr_hits_tot',20,(0.,2000),"Total N hits in showers"),
        #('shr_hits_u_tot',20,(0.,800),"Total N hits in showers (U)"),
        #('shr_hits_v_tot',20,(0.,800),"Total N hits in showers (V)"),
        #('shr_hits_y_tot',20,(0.,800),"Total N hits in showers (Y)"),
        #('shrsubclusters0',20,(0,20),"N sub-clusters in shower (U)"),
        #('shrsubclusters1',20,(0,20),"N sub-clusters in shower (V)"),
        #('shrsubclusters2',20,(0,20),"N sub-clusters in shower (Y)"),
        ('trk_theta',21,(0,3.14),r"Track $\theta$"),
        ('trk_phi',21,(-3.14, 3.14),r"Track $\phi$"),
        ('trk_len',20,(0,100),"Track length [cm]"),
        ('trk_len',21,(0,20),"Track length [cm]", "zoom"),
        ('shr_theta',21,(0,3.14),r"Shower $\theta$"),
        ('shr_phi',21,(-3.14, 3.14),r"Shower $\phi$"),
        ('nonpi0_score',10,(0.,0.5),"BDT non-$\pi^0$ score", "low_bdt"),
        ('nonpi0_score',10,(0.5,1.0),"BDT non-$\pi^0$ score", "high_bdt"),
        ('nonpi0_score',10,(0,1.0),"BDT non-$\pi^0$ score"),
        ('nonpi0_score',10,(0,1.0),"BDT non-$\pi^0$ score", "log", True),
        ('pi0_score',10,(0.,0.5),"BDT $\pi^0$ score", "low_bdt"),
        ('pi0_score',10,(0.5,1.0),"BDT $\pi^0$ score", "high_bdt"),
        ('pi0_score',10,(0,1.0),"BDT $\pi^0$ score"),
        ('pi0_score',10,(0,1.0),"BDT $\pi^0$ score", "log", True),
        ('bkg_score',10,(0,1.0),"1e0p BDT score"),
        ('bkg_score',10,(0,1.0),"1e0p BDT score", "log", True),
        ('reco_e',22,(-0.05,2.15),r"Reconstructed Energy [GeV]"),
### Pi0 variables
#         ('pi0_gammadot',20,(-1,1),"$\pi^0$ $\gamma_{\\theta\\theta}$"),
#         ('pi0energy',20,(135,1135),"$\pi^0$ Energy [MeV]"),
#         ('asymm',20,(0,1),"$\pi^0$ asymmetry $\\frac{|E_1-E_2|}{E_1+E_2}$"),
#         ('pi0thetacm',20,(0,1),"$\cos\\theta_{\gamma}^{CM} = \\frac{1}{\\beta_{\pi^0}} \\frac{|E_1-E_2|}{E_1+E_2}$"),
#         ('pi0_mass_Y',20,(10,510),"$\pi^0$ asymmetry $\pi^0$ mass [MeV]"),
#         ('reco_e',19,(0.15,2.15),"reconstructed energy [GeV]"),
#         ('shr_energy_tot_cali',20,(0.05,1.50),"reconstructed shower energy [GeV]"),
#         ('trk_energy_tot',20,(0.05,1.50),"reconstructed track energy [GeV]"),
#         ('n_tracks_contained',5,(0,5),"number of contained tracks"),
#         ('n_showers_contained',5,(2,7),"number of contained showers"),
        ]

pi0_variables = [
    ('pi0_mass_U',20,(10,510),"$M_{\gamma\gamma}$ mass U plane [MeV]"),
    ('pi0_mass_V',20,(10,510),"$M_{\gamma\gamma}$ mass V plane [MeV]"),
    ('pi0_mass_Y',20,(10,510),"$M_{\gamma\gamma}$ mass Y plane [MeV]"),
    ]
