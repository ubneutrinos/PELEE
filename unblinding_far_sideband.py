# nue preselection
PRESQ = 'nslice == 1'
PRESQ += ' and selected == 1'
PRESQ += ' and shr_energy_tot_cali > 0.07'
PRESQ += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'

# 1eNp preselection
NPPRESQ = PRESQ
NPPRESQ += ' and n_tracks_contained > 0'
NPPRESQ_one_shower = NPPRESQ + ' and n_showers_contained == 1'
NPPRESQ_one_shower_one_track = NPPRESQ_one_shower + ' and n_tracks_contained == 1'
NPPRESQ_one_shower_twoplus_tracks = NPPRESQ_one_shower + ' and n_tracks_contained > 1'
NPPRESQ_one_track = NPPRESQ + ' and n_tracks_contained == 1'
NPPRESQ_twoplus_tracks = NPPRESQ + ' and n_tracks_contained > 1'

# 2+ showers preselection
PRESQ_twoplus_showers = PRESQ + ' and n_showers_contained >= 2'
NPPRESQ_twoplus_showers = NPPRESQ + ' and n_showers_contained >= 2'

# very loose box cuts
NPVLCUTQ_all_showers = NPPRESQ
NPVLCUTQ_all_showers += ' and CosmicIPAll3D > 10.'
NPVLCUTQ_all_showers += ' and trkpid < 0.25'
NPVLCUTQ_all_showers += ' and hits_ratio > 0.5'
NPVLCUTQ_all_showers += ' and trkfit < 0.90'
NPVLCUTQ_all_showers += ' and tksh_distance < 10.0'
NPVLCUTQ_all_showers += ' and tksh_angle > -0.9'
NPVLCUTQ = NPVLCUTQ_all_showers + ' and n_showers_contained == 1'

# loose box cuts
NPLCUTQ_all_showers = NPVLCUTQ_all_showers
NPLCUTQ_all_showers += ' and CosmicIPAll3D > 10.'
NPLCUTQ_all_showers += ' and trkpid < 0.02'
NPLCUTQ_all_showers += ' and hits_ratio > 0.50'
NPLCUTQ_all_showers += ' and shrmoliereavg < 9'
NPLCUTQ_all_showers += ' and subcluster > 4'
NPLCUTQ_all_showers += ' and trkfit < 0.65'
NPLCUTQ_all_showers += ' and tksh_distance < 6.0'
NPLCUTQ_all_showers += ' and (shr_tkfit_nhits_tot > 1 and shr_tkfit_dedx_max > 0.5 and shr_tkfit_dedx_max < 5.5)'
NPLCUTQ_all_showers += ' and tksh_angle > -0.9'
NPLCUTQ_all_showers += ' and shr_trk_len < 300.'
NPLCUTQ = NPLCUTQ_all_showers + ' and n_showers_contained == 1'

# tight box cuts
NPTCUTQ_all_showers = NPVLCUTQ_all_showers
NPTCUTQ_all_showers += ' and CosmicIPAll3D > 30.'
NPTCUTQ_all_showers += ' and CosmicDirAll3D > -0.98 and CosmicDirAll3D < 0.98'
NPTCUTQ_all_showers += ' and trkpid < 0.02'
NPTCUTQ_all_showers += ' and hits_ratio > 0.65'
NPTCUTQ_all_showers += ' and shr_score < 0.25'
NPTCUTQ_all_showers += ' and shrmoliereavg > 2 and shrmoliereavg < 10'
NPTCUTQ_all_showers += ' and subcluster > 7'
NPTCUTQ_all_showers += ' and trkfit < 0.70'
NPTCUTQ_all_showers += ' and tksh_distance < 4.0'
NPTCUTQ_all_showers += ' and trkshrhitdist2 < 1.5'
NPTCUTQ_all_showers += ' and (shr_tkfit_nhits_tot > 1 and shr_tkfit_dedx_max > 1.0 and shr_tkfit_dedx_max < 3.8)'
NPTCUTQ_all_showers += ' and (secondshower_Y_nhit<=8 or secondshower_Y_dot<=0.8 or anglediff_Y<=40 or secondshower_Y_vtxdist>=100)'
NPTCUTQ_all_showers += ' and tksh_angle > -0.9 and tksh_angle < 0.70'
NPTCUTQ_all_showers += ' and shr_trk_len < 300.'
NPTCUTQ = NPTCUTQ_all_showers + ' and n_showers_contained == 1'

# BDT cuts
# 0304 extnumi, pi0 and nonpi0
BDTCQ_all_showers = NPLCUTQ_all_showers
BDTCQ_all_showers += ' and pi0_score > 0.67 and nonpi0_score > 0.70'
BDTCQ = BDTCQ_all_showers + ' and n_showers_contained == 1'

#1e0p selection
ZPPRESEL_all_tracks = PRESQ
ZPPRESEL_onep_track = ZPPRESEL_all_tracks + ' and n_tracks_contained > 0'
ZPPRESEL = ZPPRESEL_all_tracks + ' and n_tracks_contained == 0'

ZPBOXCUTS_all_tracks = ZPPRESEL_all_tracks
ZPBOXCUTS_all_tracks += ' and n_showers_contained == 1'
ZPBOXCUTS_all_tracks += ' and shrmoliereavg > 1 and shrmoliereavg < 8'
ZPBOXCUTS_all_tracks += ' and shr_score < 0.05'
ZPBOXCUTS_all_tracks += ' and CosmicIPAll3D > 20. '
ZPBOXCUTS_all_tracks += ' and (CosmicDirAll3D<0.75 and CosmicDirAll3D>-0.75)'
ZPBOXCUTS_all_tracks += ' and trkfit < 0.4'
ZPBOXCUTS_all_tracks += ' and subcluster > 6'
ZPBOXCUTS_all_tracks += " and (shr_tkfit_gap10_dedx_Y>1.5 & shr_tkfit_gap10_dedx_Y<2.5)"
ZPBOXCUTS_all_tracks += " and (shr_tkfit_gap10_dedx_U>1.5 & shr_tkfit_gap10_dedx_U<3.75)"
ZPBOXCUTS_all_tracks += " and (shr_tkfit_gap10_dedx_V>1.5 & shr_tkfit_gap10_dedx_V<3.75)"
ZPBOXCUTS_all_tracks += " and shr_tkfit_2cm_dedx_max>1. and shr_tkfit_2cm_dedx_max<4."
ZPBOXCUTS_all_tracks += ' and shr_trk_len < 300.'
ZPBOXCUTS_onep_track = ZPBOXCUTS_all_tracks + ' and n_tracks_contained > 0'
ZPBOXCUTS = ZPBOXCUTS_all_tracks + ' and n_tracks_contained == 0'

ZPLOOSESEL_all_tracks = ZPPRESEL_all_tracks
ZPLOOSESEL_all_tracks += ' and n_showers_contained == 1'
ZPLOOSESEL_all_tracks += ' and CosmicIPAll3D > 10.'
ZPLOOSESEL_all_tracks += ' and CosmicDirAll3D > -0.9 and CosmicDirAll3D < 0.9'
ZPLOOSESEL_all_tracks += ' and shrmoliereavg < 15'
ZPLOOSESEL_all_tracks += ' and subcluster > 4'
ZPLOOSESEL_all_tracks += ' and trkfit < 0.65'
ZPLOOSESEL_all_tracks += ' and secondshower_Y_nhit < 50'
ZPLOOSESEL_all_tracks += ' and shr_trk_sce_start_y > -100 and shr_trk_sce_start_y < 100'
ZPLOOSESEL_all_tracks += ' and shr_trk_sce_end_y > -100 and shr_trk_sce_end_y < 100 '
ZPLOOSESEL_all_tracks += ' and shr_trk_len < 300.'
ZPLOOSESEL_onep_track = ZPLOOSESEL_all_tracks + ' and n_tracks_contained > 0'
ZPLOOSESEL = ZPLOOSESEL_all_tracks + ' and n_tracks_contained == 0'

ZPBDTVLOOSE_all_tracks = ZPLOOSESEL_all_tracks
ZPBDTVLOOSE_all_tracks += ' and bkg_score >0.5'
ZPBDTVLOOSE_onep_track = ZPBDTVLOOSE_all_tracks + ' and n_tracks_contained > 0'
ZPBDTVLOOSE = ZPBDTVLOOSE_all_tracks + ' and n_tracks_contained == 0'

ZPBDTLOOSE_all_tracks = ZPLOOSESEL_all_tracks
ZPBDTLOOSE_all_tracks += ' and bkg_score >0.72'
ZPBDTLOOSE_onep_track = ZPBDTLOOSE_all_tracks + ' and n_tracks_contained > 0'
ZPBDTLOOSE = ZPBDTLOOSE_all_tracks + ' and n_tracks_contained == 0'

ZPBDT_all_tracks = ZPLOOSESEL_all_tracks
ZPBDT_all_tracks += ' and bkg_score >0.85'
ZPBDT_onep_track = ZPBDT_all_tracks + ' and n_tracks_contained > 0'
ZPBDT = ZPBDT_all_tracks + ' and n_tracks_contained == 0'

# SIDEBANDS CUTS
LOW_PID = '(0.0 < pi0_score < 1.0) and (0.0 < nonpi0_score < 1.0) and ~((pi0_score > 0.1) and (nonpi0_score > 0.1))'
MEDIUM_PID = '(0.1 < pi0_score < 1.0) and (0.1 < nonpi0_score < 1.0) and ~((pi0_score > 0.67) and (nonpi0_score > 0.7))'
LOW_ENERGY = '(0.05 < reco_e < 0.75)'
MEDIUM_ENERGY = '(0.75 < reco_e < 1.05)'
LOW_MEDIUM_ENERGY = '(0.05 < reco_e < 1.05)'
HIGH_ENERGY = '(1.05 < reco_e < 2.05)'
HIGH_ENERGY_NOUPBOUND = '(reco_e > 1.05)'
ALL_ENERGY = '(reco_e > 0.)'
TWOP_SHOWERS = 'n_showers_contained >= 2'

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

# sideband categories
sideband_categories = {
    'HiEmax2': {'query': HIGH_ENERGY, 'title': '1.05 GeV < Reco energy < 2.05 GeV', 'dir': 'HiEmax2'},
    'HiE': {'query': HIGH_ENERGY_NOUPBOUND, 'title': 'Reco energy > 1.05 GeV', 'dir': 'HiE'},
    'LPID': {'query': LOW_PID, 'title': 'Low BDT', 'dir': 'LPID'},
    'TwoPShr': {'query': TWOP_SHOWERS, 'title': '2+ showers', 'dir': 'TwoPShr'},
    'None': {'query': None, 'title': None, 'dir': 'None'},
}

# preselection categories
preselection_categories = {
    'NUE': {'query': PRESQ, 'title': 'Nue Presel.', 'dir': 'NUE'},
    'NP': {'query': NPPRESQ, 'title': '1eNp Presel.', 'dir': 'NP'},
    'NPOneShr': {'query': NPPRESQ_one_shower, 'title': '1eNp Presel., 1 shower', 'dir': 'NPOneShr'},
    'NPOneTrk': {'query': NPPRESQ_one_track, 'title': '1eNp Presel., 1 track', 'dir': 'NPOneTrk'},
    'NPTwoPTrk': {'query': NPPRESQ_twoplus_tracks, 'title': '1eNp Presel., 2+ tracks', 'dir': 'NPTwoPTrk'},
    'ZP': {'query': ZPPRESEL, 'title': '1e0p Presel.', 'dir': 'ZP'},
    'ZPAllTrks': {'query': ZPPRESEL_all_tracks, 'title': '1e0p Presel., 0+ tracks', 'dir': 'ZPAllTrks'},
    'None': {'query': None, 'title': None, 'dir': 'None'},
}


# selection categories
selection_categories = {
    'NPVL': {'query': NPVLCUTQ, 'title': '1eNp VL cuts', 'dir': 'NPVL'},
    'NPL': {'query': NPLCUTQ, 'title': '1eNp Loose cuts', 'dir': 'NPL'},
    'NPT': {'query': NPTCUTQ, 'title': '1eNp Tight cuts', 'dir': 'NPT'},
    'NPBDT': {'query': BDTCQ, 'title': '1eNp BDT sel.', 'dir': 'NPBDT'},
    'NPVLAllShr': {'query': NPVLCUTQ_all_showers, 'title': '1eNp VL cuts, 0+ showers', 'dir': 'NPVLAllShr'},
    'NPLAllShr': {'query': NPLCUTQ_all_showers, 'title': '1eNp Loose cuts, 0+ showers', 'dir': 'NPLAllShr'},
    'NPTAllShr': {'query': NPTCUTQ_all_showers, 'title': '1eNp Tight cuts, 0+ showers', 'dir': 'NPTAllShr'},
    'NPBDTAllShr': {'query': BDTCQ_all_showers, 'title': '1eNp BDT sel., 0+ showers', 'dir': 'NPBDTAllShr'},
    'None': {'query': None, 'title': 'NoCuts', 'dir': 'None'},
    'ZPBDT': {'query': ZPBDTLOOSE, 'title': '1e0p BDT sel.', 'dir': 'ZPBDT'},
}


stages_queries = {
    1 : ' and '.join([HIGH_ENERGY, NPPRESQ_one_shower]),
    2 : ' and '.join([LOW_PID, NPPRESQ_one_shower]),
    3 : ' and '.join([HIGH_ENERGY, NPPRESQ_one_shower, NPVLCUTQ]),
    4 : ' and '.join([HIGH_ENERGY, NPPRESQ_one_shower, NPLCUTQ]),
    5 : ' and '.join([HIGH_ENERGY, NPPRESQ_one_shower, BDTCQ]),
    6 : ' and '.join([HIGH_ENERGY, NPPRESQ_one_shower, ZPBDTVLOOSE]),
}

stages_titles = {
    1 : '1.05 GeV < Reco energy < 2.05 GeV and 1eNp preselection\nN-showers contained == 1',
    2 : 'Low PID and Np preselection cuts',
    3 : '1.05 GeV < Reco energy < 2.05 GeV and 1eNp very loose box cuts',
    4 : '1.05 GeV < Reco energy < 2.05 GeV and 1eNp loose box cuts',
    5 : '1.05 GeV < Reco energy < 2.05 GeV and high 1eNp BDT',
    6 : '1.05 GeV < Reco energy < 2.05 GeV and 0p BDT>0.5',
}

stages_queries_noupbound = {
    1 : ' and '.join([HIGH_ENERGY_NOUPBOUND, NPPRESQ_one_shower]),
    3 : ' and '.join([HIGH_ENERGY_NOUPBOUND, NPPRESQ_one_shower, NPVLCUTQ]),
    4 : ' and '.join([HIGH_ENERGY_NOUPBOUND, NPPRESQ_one_shower, NPLCUTQ]),
    5 : ' and '.join([HIGH_ENERGY_NOUPBOUND, NPPRESQ_one_shower, BDTCQ]),
    6 : ' and '.join([HIGH_ENERGY_NOUPBOUND, NPPRESQ_one_shower, ZPBDTVLOOSE]),
}

stages_titles_noupbound = {
    1 : 'Reco energy > 1.05 GeV and 1eNp preselection\nN-showers contained == 1',
    3 : 'Reco energy > 1.05 GeV and 1eNp very loose box cuts',
    4 : 'Reco energy > 1.05 GeV and 1eNp loose box cuts',
    5 : 'Reco energy > 1.05 GeV and high 1eNp BDT',
    6 : 'Reco energy > 1.05 GeV and 0p BDT>0.5',
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
    0 :  r'$\nu_e$ preselection cuts',
    1 : '1.05 GeV < Reco energy < 2.05 GeV and 1eNp preselection cuts',
    2 : 'Low PID and Np preselection cuts',
    3 : '1.05 GeV < Reco energy < 2.05 GeV and 1eNp very loose box cuts',
    4 : '1.05 GeV < Reco energy < 2.05 GeV and 1eNp loose box cuts',
    5 : '1.05 GeV < Reco energy < 2.05 GeV and high 1eNp BDT',
    6 : '1eNp preselection cuts',
    7 : '1eNp very loose box cuts',
    8 : '1eNp loose box cuts',
    9 : 'high 1eNp BDT',
    10: '1eNp tight box cuts',
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
        ('reco_e',22,(-0.05,2.15),r"Reconstructed Energy [GeV]"),
        ('reco_e',21,(-0.05,4.15),r"Reconstructed Energy [GeV]","extended"),
        ('trk_score',20,(0.5,1.0),"trk score"),
        ('slclustfrac',20,(0,1),"slice clustered fraction"),
        ('reco_nu_vtx_x',20,(0,260),"x"),
        ('reco_nu_vtx_y',20,(-120,120),"y"),
        ('reco_nu_vtx_z',20,(0,1100),"z"),
        ('tksh_angle',20,(-1,1),"cos(trk-shr angle)"),
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
        ('tksh_distance',20,(0,40),"trk-shr distance [cm]"),
        ('shr_tkfit_dedx_max',15,(0,10),"shr tkfit dE/dx (max, 0-4 cm) [MeV/cm]"),
        ('trkpid',21,(-1,1),"track LLR PID"),
        ('trkpid',2,(-1,1),"track LLR PID", 'twobins'),
        ('trk_energy_tot',10,(0,2),"trk energy (range, P) [GeV]"),
        ('shr_energy_tot_cali',10,(0,2),"shr energy (calibrated) [GeV]"),
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
        ('pt',10,(0,2),"pt [GeV]"),
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
        ('topological_score',20,(0,1),"topological score"),
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

shr12_variables = [
        ('hitratio_shr12',10,(0,1),"hit ratio two showers"),
        ('min_tksh_dist',20,(0,40),"min tksh dist of two showers"),
        ('max_tksh_dist',20,(0,40),"max tksh dist of two showers"),
        ('tksh2_dist',20,(0,40),"tksh dist of second shower"),
        ('cos_shr12',10,(-1,1),"cos two showers")
]

run_variables = [
        ('run',100,(4500,19500),"run number"),
]

pi0_variables = [
    ('pi0_mass_U',20,(10,510),"$M_{\gamma\gamma}$ mass U plane [MeV]"),
    ('pi0_mass_V',20,(10,510),"$M_{\gamma\gamma}$ mass V plane [MeV]"),
    ('pi0_mass_Y',20,(10,510),"$M_{\gamma\gamma}$ mass Y plane [MeV]"),
    ]
