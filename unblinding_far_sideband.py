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
#NPVLCUTQ = NPVLCUTQ_all_showers + ' and (n_showers_contained == 1 or (n_showers_contained>1 and shr12_cos_p1_dstart>0.99))'

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
#NPLCUTQ_all_showers += ' and secondshower_Y_nhit < 50' # old cut, no longer in selection
NPLCUTQ_all_showers += ' and tksh_angle > -0.9'
NPLCUTQ_all_showers += ' and shr_trk_len < 300.' # new cut
NPLCUTQ = NPLCUTQ_all_showers + ' and n_showers_contained == 1'
#NPLCUTQ = NPLCUTQ_all_showers + ' and (n_showers_contained == 1 or (n_showers_contained>1 and shr12_cos_p1_dstart>0.99))'

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
#NPTCUTQ = NPTCUTQ_all_showers + ' and (n_showers_contained == 1 or (n_showers_contained>1 and shr12_cos_p1_dstart>0.99))'

# box cuts, for aligned shower test
NPALTESTQ_all_showers = NPPRESQ
NPALTESTQ_all_showers += ' and CosmicIPAll3D > 30.'
NPALTESTQ_all_showers += ' and CosmicDirAll3D > -0.98 and CosmicDirAll3D < 0.98'
NPALTESTQ_all_showers += ' and hits_ratio > 0.65'
NPALTESTQ_all_showers += ' and shr_score < 0.25'
NPALTESTQ_all_showers += ' and shrmoliereavg > 2 and shrmoliereavg < 10'
NPALTESTQ_all_showers += ' and subcluster > 7'
NPALTESTQ_all_showers += ' and trkfit < 0.70'
NPALTESTQ_all_showers += ' and shr_trk_len < 300.'
NPALTESTQ_all_showers += ' and topological_score > 0.8'
NPALTESTQ = NPALTESTQ_all_showers + ' and n_showers_contained == 1'
#NPALTESTQ = NPALTESTQ_all_showers + ' and (n_showers_contained == 1 or (n_showers_contained>1 and shr12_cos_p1_dstart>0.99))'

# BDT cuts
# 0304 extnumi, pi0 and nonpi0
BDTCQ_all_showers = NPLCUTQ_all_showers
BDTCQ_all_showers += ' and pi0_score > 0.67 and nonpi0_score > 0.70'
BDTCQ = BDTCQ_all_showers + ' and n_showers_contained == 1'
#BDTCQ = BDTCQ_all_showers + ' and (n_showers_contained == 1 or (n_showers_contained>1 and shr12_cos_p1_dstart>0.99))'

BDTCQ_only = 'pi0_score > 0.67 and nonpi0_score > 0.70'

# test intermediate BDT cuts
# 0304 extnumi, pi0 and nonpi0
TESTINTBDTCQ_all_showers = NPPRESQ
TESTINTBDTCQ_all_showers += ' and pi0_score > 0.50 and pi0_score < 0.70'
TESTINTBDTCQ = TESTINTBDTCQ_all_showers + ' and n_showers_contained == 1'
TESTINTBDTCQ2 = NPVLCUTQ
TESTINTBDTCQ2 += ' and pi0_score > 0.30 and pi0_score < 0.60'
TESTBDT07CQ_all_showers = NPPRESQ
TESTBDT07CQ_all_showers += ' and pi0_score > 0.70'
TESTBDT02CQ_all_showers = NPPRESQ
TESTBDT02CQ_all_showers += ' and pi0_score > 0.20'
TESTBDT05CQ_all_showers = NPPRESQ
TESTBDT05CQ_all_showers += ' and pi0_score > 0.50'

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
ZPLOOSESEL_all_tracks += ' and shr_trk_sce_start_y > -100 and shr_trk_sce_start_y < 80'
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


ZPPRESEL_all_tracks = PRESQ
ZPPRESEL_onep_track = ZPPRESEL_all_tracks + ' and n_tracks_contained > 0'
ZPPRESEL = ZPPRESEL_all_tracks + ' and n_tracks_contained == 0'
ZPPRESEL_one_shower = ZPPRESEL + 'and n_showers_contained == 1'
PLOOSESEL_all_tracks = ZPPRESEL_all_tracks
ZPLOOSESEL_all_tracks += ' and n_showers_contained == 1'
ZPLOOSESEL_all_tracks += ' and CosmicIPAll3D > 10.'
ZPLOOSESEL_all_tracks += ' and CosmicDirAll3D > -0.9 and CosmicDirAll3D < 0.9'
ZPLOOSESEL_all_tracks += ' and shrmoliereavg < 15'
ZPLOOSESEL_all_tracks += ' and subcluster > 4'
ZPLOOSESEL_all_tracks += ' and trkfit < 0.65'
ZPLOOSESEL_all_tracks += ' and secondshower_Y_nhit < 50'
ZPLOOSESEL_all_tracks += ' and shr_trk_sce_start_y > -100 and shr_trk_sce_start_y < 80'
ZPLOOSESEL_all_tracks += ' and shr_trk_sce_end_y > -100 and shr_trk_sce_end_y < 100 '
ZPLOOSESEL_all_tracks += ' and shr_trk_len < 300.'
ZPLOOSESEL_all_tracks += 'and (n_tracks_tot == 0 or (n_tracks_tot>0 and tk1sh1_angle_alltk>-0.9))'
ZPLOOSESEL_onep_track = ZPLOOSESEL_all_tracks + ' and n_tracks_contained > 0'
ZPLOOSESEL = ZPLOOSESEL_all_tracks + ' and n_tracks_contained == 0'
ZPBDTLOOSE_all_tracks = ZPLOOSESEL_all_tracks
ZPBDTLOOSE_all_tracks += ' and bkg_score >0.72'
ZPBDTLOOSE_onep_track = ZPBDTLOOSE_all_tracks + ' and n_tracks_contained > 0'
ZPBDTLOOSE = ZPBDTLOOSE_all_tracks + ' and n_tracks_contained == 0'
ZPBDTLOOSE += 'and (n_tracks_tot == 0 or (n_tracks_tot>0 and tk1sh1_angle_alltk>-0.9))'

ZPPRESEL_two_shower = ZPPRESEL + 'and n_showers_contained > 1'
ZPLOOSESEL_two_shower = ZPPRESEL_two_shower
ZPLOOSESEL_two_shower += ' and CosmicIPAll3D > 10.'
ZPLOOSESEL_two_shower += ' and CosmicDirAll3D > -0.9 and CosmicDirAll3D < 0.9'
ZPLOOSESEL_two_shower += ' and shrmoliereavg < 15'
ZPLOOSESEL_two_shower += ' and subcluster > 4'
ZPLOOSESEL_two_shower += ' and trkfit < 0.65'
ZPLOOSESEL_two_shower += ' and secondshower_Y_nhit < 50'
ZPLOOSESEL_two_shower += ' and shr_trk_sce_start_y > -100 and shr_trk_sce_start_y < 100'
ZPLOOSESEL_two_shower += ' and shr_trk_sce_end_y > -100 and shr_trk_sce_end_y < 100 '
ZPLOOSESEL_two_shower += ' and shr_trk_len < 300.'
ZPLOOSESEL_two_shower += ' and n_tracks_contained == 0'
ZPBDTLOOSE_two_shower = ZPLOOSESEL_two_shower + 'and bkg_score > 0.72'

# SIDEBANDS CUTS
LOW_PID = '(0.0 < pi0_score < 1.0) and (0.0 < nonpi0_score < 1.0) and ~((pi0_score > 0.1) and (nonpi0_score > 0.1))'
MEDIUM_PID = '(0.1 < pi0_score < 1.0) and (0.1 < nonpi0_score < 1.0) and ~((pi0_score > 0.67) and (nonpi0_score > 0.7))'
LOW_ENERGY = '(0.05 < reco_e < 0.75)'
MEDIUM_ENERGY = '(0.75 < reco_e < 1.05)'
LOW_MEDIUM_ENERGY = '(0.05 < reco_e < 1.05)'
HIGH_ENERGY = '(1.05 < reco_e < 2.05)'
HIGH_ENERGY_ZP = '(reco_e > 0.9)'
HIGH_ENERGY_NOUPBOUND = '(reco_e > 1.05)'
HIGH_ENERGY_EXT = '(0.85 < reco_e < 2.05)'
HIGH_ENERGY_EXT_NOUPBOUND = '(reco_e > 0.85)'
ADD_ENERGY_BINS = '(reco_e > 0.85 and reco_e < 1.05)'
ALL_ENERGY = '(reco_e > 0.)'
TWOP_SHOWERS = 'n_showers_contained >= 2'
LOW_PID_ZP = '(0.0 < bkg_score < 0.4)'


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


# numu selection
NUMUPRESEL = 'nslice == 1'
NUMUPRESEL += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'
NUMUPRESEL += ' and reco_nu_vtx_sce_x > 5 and reco_nu_vtx_sce_x < 251. '
NUMUPRESEL += ' and reco_nu_vtx_sce_y > -110 and reco_nu_vtx_sce_y < 110. '
NUMUPRESEL += ' and reco_nu_vtx_sce_z > 20 and reco_nu_vtx_sce_z < 986. '
#NUMUPRESEL += ' and (reco_nu_vtx_sce_z < 675 or reco_nu_vtx_sce_z > 775) '
NUMUPRESEL += ' and topological_score > 0.06 '
#NUMUPRESEL += ' and contained_fraction > 0.9 '

NUMUCRT = ' and (crtveto != 1 or crthitpe < 100) and _closestNuCosmicDist > 5.'

NUMUPRESELCRT = NUMUPRESEL + NUMUCRT

NUMUSEL = NUMUPRESEL + ' and muon_length > 0'
NUMUSELCRT = NUMUSEL + NUMUCRT


# sideband categories
sideband_categories = {
    'HiEextmax2': {'query': HIGH_ENERGY_EXT, 'title': '0.85 GeV < Reco energy < 2.05 GeV', 'dir': 'HiEextmax2'},
    'HiEmax2': {'query': HIGH_ENERGY, 'title': '1.05 GeV < Reco energy < 2.05 GeV', 'dir': 'HiEmax2'},
    'HiE': {'query': HIGH_ENERGY_NOUPBOUND, 'title': 'Reco energy > 1.05 GeV', 'dir': 'HiE'},
    'HiEext': {'query': HIGH_ENERGY_EXT_NOUPBOUND, 'title': 'Reco energy > 0.85 GeV', 'dir': 'HiEext'},
    'HiEadd': {'query': ADD_ENERGY_BINS, 'title': '0.85 < Reco energy < 1.05 GeV', 'dir': 'HiEadd'},
    'HiEZP': {'query': HIGH_ENERGY_ZP, 'title': 'Reco energy > 0.9 GeV', 'dir': 'HiEZP'},
    'LPID': {'query': LOW_PID, 'title': 'Low BDT', 'dir': 'LPID'},
    'LPIDZP': {'query': LOW_PID_ZP, 'title': 'Low BDT', 'dir': 'LPIDZP'},
    'TwoPShr': {'query': TWOP_SHOWERS, 'title': '2+ showers', 'dir': 'TwoPShr'},
    'TwoPShrHiE': {'query': " and ".join([TWOP_SHOWERS,HIGH_ENERGY_NOUPBOUND]), 'title': '2+ showers,Reco energy > 1.05 GeV', 'dir': 'TwoPShrHiE'},
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
    'ZPOneShr': {'query': ZPPRESEL_one_shower, 'title': '1e0p Presel., 1 shower', 'dir': 'ZPOneShr'},
    'ZPAllTrks': {'query': ZPPRESEL_all_tracks, 'title': '1e0p Presel., 0+ tracks', 'dir': 'ZPAllTrks'},
    'ZPTwoShr': {'query': ZPPRESEL_two_shower, 'title': '1e0p Presel., 2+ shower', 'dir': 'ZPTwoShr'},
    'None': {'query': None, 'title': None, 'dir': 'None'},
}


# selection categories
selection_categories = {
    'NPVL': {'query': NPVLCUTQ, 'title': '1eNp VL cuts', 'dir': 'NPVL'},
    'NPL': {'query': NPLCUTQ, 'title': '1eNp Loose cuts', 'dir': 'NPL'},
    'NPT': {'query': NPTCUTQ, 'title': '1eNp Tight cuts', 'dir': 'NPT'},
    'NPALTEST': {'query': NPALTESTQ, 'title': '1eNp Test cuts', 'dir': 'NPALTEST'},
    'NPBDT': {'query': BDTCQ, 'title': '1eNp BDT sel.', 'dir': 'NPBDT'},
    'NPBDTOnly': {'query': BDTCQ_only, 'title': '1eNp BDT sel. (no Loose)', 'dir': 'NPBDTOnly'},
    'TESTINTBDTCQ2': {'query': TESTINTBDTCQ2, 'title': '1eNp VL + BDT [0.3,0.6]', 'dir': 'TESTINTBDTCQ2'},
    'NPVLAllShr': {'query': NPVLCUTQ_all_showers, 'title': '1eNp VL cuts, 0+ showers', 'dir': 'NPVLAllShr'},
    'NPLAllShr': {'query': NPLCUTQ_all_showers, 'title': '1eNp Loose cuts, 0+ showers', 'dir': 'NPLAllShr'},
    'NPTAllShr': {'query': NPTCUTQ_all_showers, 'title': '1eNp Tight cuts, 0+ showers', 'dir': 'NPTAllShr'},
    'NPBDTAllShr': {'query': BDTCQ_all_showers, 'title': '1eNp BDT sel., 0+ showers', 'dir': 'NPBDTAllShr'},
    'TESTINTBDTCQAllShr': {'query': TESTINTBDTCQ_all_showers, 'title': '1eNp BDT [0.5,0.7], 0+ showers', 'dir': 'TESTINTBDTCQAllShr'},
    'TESTBDT07AllShr': {'query': TESTBDT07CQ_all_showers, 'title': '1eNp BDT > 0.7, 0+ showers', 'dir': 'TESTBDT07AllShr'},
    'TESTBDT05AllShr': {'query': TESTBDT05CQ_all_showers, 'title': '1eNp BDT > 0.5, 0+ showers', 'dir': 'TESTBDT05AllShr'},
    'TESTBDT02AllShr': {'query': TESTBDT02CQ_all_showers, 'title': '1eNp BDT > 0.2, 0+ showers', 'dir': 'TESTBDT02AllShr'},
    'None': {'query': None, 'title': 'NoCuts', 'dir': 'None'},
    'ZPBDT': {'query': ZPBDTLOOSE, 'title': '1e0p BDT sel.', 'dir': 'ZPBDT'},
    'ZPBDTAllTrk': {'query': ZPBDTLOOSE_all_tracks, 'title': '1e0p BDT sel.', 'dir': 'ZPBDTAllTrk'},
    'ZPLAllTrk': {'query': ZPLOOSESEL_all_tracks, 'title': '1e0p Loose sel.', 'dir': 'ZPLAllTrk'},
    'ZPLOOSETWOSHR': {'query': ZPLOOSESEL_two_shower, 'title': '1e0p loose sel. 2+ shr', 'dir': 'ZPLOOSE_two_shower'},
    'ZPBDTTWOSHR': {'query': ZPBDTLOOSE_two_shower, 'title': '1e0p BDT sel. 2+shr', 'dir': 'ZPBDT_two_shower'},
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

basic_variables = [
        ('n_showers_contained',1,(-0.5, 9.5),"normalization","onebin"),
        ('n_showers_contained',10,(-0.5, 9.5),"n showers contained"),
        ('n_tracks_contained',6,(-0.5, 5.5),"n tracks contained"),
        ('n_tracks_tot',6,(-0.5, 5.5),"n tracks total"),
        #('reco_e',21,(0.05,2.15),r"Reconstructed Energy [GeV]"),
        #('reco_e',20,(0.05,3.05),r"Reconstructed Energy [GeV]","extended"),
        #('reco_e',7,(0.05,2.85),r"Reconstructed Energy [GeV]","coarse"),
        ('reco_e',22,(-0.05,2.15),r"Reconstructed Energy [GeV]"),
        ('reco_e',21,(-0.05,4.15),r"Reconstructed Energy [GeV]","extended"),
        ('reco_e',14,(0.15,1.55),r"Reconstructed Energy [GeV]","note"),
        ('reco_e',10,(0.9,3.9),r"Reconstructed Energy [GeV]","highe"),
]
evtsel_variabls = [
        ('hits_ratio',20,(0,1),"shower hits/all hits"),
        ('CosmicIPAll3D',20,(0,200),"CosmicIPAll3D [cm]"),
        ('CosmicDirAll3D',20,(-1,1),"cos(CosmicDirAll3D)"),
        ('tk1sh1_angle_alltk',20,(-1,1),"cos(tk1sh1)"),
]
shrsel_variables = [
        ('trkfit',10,(0,1.0),"Fraction of Track-fitted points"),
        ('shrmoliereavg',20,(0,50),"average Moliere angle [degrees]"),
        #('shrmoliereavg',10,(0,10),"average Moliere angle [degrees]","zoomed")
        ('shr_score',20,(0,0.5),"shr score"),
        ('subcluster',20,(0,40),"N sub-clusters in shower"),
        #('subcluster',20,(0,80),"N sub-clusters in shower","extended"),
        ('secondshower_Y_nhit',20,(0,200),"Nhit 2nd shower (Y)"),
        ('secondshower_Y_dot',20,(-1,1),"cos(2nd shower direction wrt vtx) (Y)"),
        ('anglediff_Y',14,(0,350),"angle diff 1st-2nd shower (Y) [degrees]"),
        ('secondshower_Y_vtxdist',20,(0.,200),"vtx dist 2nd shower (Y)"),
        ('secondshower_V_nhit',20,(0,200),"Nhit 2nd shower (V)"),
        ('secondshower_V_dot',20,(-1,1),"cos(2nd shower direction wrt vtx) (V)"),
        ('anglediff_V',14,(0,350),"angle diff 1st-2nd shower (V) [degrees]"),
        ('secondshower_V_vtxdist',20,(0.,200),"vtx dist 2nd shower (V)"),
        ('secondshower_U_nhit',20,(0,200),"Nhit 2nd shower (U)"),
        ('secondshower_U_dot',20,(-1,1),"cos(2nd shower direction wrt vtx) (U)"),
        ('anglediff_U',14,(0,350),"angle diff 1st-2nd shower (U) [degrees]"),
        ('secondshower_U_vtxdist',20,(0.,200),"vtx dist 2nd shower (U)"),
        ('shr_tkfit_dedx_max',15,(0,10),"shr tkfit dE/dx (max, 0-4 cm) [MeV/cm]"),
        ('shr_trk_sce_start_y',20,(-120,120),"shr_trk_sce_start y [cm]"),
        ('shr_trk_sce_end_y',20,(-120,120),"shr_trk_sce_end y [cm]"),
        ('shr_trk_len',40,(0,400),"Shower track fit length [cm]"),
        ('shr_tkfit_2cm_dedx_Y',10,(0.5,10.5),"shr tkfit dE/dx (Y, 0-2 cm) [MeV/cm]"),
        ('shr_tkfit_2cm_dedx_V',10,(0.5,10.5),"shr tkfit dE/dx (V, 0-2 cm) [MeV/cm]"),
        ('shr_tkfit_2cm_dedx_U',10,(0.5,10.5),"shr tkfit dE/dx (U, 0-2 cm) [MeV/cm]"),
        ('shr_tkfit_gap10_dedx_Y',10,(0.5,10.5),"shr tkfit dE/dx (Y, 1-5 cm) [MeV/cm]"),
        ('shr_tkfit_gap10_dedx_V',10,(0.5,10.5),"shr tkfit dE/dx (V, 1-5 cm) [MeV/cm]"),
        ('shr_tkfit_gap10_dedx_U',10,(0.5,10.5),"shr tkfit dE/dx (U, 1-5 cm) [MeV/cm]"),
]
trksel_variables = [
        ('tksh_angle',20,(-1,1),"cos(trk-shr angle)"),
        ('trkshrhitdist2',20,(0,10),"2D trk-shr distance (Y)"),
        ('tksh_distance',20,(0,40),"trk-shr distance [cm]"),
        #('tksh_distance',12,(0,6),"trk-shr distance [cm]","zoomed")
        ('trkpid',21,(-1,1),"track LLR PID"),
        #('trkpid',2,(-1,1),"track LLR PID", 'twobins'),
        #('trkpid',15,(-1,1),"track LLR PID","coarse")
]
bdtscore_variables = [
        #('nonpi0_score',10,(0.,0.5),"BDT non-$\pi^0$ score", "low_bdt"),
        ('nonpi0_score',10,(0.5,1.0),"BDT non-$\pi^0$ score", "high_bdt"),
        #('nonpi0_score',10,(0,1.0),"BDT non-$\pi^0$ score"),
        ('nonpi0_score',10,(0,1.0),"BDT non-$\pi^0$ score", "log", True),
        #('pi0_score',10,(0.,0.5),"BDT $\pi^0$ score", "low_bdt"),
        ('pi0_score',10,(0.5,1.0),"BDT $\pi^0$ score", "high_bdt"),
        #('pi0_score',10,(0,1.0),"BDT $\pi^0$ score"),
        ('pi0_score',10,(0,1.0),"BDT $\pi^0$ score", "log", True),
        #('bkg_score',10,(0,1.0),"1e0p BDT score"),
        #('bkg_score',10,(0,1.0),"1e0p BDT score", "log", True),
        ('bkg_score',10,(0,1.0),"1e0p BDT score"),
        ('bkg_score',5,(0,0.5),"1e0p BDT score","low_bdt"),
        ('bkg_score',5,(0.5,1.0),"1e0p BDT score","high_bdt"),
        ('bkg_score',10,(0,1.0),"1e0p BDT score", "log", True),
]
energy_variables = [
        ('trk_energy_tot',10,(0,2),"trk energy (range, P) [GeV]"),
        ('shr_energy_tot_cali',10,(0,2),"shr energy (calibrated) [GeV]"),
        #('NeutrinoEnergy0', 20, (0,2000), r"Reconstructed Calorimetric Energy U [MeV]"),
        #('NeutrinoEnergy1', 20, (0,2000), r"Reconstructed Calorimetric Energy V [MeV]"),
        #('NeutrinoEnergy2', 20, (0,2000), r"Reconstructed Calorimetric Energy Y [MeV]"),
]
kinematic_variables = [
        ('protonenergy',12,(0,0.6),"proton kinetic energy [GeV]"),
        ('pt',10,(0,2),"pt [GeV]"),
        ('ptOverP',20,(0,1),"pt/p"),
        ('phi1MinusPhi2',13,(-6.5,6.5),"shr phi - trk phi"),
        ('theta1PlusTheta2',13,(0,6.5),"shr theta + trk theta"),
        ('trk_theta',21,(0,3.14),r"Track $\theta$"),
        ('trk_phi',21,(-3.14, 3.14),r"Track $\phi$"),
        ('trk_len',20,(0,100),"Track length [cm]"),
        ('trk_len',21,(0,20),"Track length [cm]", "zoom"),
        ('shr_theta',21,(0,3.14),r"Shower $\theta$"),
        ('shr_phi',21,(-3.14, 3.14),r"Shower $\phi$"),
        ('n_trks_gt10cm',6,(-0.5, 5.5),"n tracks longer than 10 cm"),
        #('n_trks_gt25cm',6,(-0.5, 5.5),"n tracks longer than 25 cm"),
]
other_variables = [
        ('slclustfrac',20,(0,1),"slice clustered fraction"),
        ('reco_nu_vtx_x',10,(0,260),"vertex x [cm]"),
        ('reco_nu_vtx_y',10,(-120,120),"vertex y [cm]"),
        ('reco_nu_vtx_z',10,(0,1000),"vertex z [cm]"),
        #('slnhits',20,(0.,5000),"N total slice hits"),
        #('hits_u',20,(0.,1000),"N clustered hits U plane"),
        #('hits_v',20,(0.,1000),"N clustered hits V plane"),
        #('hits_y',20,(0.,1000),"N clustered hits Y plane"),
        #('trk_hits_tot',20,(0.,2000),"Total N hits in tracks"),
        #('trk_hits_u_tot',20,(0.,700),"Total N hits in tracks (U)"),
        #('trk_hits_v_tot',20,(0.,700),"Total N hits in tracks (V)"),
        #('trk_hits_y_tot',20,(0.,700),"Total N hits in tracks (Y)"),
        #('shr_hits_tot',20,(0.,2000),"Total N hits in showers"),
        #('shr_hits_u_tot',20,(0.,800),"Total N hits in showers (U)"),
        #('shr_hits_v_tot',20,(0.,800),"Total N hits in showers (V)"),
        #('shr_hits_y_tot',20,(0.,800),"Total N hits in showers (Y)"),
        ('topological_score',20,(0,1),"topological score"),
        ('trk_score',20,(0.5,1.0),"trk score"),
        ('shr_tkfit_nhits_tot',20,(0,20),"shr tkfit nhits (tot, 0-4 cm) [MeV/cm]"),
        #('shrsubclusters0',20,(0,20),"N sub-clusters in shower (U)"),
        #('shrsubclusters1',20,(0,20),"N sub-clusters in shower (V)"),
        #('shrsubclusters2',20,(0,20),"N sub-clusters in shower (Y)"),
        ('shrMCSMom',20,(0, 200),"shr mcs mom [MeV]"),
        ('DeltaRMS2h',20,(0,10),"Median spread of spacepoints"),
        ('CylFrac2h_1cm',20,(0,1),"Frac. of spacepoints in 1cm cylinder (2nd half of shr)"),
        ('shrPCA1CMed_5cm',10,(0.5,1),"Median of 1st component of shr PCA (5cm window)"),
]
pi0_variables = [
        ('pi0_gammadot',20,(-1,1),"$\pi^0$ $\gamma_{\\theta\\theta}$"),
        ('pi0energy',20,(135,1135),"$\pi^0$ Energy [MeV]"),
        ('pi0energyraw',20,(0,1135),"$\pi^0$ Calorimeric Energy $E_1 + E_2$ [MeV]"),
        ('pi0momentum',20,(0,1000),"$\pi^0$ Momentum [MeV]"),
        ('pi0beta',40,(0,1),"$\pi^0$ $\\beta$"),
        ('pi0momanglecos',40,(0,1),"$\pi^0$ $\cos\theta$"),
        ('epicospi',40,(0,1),"$\pi^0$ $\cos\theta$ \times $E_{\pi}$"),
        ('asymm',20,(0,1),"$\pi^0$ asymmetry $\\frac{|E_1-E_2|}{E_1+E_2}$"),
        ('pi0thetacm',20,(0,1),"$\cos\\theta_{\gamma}^{CM} = \\frac{1}{\\beta_{\pi^0}} \\frac{|E_1-E_2|}{E_1+E_2}$"),
        ('pi0_mass_Y',20,(10,510),"$\pi^0$ asymmetry $\pi^0$ mass [MeV]"),
        ('pi0_mass_Y_corr',49,(10,500),"$\pi^0$ asymmetry $\pi^0$ mass [MeV]"),
        ('reco_e',19,(0.15,2.15),"reconstructed energy [GeV]"),
        ('shr_energy_tot_cali',20,(0.05,1.50),"reconstructed shower energy [GeV]"),
        ('trk_energy_tot',20,(0.05,1.50),"reconstructed track energy [GeV]"),
        ('n_tracks_contained',5,(0,5),"number of contained tracks"),
        ('n_showers_contained',5,(2,7),"number of contained showers"),
        ('pi0_mass_U',20,(10,510),"$M_{\gamma\gamma}$ mass U plane [MeV]"),
        ('pi0_mass_V',20,(10,510),"$M_{\gamma\gamma}$ mass V plane [MeV]"),
        ('pi0_mass_Y',20,(10,510),"$M_{\gamma\gamma}$ mass Y plane [MeV]"),
        ('pi0_shrscore1',20,(0,1),"leading $\gamma$ shower score"),
        ('pi0_shrscore2',20,(0,1),"sub-leading $\gamma$ shower score"),
        ('pi0_radlen1',20,(3,103),"leading $\gamma$ shower conversion distance [cm]"),
        ('pi0_radlen2',20,(3,103),"sub-leading $\gamma$ shower conversion distance [cm]"),
        ('pi0_energy1_Y',20,(60,460),"leading $\gamma$ shower energy [MeV]"),
        ('pi0_energy2_Y',20,(40,240),"sub-leading $\gamma$ shower energy [MeV]"),
        ('pi0_dedx1_fit_Y',20,(1.0,11.0),"leading $\gamma$ shower dE/dx [MeV/cm]"),
]
pi0_truth_variables = [
        ('shr_bkt_pdg',20,(5, 25),r"shower backtracked pdg"),
        ('trk_bkt_pdg',20,(5, 2500),r"track backtracked pdg"),
        ('pi0truth_gamma1_dist',20,(0, 100),r"leading photon conv. distance [cm]"),
        ('pi0truth_gamma2_dist',20,(0, 100),r"sub-leading photon conv. distance [cm]"),
        ('pi0truth_gamma1_etot',20,(0, 1000),r"leading photon true Energy [ MeV ]"),
        ('pi0truth_gamma2_etot',20,(0, 500),r"sub-leading photon true Energy [ MeV ]"),
        ('pi0truth_gamma1_edep',20,(0, 1000),r"leading photon deposited Energy [ MeV ]"),
        ('pi0truth_gamma2_edep',20,(0, 500),r"sub-leading photon deposited Energy [ MeV ]"),
        ('pi0truth_gamma1_edep_frac',20,(0, 1),r"leading photon deposited/total Energy"),
        ('pi0truth_gamma2_edep_frac',20,(0, 1),r"sub-leading photon deposited/toal Energy"),
        ('true_nu_vtx_x',12,(0,252),"true vtx x [cm]"),
        ('true_nu_vtx_y',12,(-120,120),"true vtx y [cm]"),
        ('true_nu_vtx_z',12,(0,1200),"true vtx z [cm]"),
        #('weightSplineTimesTune',20,(0,2),"event weight"),
        ('pi0truth_gammadot',20,(-1,1),"cos opening angle"),
        ('muon_e',20,(0., 1.),r"Muon Energy [ GeV ]"),
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

plot_variables = basic_variables + evtsel_variabls + shrsel_variables + bdtscore_variables
plot_variables += kinematic_variables

