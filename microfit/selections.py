import re
from typing import List, Tuple


# pi0 preselection
PREPI0Q = 'nslice == 1'
PREPI0Q += ' and contained_fraction > 0.4'
PREPI0Q += ' and shr_energy_tot_cali > 0.07'
PREPI0Q += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'

# nue preselection
PRESQ = 'nslice == 1'
PRESQ += ' and selected == 1'
PRESQ += ' and shr_energy_tot_cali > 0.07'
PRESQ += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'

# 1e1p preselection
OnePPRESQ = PRESQ
OnePPRESQ += ' and n_tracks_contained == 1 and n_showers_contained == 1'

# 1e1p selection (loose box cuts, same as 1eNp loose box cuts)
OnePLCUTQ = OnePPRESQ
OnePLCUTQ += ' and CosmicIPAll3D > 10.'
OnePLCUTQ += ' and trkpid < 0.02'
OnePLCUTQ += ' and hits_ratio > 0.50'
OnePLCUTQ += ' and shrmoliereavg < 9'
OnePLCUTQ += ' and subcluster > 4'
OnePLCUTQ += ' and trkfit < 0.65'
OnePLCUTQ += ' and tksh_distance < 6.0'
OnePLCUTQ += ' and (shr_tkfit_nhits_tot > 1 and shr_tkfit_dedx_max > 0.5 and shr_tkfit_dedx_max < 5.5)' 
OnePLCUTQ += ' and tksh_angle > -0.9'
OnePLCUTQ += ' and shr_trk_len < 300.'

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
NPVLCUTQ_two_shower = NPVLCUTQ_all_showers + ' and n_showers_contained > 1'
#NPVLCUTQ = NPVLCUTQ_all_showers + ' and (n_showers_contained == 1 or (n_showers_contained>1 and shr12_cos_p1_dstart>0.99))'

# loose box cuts
NPLCUTQ_all_showers = NPPRESQ
NPLCUTQ_all_showers += ' and CosmicIPAll3D > 10.'
NPLCUTQ_all_showers += ' and trkpid < 0.02'
NPLCUTQ_all_showers += ' and hits_ratio > 0.50'
NPLCUTQ_all_showers += ' and shrmoliereavg < 9'
NPLCUTQ_all_showers += ' and subcluster > 4'
NPLCUTQ_all_showers += ' and trkfit < 0.65'
NPLCUTQ_all_showers += ' and tksh_distance < 6.0'
NPLCUTQ_all_showers += ' and (shr_tkfit_nhits_tot > 1 and shr_tkfit_dedx_max > 0.5 and shr_tkfit_dedx_max < 5.5)' 
NPLCUTQ_all_showers += ' and tksh_angle > -0.9'
NPLCUTQ_all_showers += ' and shr_trk_len < 300.' # new cut
NPLCUTQ = NPLCUTQ_all_showers + ' and n_showers_contained == 1'
NPLCUTQ_two_shower = NPLCUTQ_all_showers + ' and n_showers_contained > 1'

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
NPTCUTQ_two_shower = NPTCUTQ_all_showers + ' and n_showers_contained > 1'
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
NPALTESTQ_two_showers = NPALTESTQ_all_showers + ' and n_showers_contained > 1'
#NPALTESTQ = NPALTESTQ_all_showers + ' and (n_showers_contained == 1 or (n_showers_contained>1 and shr12_cos_p1_dstart>0.99))'

# BDT cuts
# 0304 extnumi, pi0 and nonpi0
BDTCQ_all_showers = NPLCUTQ_all_showers
BDTCQ_all_showers += ' and pi0_score > 0.67 and nonpi0_score > 0.70'
BDTCQ = BDTCQ_all_showers + ' and n_showers_contained == 1'
BDTCQ_two_shower = BDTCQ_all_showers + ' and n_showers_contained > 1'

# CT: Adding inverted BDT cuts
BDTCQ_all_showers_INV = NPLCUTQ_all_showers
BDTCQ_all_showers_INV += ' and (pi0_score < 0.67 or nonpi0_score < 0.70)'
BDTCQ_INV = BDTCQ_all_showers_INV + ' and n_showers_contained == 1'

#BDTCQ = BDTCQ_all_showers + ' and (n_showers_contained == 1 or (n_showers_contained>1 and shr12_cos_p1_dstart>0.99))'

BDTCQ_only = 'pi0_score > 0.67 and nonpi0_score > 0.70'

# xsec selection
NPXSLQ_all_showers = NPPRESQ
NPXSLQ_all_showers += ' and CosmicIPAll3D > 10.'
NPXSLQ_all_showers += ' and trkpid<(0.015*trk_len+0.02)'
NPXSLQ_all_showers += ' and hits_ratio > 0.50'
NPXSLQ_all_showers += ' and shrmoliereavg < 9'
NPXSLQ_all_showers += ' and subcluster > 4'
NPXSLQ_all_showers += ' and trkfit < 0.65'
NPXSLQ_all_showers += ' and tksh_distance < 10.0'
NPXSLQ_all_showers += ' and tksh_angle > -0.9'
NPXSLQ_all_showers += ' and shr_trk_len < 300.'
NPXSLQ_all_showers += ' and protonenergy_corr > 0.05'
NPXSLQ = NPXSLQ_all_showers + ' and n_showers_contained == 1'
NPXSBDTQ_all_showers = NPXSLQ_all_showers
NPXSBDTQ_all_showers += ' and pi0_score > 0.50 and nonpi0_score > 0.50'
NPXSBDTQ = NPXSBDTQ_all_showers + ' and n_showers_contained == 1'

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

ZPPRESEL_all_tracks = PRESQ
ZPPRESEL_onep_track = ZPPRESEL_all_tracks + ' and n_tracks_contained > 0'
ZPPRESEL = ZPPRESEL_all_tracks + ' and n_tracks_contained == 0'
ZPPRESEL_one_shower = ZPPRESEL + ' and n_showers_contained == 1'
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
ZPLOOSESEL_all_tracks += ' and (n_tracks_tot == 0 or (n_tracks_tot>0 and tk1sh1_angle_alltk>-0.9))'
ZPLOOSESEL_onep_track = ZPLOOSESEL_all_tracks + ' and n_tracks_contained > 0'
ZPLOOSESEL = ZPLOOSESEL_all_tracks + ' and n_tracks_contained == 0'
ZPBDTLOOSE_all_tracks = ZPLOOSESEL_all_tracks
ZPBDTLOOSE_all_tracks += ' and bkg_score > 0.72'

# CT: Adding new query for inverted BDT cuts
ZPBDTLOOSE_all_tracks_INV = ZPLOOSESEL_all_tracks 
ZPBDTLOOSE_all_tracks_INV  += ' and bkg_score < 0.72'

ZPBDTLOOSE_onep_track = ZPBDTLOOSE_all_tracks + ' and n_tracks_contained > 0'
ZPBDTLOOSE_onep_track_INV = ZPBDTLOOSE_all_tracks_INV + ' and n_tracks_contained > 0'

ZPBDTLOOSE = ZPBDTLOOSE_all_tracks + ' and n_tracks_contained == 0'
ZPBDTLOOSE += ' and (n_tracks_tot == 0 or (n_tracks_tot>0 and tk1sh1_angle_alltk>-0.9))'

ZPBDTLOOSE_INV = ZPBDTLOOSE_all_tracks_INV + ' and n_tracks_contained == 0'
ZPBDTLOOSE_INV += ' and (n_tracks_tot == 0 or (n_tracks_tot>0 and tk1sh1_angle_alltk>-0.9))'

ZPBDTVLOOSE_all_tracks = ZPLOOSESEL_all_tracks
ZPBDTVLOOSE_all_tracks += ' and bkg_score >0.5'
ZPBDTVLOOSE_onep_track = ZPBDTVLOOSE_all_tracks + ' and n_tracks_contained > 0'
ZPBDTVLOOSE = ZPBDTVLOOSE_all_tracks + ' and n_tracks_contained == 0'

ZPBDT_all_tracks = ZPLOOSESEL_all_tracks
ZPBDT_all_tracks += ' and bkg_score >0.85'
ZPBDT_onep_track = ZPBDT_all_tracks + ' and n_tracks_contained > 0'
ZPBDT = ZPBDT_all_tracks + ' and n_tracks_contained == 0'

ZPPRESEL_two_shower = ZPPRESEL + ' and n_showers_contained > 1'
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
ZPLOOSESEL_two_shower += ' and (n_tracks_tot == 0 or (n_tracks_tot>0 and tk1sh1_angle_alltk>-0.9))'
ZPLOOSESEL_two_shower += ' and n_tracks_contained == 0'
ZPBDTLOOSE_two_shower = ZPLOOSESEL_two_shower + ' and bkg_score > 0.72'

ZPPRESEL_two_shower_CRT = ZPPRESEL_two_shower + ' and (crtveto != 1 or crthitpe < 100) and _closestNuCosmicDist > 5.'

ZPXSLQ_all_showers = ZPPRESEL
ZPXSLQ_all_showers += ' and shrmoliereavg < 10'
ZPXSLQ_all_showers += ' and subcluster > 4'
ZPXSLQ_all_showers += ' and trkfit < 0.65'
ZPXSLQ_all_showers += ' and secondshower_Y_nhit < 50'
ZPXSLQ_all_showers += ' and shr_trk_sce_start_y > -100 and shr_trk_sce_start_y < 90'
ZPXSLQ_all_showers += ' and shr_trk_sce_end_y > -100 and shr_trk_sce_end_y < 100 '
ZPXSLQ_all_showers += ' and shr_trk_len < 300.'
ZPXSLQ_all_showers += ' and n_tracks_tot == 0'
ZPXSLQ_all_showers += ' and shr_tkfit_gap10_dedx_max<4'
ZPXSLQ = ZPXSLQ_all_showers + ' and n_showers_contained == 1'
ZPXSBDTQ_all_showers = ZPXSLQ_all_showers
ZPXSBDTQ_all_showers += ' and bkg_score>0.4'
ZPXSBDTQ_all_showers += ' and cos_shr_theta>0.6'
ZPXSBDTQ_all_showers += ' and electron_e>0.51'
ZPXSBDTQ = ZPXSBDTQ_all_showers + ' and n_showers_contained == 1'

XPXSBDTQ = "(("+ZPXSBDTQ+") or ("+NPXSBDTQ+"))"

ZPONEGAMMA = ZPLOOSESEL
ZPONEGAMMA += ' and (CosmicDirAll3D<0.8 and CosmicDirAll3D>-0.8)'
ZPONEGAMMA += ' and (shr_phi<-2.2 or shr_phi>-0.8)'
ZPONEGAMMA += ' and shr_tkfit_dedx_max>3.5'
ZPONEGAMMA += ' and shr_score<0.2'
ZPONEGAMMA += ' and (secondshower_Y_nhit<=8 or secondshower_Y_dot<=0.8 or anglediff_Y<=40 or secondshower_Y_vtxdist>=100)'
ZPONEGAMMA += ' and shr_trk_len < 300.'
ZPONEGAMMA += ' and n_tracks_tot==0'
ZPONEGAMMA += ' and subcluster > 6'

CCNCPI0SEL = 'CosmicIPAll3D > 30.'
#CCNCPI0SEL += ' and CosmicDirAll3D > -0.90 and CosmicDirAll3D < 0.90'
#CCNCPI0SEL += ' and shr_trk_sce_start_y < 100 and shr_trk_sce_end_y < 100 and shr_trk_sce_start_y > -100 and shr_trk_sce_end_y > -100'
CCNCPI0SEL += ' and shr_trk_sce_end_y < 105'
CCNCPI0SEL += ' and hits_y > 80'
CCNCPI0SEL += ' and shr_score < 0.25'
#v1
CCNCPI0SEL += ' and topological_score > 0.1'

CCNCPI0SEL1TRK = CCNCPI0SEL
CCNCPI0SEL1TRK += ' and n_tracks_contained > 0'

CCPI0SEL = CCNCPI0SEL
CCPI0SEL += ' and ((trkpid>0.6 and n_tracks_contained>0) or (n_tracks_contained < n_tracks_tot))'

NCPI0SEL = CCNCPI0SEL
NCPI0SEL += ' and (n_tracks_contained == 0 or trkpid < 0.6)'
NCPI0SEL += ' and (n_tracks_contained == n_tracks_tot)' # no exiting tracks

# SIDEBANDS CUTS
LOW_PID = '(0.0 < pi0_score < 1.0) and (0.0 < nonpi0_score < 1.0) and ~((pi0_score > 0.1) and (nonpi0_score > 0.1))'
MEDIUM_PID = '(0.1 < pi0_score < 1.0) and (0.1 < nonpi0_score < 1.0) and ~((pi0_score > 0.67) and (nonpi0_score > 0.7))'
LOW_ENERGY = '(0.15 < reco_e < 0.65)'
MEDIUM_ENERGY = '(0.75 < reco_e < 1.05)'
LOW_MEDIUM_ENERGY = '(0.05 < reco_e < 1.05)'
HIGH_ENERGY = '(1.05 < reco_e < 2.05)'
HIGH_ENERGY_ZP = '(reco_e > 0.9)'
MEDIUM_ENERGY_ZP = '(0.65 < reco_e < 0.9)'
HIGH_ENERGY_NOUPBOUND = '(reco_e > 1.05)'
NEAR_ENERGY_NOUPBOUND = '(reco_e > 0.65)'
NEAR_ENERGY_ONLY = '(0.65 < reco_e < 0.85)'
HIGH_ENERGY_EXT = '(0.85 < reco_e < 2.05)'
HIGH_ENERGY_EXT_NOUPBOUND = '(reco_e > 0.85)'
ADD_ENERGY_BINS = '(reco_e > 0.85 and reco_e < 1.05)'
ALL_ENERGY = '(reco_e > 0.)'
TWOP_SHOWERS = 'n_showers_contained >= 2'
LOW_PID_ZP = '(0.0 < bkg_score < 0.4)'
MEDIUM_PID_ZP = '(0.4 < bkg_score < 0.72)'
BLIND = '(bnbdata == 0)'

# CT Defining near and far sideband selection queries, these are what's used in the technote
# Everything in the NP selection except the BDT cuts
NP_SIDEBANDS_OTHERCRITERIA = "CosmicIPAll3D > 10. and trkpid < 0.02 and hits_ratio > 0.50 and shrmoliereavg < 9 and subcluster > 4 and trkfit < 0.65 and tksh_distance < 6.0 and (shr_tkfit_nhits_tot > 1 and shr_tkfit_dedx_max > 0.5 and shr_tkfit_dedx_max < 5.5) and tksh_angle > -0.9 and shr_trk_len < 300. and n_showers_contained == 1" 
NP_FAR_SIDEBAND = "reco_e > 1.05 or (pi0_score < 0.1 and nonpi0_score < 0.1)" 
NP_NEAR_SIDEBAND = "(0.75 < reco_e < 1.05 and (pi0_score > 0.1 and nonpi0_score > 0.1)) or (reco_e < 0.75 and (0.1 < pi0_score < 0.67 and 0.1 < nonpi0_score < 0.7))"

# Everything in the NP selection except the BDT cuts
ZP_SIDEBANDS_OTHERCRITERIA="n_showers_contained == 1 and CosmicIPAll3D > 10. and CosmicDirAll3D > -0.9 and CosmicDirAll3D < 0.9 and shrmoliereavg < 15 and subcluster > 4 and trkfit < 0.65 and secondshower_Y_nhit < 50 and shr_trk_sce_start_y > -100 and shr_trk_sce_start_y < 80 and shr_trk_sce_end_y > -100 and shr_trk_sce_end_y < 100  and shr_trk_len < 300. and (n_tracks_tot == 0 or (n_tracks_tot>0 and tk1sh1_angle_alltk>-0.9)) and n_tracks_contained == 0 and (n_tracks_tot == 0 or (n_tracks_tot>0 and tk1sh1_angle_alltk>-0.9))"
ZP_FAR_SIDEBAND = "reco_e > 0.9 or bkg_score < 0.4" 
ZP_NEAR_SIDEBAND = "(0.65 < reco_e < 0.9 and bkg_score > 0.4) or (reco_e < 0.65 and 0.4 < bkg_score < 0.72)" 

NEAR_SIDEBAND = "(" + NP_NEAR_SIDEBAND + ") or (" + ZP_NEAR_SIDEBAND + ")"
FAR_SIDEBAND = "(" + NP_FAR_SIDEBAND + ") or (" + ZP_FAR_SIDEBAND + ")"

#ZP_FAR_SIDEBAND = HIGH_ENERGY_ZP + " or " + LOW_PID_ZP 
#ZP_NEAR_SIDEBAND = MEDIUM_ENERGY_ZP + " or " + MEDIUM_PID_ZP 

# High and medium energy sidebands

NP_HIGH_ENERGY = "(reco_e > 1.05)"
NP_MEDIUM_ENERGY = "(0.75 < reco_e < 1.05 and pi0_score > 0.1 and nonpi0_score > 0.1)"

ZP_HIGH_ENERGY = "(reco_e > 0.90)"
ZP_MEDIUM_ENERGY = "(0.65 < reco_e < 0.90 and bkg_score > 0.4)"

# Low and medium PID sidebands 

NP_LOW_PID = "((0.0 < pi0_score < 1.0) and (0.0 < nonpi0_score < 1.0) and ~((pi0_score > 0.1) and (nonpi0_score > 0.1)))"
NP_MEDIUM_PID = "(0.1 < pi0_score < 0.67 and 0.1 < nonpi0_score < 0.7 and reco_e < 1.05)"

ZP_LOW_PID = "(0.0 < bkg_score < 0.4)"
ZP_MEDIUM_PID = "(0.4 < bkg_score < 0.72 and reco_e < 0.90)" 

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
NUMUPRESEL += ' and (reco_nu_vtx_sce_z < 675 or reco_nu_vtx_sce_z > 775) '
NUMUPRESEL += ' and topological_score > 0.06 '
#NUMUPRESEL += ' and contained_fraction > 0.9 '

NUMUCRT = ' and (crtveto != 1 or crthitpe < 100) and _closestNuCosmicDist > 5.'

NUMUPRESELCRT = NUMUPRESEL + NUMUCRT

NUMUSEL = NUMUPRESEL + ' and n_muons_tot > 0'
NUMUSEL0PI = NUMUSEL + ' and n_muons_tot == 1 and n_showers_tot == 0'

NUMUSEL0PI = NUMUSEL + ' and n_muons_tot == 1 and n_showers_tot == 0'

NUMUSELNP = NUMUSEL + ' and n_protons_tot > 0'
NUMUSEL0P = NUMUSEL + ' and n_protons_tot == 0'
NUMUSELNP0PI = NUMUSEL0PI + ' and n_protons_tot > 0'
NUMUSEL0P0PI = NUMUSEL0PI + ' and n_protons_tot == 0'

NUMUSELNP0PI = NUMUSEL0PI + ' and n_protons_tot > 0'
NUMUSEL0P0PI = NUMUSEL0PI + ' and n_protons_tot == 0'

NUMUSELCRT = NUMUSEL + NUMUCRT
NUMUSELCRT0PI = NUMUSEL0PI + NUMUCRT

NUMUSELCRT0PI = NUMUSEL0PI + NUMUCRT

NUMUSELCRTNP = NUMUSELCRT + ' and n_protons_tot > 0'
NUMUSELCRT0P = NUMUSELCRT + ' and n_protons_tot == 0'
NUMUSELCRTNP0PI = NUMUSELCRT0PI + ' and n_protons_tot > 0'
NUMUSELCRT0P0PI = NUMUSELCRT0PI + ' and n_protons_tot == 0 and topological_score > 0.2'

NUMUSELCRTNP0PI = NUMUSELCRT0PI + ' and n_protons_tot > 0'
NUMUSELCRT0P0PI = NUMUSELCRT0PI + ' and n_protons_tot == 0 and topological_score > 0.2'

NUMUSEL1MU1P = NUMUSEL + ' and n_tracks_contained == 2 and trk2_pid < -0.2'

# eta queries
ETASLICE = ' nslice == 1'
ETASLICE += ' and topological_score > 0.1'
ETASLICE += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'

ETACONTAINMENT  = ETASLICE + ' and ' + 'reco_nu_vtx_sce_x > 10 and reco_nu_vtx_sce_x < 250'
ETACONTAINMENT += ' and reco_nu_vtx_sce_y > -110 and reco_nu_vtx_sce_y < 100'
ETACONTAINMENT += ' and reco_nu_vtx_sce_z > 25 and reco_nu_vtx_sce_z < 990'

ETATWOSHRQUERY  = ETACONTAINMENT + ' and ' + 'n_showers_050_tot == 2'

ETAPI0QUERY = ETATWOSHRQUERY + ' and ' + 'pi0_dot1 > 0.9 and pi0_dot2 > 0.9'
ETAPI0QUERY += ' and pi0_radlen1 > 2 and pi0_radlen2 > 2'

ETAQUERY = ETAPI0QUERY + ' and ' + 'pi0_mass_Y_corr > 250. and pi0_mass_Y_corr < 750.'

# sideband categories
sideband_categories = {
    'HiEextmax2': {'query': HIGH_ENERGY_EXT, 'title': '0.85 GeV < Reco energy < 2.05 GeV', 'dir': 'HiEextmax2'},
    'HiEmax2': {'query': HIGH_ENERGY, 'title': '1.05 GeV < Reco energy < 2.05 GeV', 'dir': 'HiEmax2'},
    'HiE': {'query': HIGH_ENERGY_NOUPBOUND, 'title': 'Reco energy > 1.05 GeV', 'dir': 'HiE'},
    'NearE': {'query': NEAR_ENERGY_NOUPBOUND, 'title': 'Reco energy > 0.65 GeV', 'dir': 'HiEnear'},
    'NearEOnly': {'query': NEAR_ENERGY_ONLY, 'title': '0.65 GeV < Reco energy < 0.85 GeV', 'dir': 'HiEnearOnly'},
    'LowE': {'query': LOW_ENERGY, 'title': '0.15 < Reco energy < 0.65 GeV', 'dir': 'LowE'},
    'HiEext': {'query': HIGH_ENERGY_EXT_NOUPBOUND, 'title': 'Reco energy > 0.85 GeV', 'dir': 'HiEext'},
    'HiEadd': {'query': ADD_ENERGY_BINS, 'title': '0.85 < Reco energy < 1.05 GeV', 'dir': 'HiEadd'},
    'HiEZP': {'query': HIGH_ENERGY_ZP, 'title': 'Reco energy > 0.9 GeV', 'dir': 'HiEZP'},
    'LPID': {'query': LOW_PID, 'title': 'Low BDT', 'dir': 'LPID'},
    'MPID': {'query': MEDIUM_PID, 'title': 'Medium BDT', 'dir': 'MPID'},
    'LPIDZP': {'query': LOW_PID_ZP, 'title': 'Low BDT', 'dir': 'LPIDZP'},
    'MPIDZP': {'query': MEDIUM_PID_ZP, 'title': 'Medium BDT', 'dir': 'MPIDZP'},
    'TwoPShr': {'query': TWOP_SHOWERS, 'title': '2+ showers', 'dir': 'TwoPShr'},
    'TwoPShrHiE': {'query': " and ".join([TWOP_SHOWERS,HIGH_ENERGY_NOUPBOUND]), 'title': '2+ showers,Reco energy > 1.05 GeV', 'dir': 'TwoPShrHiE'},
    'Blind': {'query': BLIND, 'title': 'Blind', 'dir': 'Blind'},
    'None': {'query': None, 'title': None, 'dir': 'None'},
}

# preselection categories
preselection_categories = {
    'PI0': {'query': PREPI0Q, 'title': 'Pi0 Presel.', 'dir': 'PI0'},
    'NUE': {'query': PRESQ, 'title': 'Nue Presel.', 'dir': 'NUE'},
    'NP': {'query': NPPRESQ, 'title': '1eNp Presel.', 'dir': 'NP'},
    'NPOneShr': {'query': NPPRESQ_one_shower, 'title': '1eNp Presel., 1 shower', 'dir': 'NPOneShr'},
    'NPOneTrk': {'query': NPPRESQ_one_track, 'title': '1eNp Presel., 1 track', 'dir': 'NPOneTrk'},
    'NPTwoPTrk': {'query': NPPRESQ_twoplus_tracks, 'title': '1eNp Presel., 2+ tracks', 'dir': 'NPTwoPTrk'},
    'ZP': {'query': ZPPRESEL, 'title': '1e0p Presel.', 'dir': 'ZP'},
    'ZPOneShr': {'query': ZPPRESEL_one_shower, 'title': '1e0p Presel., 1 shower', 'dir': 'ZPOneShr'},
    'ZPAllTrks': {'query': ZPPRESEL_all_tracks, 'title': '1e0p Presel., 0+ tracks', 'dir': 'ZPAllTrks'},
    'ZPTwoShr': {'query': ZPPRESEL_two_shower, 'title': '1e0p Presel., 2+ shower', 'dir': 'ZPTwoShr'},
    'ZPTwoShrCRT': {'query': ZPPRESEL_two_shower_CRT, 'title': '1e0p Presel. w/ CRT, 2+ shower', 'dir': 'ZPTwoShrCRT'},
    'None': {'query': None, 'title': None, 'dir': 'None'},
    'NSLICE': {'query': 'nslice==1', 'title': r"SliceID selection", 'dir': 'NSLICE'},
    'NUMU': {'query': NUMUPRESEL, 'title': r"$\nu_{\mu}$ selection", 'dir': 'NUMU'},
    'NUMUCRT': {'query': NUMUPRESELCRT, 'title': r"$\nu_{\mu}$ pre-selection w/ CRT", 'dir': 'NUMUCRT'},
    'OneP': {'query': OnePPRESQ, 'title': '1e1p Presel.', 'dir': 'OneP'}

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
    'ZPBDTTIGHT': {'query': ZPBDT, 'title': '1e0p BDT Tighter sel.', 'dir': 'ZPBDTTIGHT'},
    'ZPT': {'query': ZPBOXCUTS, 'title': '1e0p Tight Cuts sel.', 'dir': 'ZPT'},
    'ZPBDTAllTrk': {'query': ZPBDTLOOSE_all_tracks, 'title': '1e0p BDT sel.', 'dir': 'ZPBDTAllTrk'},
    'ZPLOOSESEL': {'query': ZPLOOSESEL, 'title': '1e0p Loose sel.', 'dir': 'ZPLOOSESEL'},
    'ZPONEGAMMA': {'query': ZPONEGAMMA, 'title': '1g1p sel.', 'dir': 'ZPONEGAMMA'},
    'ZPLAllTrk': {'query': ZPLOOSESEL_all_tracks, 'title': '1e0p Loose sel.', 'dir': 'ZPLAllTrk'},
    'NUMUPRE': {'query': NUMUPRESEL, 'title': r"$\nu_{\mu}$ pre-selection", 'dir': 'NUMU'},
    'NUMU': {'query': NUMUSEL, 'title': r"$\nu_{\mu}$ selection", 'dir': 'NUMU'},
    'NUMUCRT': {'query': NUMUSELCRT, 'title': r"$\nu_{\mu}$ selection w/ CRT", 'dir': 'NUMUCRT'},
    'NUMUCRT0PI': {'query': NUMUSELCRT0PI, 'title': r"$\nu_{\mu}$0$\pi$ selection w/ CRT", 'dir': 'NUMUCRT0PI'},
    'NUMU1MU1P': {'query': NUMUSEL1MU1P, 'title': r"$\nu_{\mu}$ 1$\mu$1$p$ selection", 'dir': 'NUMU1MU1P'},
    'PI0SEL': {'query': PI0SEL,'title': r"$\pi^0$ selection",'dir':"PI0"},
    'CCNCPI0': {'query': CCNCPI0SEL, 'title': r"CC/NC pi0 selection", 'dir': 'CCNCPI0'},
    'CCPI0': {'query': CCPI0SEL, 'title': r"CC pi0 selection", 'dir': 'CCPI0'},
    'NCPI0': {'query': NCPI0SEL, 'title': r"NC pi0 selection", 'dir': 'NCPI0'},
    'PI0': {'query': PI0SEL, 'title': r"$\pi^0$ selection", 'dir': 'PI0'},
    'NUMUCRTNP': {'query': NUMUSELCRTNP, 'title': r"$1\mu$Np selection w/ CRT", 'dir': 'NUMUCRTNP'},
    'NUMUCRT0P': {'query': NUMUSELCRT0P, 'title': r"$1\mu$0p selection w/ CRT", 'dir': 'NUMUCRT0P'},
    'NUMUCRTNP0PI': {'query': NUMUSELCRTNP0PI, 'title': r"$1\mu$Np0$\pi$ selection w/ CRT", 'dir': 'NUMUCRTNP0PI'},
    'NUMUCRT0P0PI': {'query': NUMUSELCRT0P0PI, 'title': r"$1\mu$0p0$\pi$ selection w/ CRT", 'dir': 'NUMUCRT0P0PI'},
    'NUMUNP': {'query': NUMUSELNP, 'title': r"$1\mu$Np selection", 'dir': 'NUMUNP'},
    'NUMU0P': {'query': NUMUSEL0P, 'title': r"$1\mu$0p selection", 'dir': 'NUMU0P'},
    'ETATWOSHR': {'query': ETATWOSHRQUERY, 'title': r"$\eta$ selection - two-shower cuts",'dir': 'ETATWOSHR' },
    'ETAPI0': {'query': ETAPI0QUERY, 'title': r"$\eta$ selection - $\pi^0$ cuts",'dir': 'ETAPI0' },
    'ETA' : {'query': ETAQUERY, 'title': r"$\eta$ selection",'dir': 'ETA' },
    'NPXSL': {'query': NPXSLQ, 'title': '1eNp xsec Loose cuts', 'dir': 'NPXSL'},
    'NPXSLAllShr': {'query': NPXSLQ_all_showers, 'title': '1eNp xsec Loose cuts, 0+ showers', 'dir': 'NPXSLAllShr'},
    'NPXSBDT': {'query': NPXSBDTQ, 'title': '1eNp xsec BDT sel.', 'dir': 'NPXSBDT'},
    'NPXSBDTAllShr': {'query': NPXSBDTQ_all_showers, 'title': '1eNp xsec BDT sel., 0+ showers', 'dir': 'NPXSBDTAllShr'},
    'ZPXSL': {'query': ZPXSLQ, 'title': '1e0p xsec Loose cuts', 'dir': 'ZPXSL'},
    'ZPXSLAllShr': {'query': ZPXSLQ_all_showers, 'title': '1e0p xsec Loose cuts, 0+ showers', 'dir': 'ZPXSLAllShr'},
    'ZPXSBDT': {'query': ZPXSBDTQ, 'title': '1e0p xsec BDT sel.', 'dir': 'ZPXSBDT'},
    'ZPXSBDTAllShr': {'query': ZPXSBDTQ_all_showers, 'title': '1e0p xsec BDT sel., 0+ showers', 'dir': 'ZPXSBDTAllShr'},
    'XPXSBDT': {'query': XPXSBDTQ, 'title': '1eXp xsec BDT sel.', 'dir': 'XPXSBDT'},
    'OnePL': {'query': OnePLCUTQ, 'title': '1e1p Loose cuts', 'dir': 'OnePL'},

    # CT: Full selections with BDT cuts inverted
    'ZPBDT_INV': {'query': ZPBDTLOOSE_INV, 'title': 'Inverted 1e0p BDT sel.', 'dir': 'ZPBDT_INV'},
    'NPBDT_INV': {'query': BDTCQ_INV, 'title': 'Inverted 1eNp BDT sel.', 'dir': 'NPBDT_INV'},

    # Selections with only the BDT score cuts inverted but no other criteria
    'BDT_SIDEBAND': {'query': "(pi0_score < 0.67 or nonpi0_score < 0.70) and bkg_score < 0.72", 'title': '1e0p Sideband', 'dir': 'BDT_SIDEBAND'},

    # Additional sideband definitions
    'SHR_ENERGY_SIDEBAND': {'query': "shr_energy_tot_cali > 0.75", 'title': 'Shower Energy Sideband', 'dir': 'SHR_ENERGY_SIDEBAND'},
    'MED_SHR_ENERGY_SIDEBAND': {'query': "shr_energy_tot_cali > 0.65", 'title': 'Medium Shower Energy Sideband', 'dir': 'MED_SHR_ENERGY_SIDEBAND'},
    'TWO_SHR_SIDEBAND': {'query': "n_showers_contained >= 2", 'title': '1e0p Sideband', 'dir': 'TWO_SHR_SIDEBAND'},
    'NUMU_SIDEBAND': {'query': "n_muons_tot > 0", 'title': 'NuMu Sideband', 'dir': 'NUMU_SIDEBAND'},
    'NP_NEAR_SIDEBAND': {'query': NP_NEAR_SIDEBAND, 'title': '1eNp Near Sideband', 'dir': 'NP_NEAR_SIDEBAND'},
    'ZP_NEAR_SIDEBAND': {'query': ZP_NEAR_SIDEBAND, 'title': '1e0p Near Sideaband', 'dir': 'ZP_NEAR_SIDEBAND'},
    'NP_FAR_SIDEBAND': {'query': NP_FAR_SIDEBAND, 'title': '1eNp Far Sideband', 'dir': 'NP_FAR_SIDEBAND'},
    'ZP_FAR_SIDEBAND': {'query': ZP_FAR_SIDEBAND, 'title': '1e0p Far Sideband', 'dir': 'ZP_FAR_SIDEBAND'},
    'NEAR_SIDEBAND': {'query': NEAR_SIDEBAND, 'title': 'Near Sideband', 'dir': 'NEAR_SIDEBAND'},
    'FAR_SIDEBAND': {'query': FAR_SIDEBAND, 'title': 'Far Sideband', 'dir': 'FAR_SIDEBAND'},

    # NuMu TKI Selections
    'SIGNAL_1MU1P': {'query': "Signal_1mu1p == True", 'title': 'True 1mu1p Events', 'dir': 'SIGNAL_1MU1P'},
    'SIGNAL_1MUNP': {'query': "Signal_1muNp == True", 'title': 'True 1muNp Events', 'dir': 'SIGNAL_1MUNP'},
    'SG_1MUNP': {'query': "sel_CCNp0pi == True", 'title': 'Selected 1muNp0pi Events', 'dir': 'SG_1MUNP'},
    'SG_1MU1P': {'query': "sel_CC1p0pi == True", 'title': 'Selected 1mu1p0pi Events', 'dir': 'SG_1MU1P'},

    # Two Shower Selections
    'ZPLOOSESELTWOSHR': {'query': ZPLOOSESEL_two_shower, 'title': '1e0p loose sel. 2+ shr', 'dir': 'ZPLOOSE_two_shower'},
    'ZPBDTTWOSHR': {'query': ZPBDTLOOSE_two_shower, 'title': '1e0p BDT sel. 2+shr', 'dir': 'ZPBDT_two_shower'},
    'NPVLTWOSHR': {'query': NPVLCUTQ_two_shower, 'title': '1eNp VL cuts 2+shr', 'dir': 'NPVL_two_shower'},
    'NPLTWOSHR': {'query': NPLCUTQ_two_shower, 'title': '1eNp Loose cuts 2+shr', 'dir': 'NPL_two_shower'},
    'NPTTWOSHR': {'query': NPTCUTQ_two_shower, 'title': '1eNp Tight cuts 2+shr', 'dir': 'NPT_two_shower'},
    'NPBDTTWOSHR': {'query': BDTCQ_two_shower, 'title': '1eNp BDT sel. 2+shr', 'dir': 'NPBDT_two_shower'},

    # High Energy Sidebands
    'NP_HIGH_ENERGY': {'query': NP_HIGH_ENERGY , 'title': '1eNp VL cuts, High Energy', 'dir': 'NP_HIGH_ENERGY'},
    'NPVL_HIGH_ENERGY': {'query': NPVLCUTQ+" and "+NP_HIGH_ENERGY , 'title': '1eNp VL cuts, High Energy', 'dir': 'NPVL_HIGH_ENERGY'},
    'NPL_HIGH_ENERGY': {'query': NPLCUTQ+" and "+NP_HIGH_ENERGY , 'title': '1eNp Loose cuts, High Energy', 'dir': 'NPL_HIGH_ENERGY'},
    'NPT_HIGH_ENERGY': {'query': NPTCUTQ+" and "+NP_HIGH_ENERGY , 'title': '1eNp Tight cuts, High Energy', 'dir': 'NPT_HIGH_ENERGY'},
    'NPBDT_HIGH_ENERGY': {'query': BDTCQ+" and "+NP_HIGH_ENERGY , 'title': '1eNp BDT sel., High Energy', 'dir': 'NPBDT_HIGH_ENERGY'},
    'ZP_HIGH_ENERGY': {'query': ZP_HIGH_ENERGY , 'title': '1eNp VL cuts, High Energy', 'dir': 'ZP_HIGH_ENERGY'},
    'ZPLOOSESEL_HIGH_ENERGY': {'query': ZPLOOSESEL+" and "+ZP_HIGH_ENERGY , 'title': '1e0p Loose sel., High Energy', 'dir': 'ZPLOOSESEL_HIGH_ENERGY'},
    'ZPBDT_HIGH_ENERGY': {'query': ZPBDTLOOSE+" and "+ZP_HIGH_ENERGY , 'title': '1e0p BDT sel., High Energy', 'dir': 'ZPBDT_HIGH_ENERGY'},

    # Medium Energy Sidebands
    'NP_MEDIUM_ENERGY': {'query': NP_MEDIUM_ENERGY , 'title': '1eNp VL cuts, Medium Energy', 'dir': 'NP_MEDIUM_ENERGY'},
    'NPVL_MEDIUM_ENERGY': {'query': NPVLCUTQ+" and "+NP_MEDIUM_ENERGY , 'title': '1eNp VL cuts, Medium Energy', 'dir': 'NPVL_MEDIUM_ENERGY'},
    'NPL_MEDIUM_ENERGY': {'query': NPLCUTQ+" and "+NP_MEDIUM_ENERGY , 'title': '1eNp Loose cuts, Medium Energy', 'dir': 'NPL_MEDIUM_ENERGY'},
    'NPT_MEDIUM_ENERGY': {'query': NPTCUTQ+" and "+NP_MEDIUM_ENERGY , 'title': '1eNp Tight cuts, Medium Energy', 'dir': 'NPT_MEDIUM_ENERGY'},
    'NPBDT_MEDIUM_ENERGY': {'query': BDTCQ+" and "+NP_MEDIUM_ENERGY , 'title': '1eNp BDT sel., Medium Energy', 'dir': 'NPBDT_MEDIUM_ENERGY'},
    'ZP_MEDIUM_ENERGY': {'query': ZP_MEDIUM_ENERGY , 'title': '1eNp VL cuts, Medium Energy', 'dir': 'ZP_MEDIUM_ENERGY'},
    'ZPLOOSESEL_MEDIUM_ENERGY': {'query': ZPLOOSESEL+" and "+ZP_MEDIUM_ENERGY , 'title': '1e0p Loose sel., Medium Energy', 'dir': 'ZPLOOSESEL_MEDIUM_ENERGY'},
    'ZPBDT_MEDIUM_ENERGY': {'query': ZPBDTLOOSE+" and "+ZP_MEDIUM_ENERGY , 'title': '1e0p BDT sel., Medium Energy', 'dir': 'ZPBDT_MEDIUM_ENERGY'},

    # Low PID Sidebands 
    'NP_LOW_PID': {'query': NP_LOW_PID , 'title': '1eNp Presel., Low PID', 'dir': 'NP_LOW_PID'},
    'NPVL_LOW_PID': {'query': NPVLCUTQ+" and "+NP_LOW_PID , 'title': '1eNp VL cuts, Low PID', 'dir': 'NPVL_LOW_PID'},
    'NPL_LOW_PID': {'query': NPLCUTQ+" and "+NP_LOW_PID , 'title': '1eNp Loose cuts, Low PID', 'dir': 'NPL_LOW_PID'},
    'NPT_LOW_PID': {'query': NPTCUTQ+" and "+NP_LOW_PID , 'title': '1eNp Tight cuts, Low PID', 'dir': 'NPT_LOW_PID'},
    'ZP_LOW_PID': {'query': ZP_LOW_PID , 'title': '1e0p Presel., Low PID', 'dir': 'ZP_LOW_PID'},
    'ZPLOOSESEL_LOW_PID': {'query': ZPLOOSESEL+" and "+ZP_LOW_PID , 'title': '1e0p Loose sel., Low PID', 'dir': 'ZPLOOSESEL_LOW_PID'},

    # Low PID Sidebands 
    'NP_MEDIUM_PID': {'query': NP_MEDIUM_PID , 'title': '1eNp Presel., Medium PID', 'dir': 'NP_MEDIUM_PID'},
    'NPVL_MEDIUM_PID': {'query': NPVLCUTQ+" and "+NP_MEDIUM_PID , 'title': '1eNp VL cuts, Medium PID', 'dir': 'NPVL_MEDIUM_PID'},
    'NPL_MEDIUM_PID': {'query': NPLCUTQ+" and "+NP_MEDIUM_PID , 'title': '1eNp Loose cuts, Medium PID', 'dir': 'NPL_MEDIUM_PID'},
    'NPT_MEDIUM_PID': {'query': NPTCUTQ+" and "+NP_MEDIUM_PID , 'title': '1eNp Tight cuts, Medium PID', 'dir': 'NPT_MEDIUM_PID'},
    'ZP_MEDIUM_PID': {'query': ZP_MEDIUM_PID , 'title': '1e0p Presel., Medium PID', 'dir': 'ZP_MEDIUM_PID'},
    'ZPLOOSESEL_MEDIUM_PID': {'query': ZPLOOSESEL+" and "+ZP_MEDIUM_PID , 'title': '1e0p Loose sel., Medium PID', 'dir': 'ZPLOOSESEL_MEDIUM_PID'},

    # Giuseppe's selections
    'ZPTwoShrCRT': {'query': ZPPRESEL_two_shower_CRT, 'title': '1e0p Presel. w/ CRT, 2+ shower', 'dir': 'ZPTwoShrCRT'},
    'NUMUCRT0PI': {'query': NUMUSELCRT0PI, 'title': r"$\nu_{\mu}$0$\pi$ selection w/ CRT", 'dir': 'NUMUCRT0PI'},
    'NUMUCRTNP0PI': {'query': NUMUSELCRTNP0PI, 'title': r"$1\mu$Np0$\pi$ selection w/ CRT", 'dir': 'NUMUCRTNP0PI'},
    'NUMUCRT0P0PI': {'query': NUMUSELCRT0P0PI, 'title': r"$1\mu$0p0$\pi$ selection w/ CRT", 'dir': 'NUMUCRT0P0PI'},
    'NUMU0PI': {'query': NUMUSEL0PI, 'title': r"$\nu_{\mu}$0$\pi$ selection", 'dir': 'NUMU0PI'},
    'NUMUNP0PI': {'query': NUMUSELNP0PI, 'title': r"$1\mu$Np0$\pi$ selection", 'dir': 'NUMUNP0PI'},
    'NUMU0P0PI': {'query': NUMUSEL0P0PI, 'title': r"$1\mu$0p0$\pi$ selection", 'dir': 'NUMU0P0PI'},

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

def extract_variables_from_query(query):
    # Find all instances of words which are directly followed by either a comparison operator or a bracket.
    # This is done to exclude functions or methods acting on the DataFrame columns.
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?=[\s]*[!=<>]+|\))'
    variables = re.findall(pattern, query)
    # Use a set to remove duplicates
    unique_variables = set(variables)
    return unique_variables

def get_required_variables(preselections=None, selections=None):
    required_variables = set()
    if preselections is not None:
        for selection in preselections:
            required_variables.update(extract_variables_from_query(preselection_categories[selection]["query"]))
    if selections is not None:
        for selection in selections:
            required_variables.update(extract_variables_from_query(selection_categories[selection]["query"]))
    return required_variables


def get_selection_query(selection, preselection, extra_queries=None):
    """Get the query for the given selection and preselection.

    Optionally, add any extra queries to the selection query. These will
    be joined with an 'and' operator.

    Parameters
    ----------
    selection : str
        Name of the selection category.
    preselection : str
        Name of the preselection category.
    extra_queries : list of str, optional
        List of additional queries to apply to the dataframe.

    Returns
    -------
    query : str
        Query to apply to the dataframe.
    """

    if selection is None and preselection is None:
        if extra_queries is not None:
            return " and ".join(extra_queries)
        return None
    presel_query = preselection_categories[preselection]["query"]
    sel_query = selection_categories[selection]["query"]

    if presel_query is None:
        query = sel_query
    elif sel_query is None:
        query = presel_query
    else:
        query = f"{presel_query} and {sel_query}"

    if extra_queries is not None:
        for q in extra_queries:
            query = f"{query} and {q}"
    return query

def _shorten_title(title):
    """Heuristically shorten the title of a selection.
    
    This tries to remove redundant words like "selection" or "sel.".
    """

    # Remove "selection" from the title
    title = title.replace("selection", "")
    # Remove "sel." from the title
    title = title.replace("sel.", "")
    # Remove double whitespaces
    title = re.sub(r"\s+", " ", title)
    return title.strip()

def get_selection_title(selection, preselection, with_presel=False, short=False):
    """Get the title for the given selection and preselection.

    Parameters
    ----------
    selection : str
        Name of the selection category.
    preselection : str
        Name of the preselection category.
    with_presel : bool, optional
        Whether to include the preselection title in the selection title.
    short : bool, optional
        Whether to use the short title. If a short title is not defined for a selection,
        the function will try to shorten the title heuristically.

    Returns
    -------
    title : str
        Title of the selection.
    """
    if selection is None and preselection is None:
        return None
    if short and "short_title" in selection_categories[selection]:
        sel_title = selection_categories[selection]["short_title"]
    else:
        sel_title = selection_categories[selection]["title"]
        if short:
            sel_title = _shorten_title(sel_title)
    if short and "short_title" in preselection_categories[preselection]:
        presel_title = preselection_categories[preselection]["short_title"]
    else:
        presel_title = preselection_categories[preselection]["title"]
        if short:
            presel_title = _shorten_title(presel_title)

    if presel_title is None:
        title = sel_title
    elif sel_title is None:
        title = presel_title
    elif with_presel:
        title = f"{sel_title} ({presel_title})"
    else:
        title = sel_title

    return title

def _find_parentheses_groups(s):
    stack = []
    result = []
    open_count = 0
    for i, c in enumerate(s):
        if c == '(':
            if open_count == 0:
                stack.append(i)
            open_count += 1
        elif c == ')' and stack:
            open_count -= 1
            if open_count == 0:
                start = stack.pop()
                result.append(s[start:i+1])
    return result

def _replace_parentheses_groups(s):
    matches = _find_parentheses_groups(s)
    replacements = {}
    for match in matches:
        group_id = f'group_{hash(match.replace(" ", "").strip())}'
        s = s.replace(match, group_id)
        replacements[group_id] = match
    return s, replacements

def _find_common_selection(s1, s2):
    # Replace parentheses groups with unique identifiers
    s1, replacements1 = _replace_parentheses_groups(s1)
    s2, replacements2 = _replace_parentheses_groups(s2)
    
    # Split the strings into sets of conditions
    conditions1 = set(s1.split(' and '))
    conditions2 = set(s2.split(' and '))
    
    # Find the intersection and differences of the sets
    common_conditions = conditions1 & conditions2
    unique_conditions1 = conditions1 - conditions2
    unique_conditions2 = conditions2 - conditions1
    
    # Sort the conditions and join them back into strings
    common_selection = ' and '.join(sorted(common_conditions))
    unique_selection1 = ' and '.join(sorted(unique_conditions1))
    unique_selection2 = ' and '.join(sorted(unique_conditions2))
    
    # Replace the identifiers with their corresponding parentheses groups
    for group_id, group in replacements1.items():
        common_selection = common_selection.replace(group_id, group)
        unique_selection1 = unique_selection1.replace(group_id, group)
    for group_id, group in replacements2.items():
        unique_selection2 = unique_selection2.replace(group_id, group)
    
    return common_selection, unique_selection1, unique_selection2

def find_common_selection(strings: List[str]) -> Tuple[str, List[str]]:
    """Find the common selection between a list of selection strings."""
    if len(strings) == 0:
        return "", []
    elif len(strings) == 1:
        return strings[0], [""]
    common = strings[0]
    unique = [""] * len(strings)
    for i in range(1, len(strings)):
        common, *_ = _find_common_selection(common, strings[i])
    for i in range(len(strings)):
        _, _, unique[i] = _find_common_selection(common, strings[i])

    return common, unique
