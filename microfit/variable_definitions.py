basic_variables = [
    ("n_showers_contained", 1, (0.5, 1.5), "normalization", "onebin"),
    # ('n_showers_contained',10,(-0.5, 9.5),"n showers contained"),
    ("n_tracks_contained", 6, (-0.5, 5.5), "n tracks contained"),
    # ('n_tracks_tot',6,(-0.5, 5.5),"n tracks total"),
    # ('reco_e',21,(0.05,2.15),r"Reconstructed Energy [GeV]"),
    # ('reco_e',20,(0.05,3.05),r"Reconstructed Energy [GeV]","extended"),
    # ('reco_e',7,(0.05,2.85),r"Reconstructed Energy [GeV]","coarse"),
    # ('reco_e',22,(-0.05,2.15),r"Reconstructed Energy [GeV]"),
    # ('reco_e',21,(-0.05,4.15),r"Reconstructed Energy [GeV]","extended"),
    ("reco_e", 20, (0.15, 2.95), r"Reconstructed Energy [GeV]", "note"),
    # ('reco_e',10,(0.9,3.9),r"Reconstructed Energy [GeV]","highe"),
]

variables_1e1p = [
    # ('reco_e',21,(0.05,2.15),r"Reconstructed Energy [GeV]"),
    # ('reco_e',20,(0.05,3.05),r"Reconstructed Energy [GeV]","extended"),
    # ('reco_e',7,(0.05,2.85),r"Reconstructed Energy [GeV]","coarse"),
    # ('reco_e',22,(-0.05,2.15),r"Reconstructed Energy [GeV]"),
    # ('reco_e',21,(-0.05,4.15),r"Reconstructed Energy [GeV]","extended"),
    ("reco_e", 20, (0.15, 2.95), "Reconstructed Neutrino Energy [GeV] \n (reco_e)", "note"),
    # ('reco_e',10,(0.9,3.9),r"Reconstructed Energy [GeV]","highe"),
    ("p", 20, (0, 4), "Total Reconstructed Momentum [GeV/c] \n (p)"),
    ("pt", 20, (0, 2), "Total Reconstructed Transverse Momentum [GeV/c] \n (pt)"),
    ("trk_energy", 10, (0, 1), "Reconstructed Proton Kinetic Energy [GeV] \n (trk_energy)"),
    ("shr_energy_cali", 16, (0, 4), "Reconstructed Electron Energy [GeV] \n (shr_energy_cali)"),
    ("mod_shr_p", 20, (0, 5), "Modulus of the Reconstructed Electron Momentum [GeV/c] \n (mod_shr_p)"),
    ("shr_px", 12, (-1.5, 1.5), "x component of Reconstructed Electron Momentum [GeV/c] \n (shr_px)"),
    ("shr_py", 12, (-1.5, 1.5), "y component of Reconstructed Electron Momentum [GeV/c] \n (shr_py)"),
    ("shr_pz", 12, (-1, 5), "z component of Reconstructed Electron Momentum [GeV/c] \n (shr_pz)"),
    ("mod_trk_p", 20, (0, 1.5), "Modulus of the Reconstructed Proton Momentum [GeV/c] \n (mod_trk_p)"),
    ("trk_px", 20, (-1, 1), "x component of Reconstructed Proton Momentum [GeV/c] \n (trk_px)"),
    ("trk_py", 20, (-1.5, 1.5), "y component of Reconstructed Proton Momentum [GeV/c] \n (trk_py)"),
    ("trk_pz", 20, (-1, 1.5), "z component of Reconstructed Proton Momentum [GeV/c] \n (trk_pz)"),
]

TKI_variables_1e1p = [
    ("mod_delta_pt", 20, (0, 2), "$\\delta p_T$ [GeV/c] \n (mod_delta_pt)"),
    ("delta_alpha", 20, (0, 180), "$\\delta \\alpha_T$ [degrees] \n (delta_alpha)"),
]

loosesel_variables_1eNp = [
    ("hits_ratio", 20, (0, 1), "shower hits/all hits"),
    ("trkfit", 20, (0, 1.0), "Fraction of Track-fitted points"),
    ("subcluster", 20, (0, 40), "N sub-clusters in shower"),
    ("CosmicIPAll3D", 20, (0, 200), "CosmicIPAll3D [cm]"),
    ("shr_tkfit_dedx_max", 20, (0, 10), "shr tkfit dE/dx (max, 0-4 cm) [MeV/cm]"),
    ("tksh_angle", 20, (-1, 1), "cos(trk-shr angle)"),
    ("tksh_distance", 20, (0, 40), "trk-shr distance [cm]"),
    ("shr_tkfit_nhits_tot", 20, (0, 20), "shr tkfit nhits (tot, 0-4 cm) [MeV/cm]"),
    ("trkpid", 21, (-1, 1), "track LLR PID"),
    ("shrmoliereavg", 20, (0, 50), "average Moliere angle [degrees]"),
    ("shr_trk_len", 40, (0, 400), "Shower track fit length [cm]"),
    ("nonpi0_score", 10, (0, 1.0), "BDT non-$\pi^0$ score", "log", True),
    ("pi0_score", 10, (0, 1.0), "BDT $\pi^0$ score", "log", True),
    ("bkg_score", 10, (0, 1.0), "1e0p BDT score", "log", True),
]

evtsel_variabls = [
    ("hits_ratio", 20, (0, 1), "shower hits/all hits"),
    ("CosmicIPAll3D", 20, (0, 200), "CosmicIPAll3D [cm]"),
    ("CosmicDirAll3D", 20, (-1, 1), "cos(CosmicDirAll3D)"),
    ("tk1sh1_angle_alltk", 20, (-1, 1), "cos(tk1sh1)"),
]
shrsel_variables = [
    ("trkfit", 10, (0, 1.0), "Fraction of Track-fitted points"),
    ("shrmoliereavg", 20, (0, 50), "average Moliere angle [degrees]"),
    # ('shrmoliereavg',10,(0,10),"average Moliere angle [degrees]","zoomed")
    ("shr_score", 20, (0, 0.5), "shr score"),
    ("subcluster", 20, (0, 40), "N sub-clusters in shower"),
    # ('subcluster',20,(0,80),"N sub-clusters in shower","extended"),
    ("secondshower_Y_nhit", 20, (0, 200), "Nhit 2nd shower (Y)"),
    ("secondshower_Y_dot", 20, (-1, 1), "cos(2nd shower direction wrt vtx) (Y)"),
    ("anglediff_Y", 14, (0, 350), "angle diff 1st-2nd shower (Y) [degrees]"),
    ("secondshower_Y_vtxdist", 20, (0.0, 200), "vtx dist 2nd shower (Y)"),
    ("secondshower_V_nhit", 20, (0, 200), "Nhit 2nd shower (V)"),
    ("secondshower_V_dot", 20, (-1, 1), "cos(2nd shower direction wrt vtx) (V)"),
    ("anglediff_V", 14, (0, 350), "angle diff 1st-2nd shower (V) [degrees]"),
    ("secondshower_V_vtxdist", 20, (0.0, 200), "vtx dist 2nd shower (V)"),
    ("secondshower_U_nhit", 20, (0, 200), "Nhit 2nd shower (U)"),
    ("secondshower_U_dot", 20, (-1, 1), "cos(2nd shower direction wrt vtx) (U)"),
    ("anglediff_U", 14, (0, 350), "angle diff 1st-2nd shower (U) [degrees]"),
    ("secondshower_U_vtxdist", 20, (0.0, 200), "vtx dist 2nd shower (U)"),
    ("shr_tkfit_dedx_max", 15, (0, 10), "shr tkfit dE/dx (max, 0-4 cm) [MeV/cm]"),
    ("shr_trk_sce_start_y", 20, (-120, 120), "shr_trk_sce_start y [cm]"),
    ("shr_trk_sce_end_y", 20, (-120, 120), "shr_trk_sce_end y [cm]"),
    ("shr_trk_len", 40, (0, 400), "Shower track fit length [cm]"),
    ("shr_tkfit_2cm_dedx_Y", 10, (0.5, 10.5), "shr tkfit dE/dx (Y, 0-2 cm) [MeV/cm]"),
    ("shr_tkfit_2cm_dedx_V", 10, (0.5, 10.5), "shr tkfit dE/dx (V, 0-2 cm) [MeV/cm]"),
    ("shr_tkfit_2cm_dedx_U", 10, (0.5, 10.5), "shr tkfit dE/dx (U, 0-2 cm) [MeV/cm]"),
    ("shr_tkfit_gap10_dedx_Y", 10, (0.5, 10.5), "shr tkfit dE/dx (Y, 1-5 cm) [MeV/cm]"),
    ("shr_tkfit_gap10_dedx_V", 10, (0.5, 10.5), "shr tkfit dE/dx (V, 1-5 cm) [MeV/cm]"),
    ("shr_tkfit_gap10_dedx_U", 10, (0.5, 10.5), "shr tkfit dE/dx (U, 1-5 cm) [MeV/cm]"),
]
trksel_variables = [
    ("tksh_angle", 20, (-1, 1), "cos(trk-shr angle)"),
    ("trkshrhitdist2", 20, (0, 10), "2D trk-shr distance (Y)"),
    ("tksh_distance", 20, (0, 40), "trk-shr distance [cm]"),
    # ('tksh_distance',12,(0,6),"trk-shr distance [cm]","zoomed")
    ("trkpid", 21, (-1, 1), "track LLR PID"),
    # ('trkpid',2,(-1,1),"track LLR PID", 'twobins'),
    # ('trkpid',15,(-1,1),"track LLR PID","coarse")
]
shr2sel_variables = [
    ("shr2_moliereavg", 20, (0, 50), "shr2 average Moliere angle [degrees]"),
    ("shr2_score", 20, (0, 0.5), "shr2 score"),
    ("subcluster2", 20, (0, 40), "N sub-clusters in shower2"),
    ("tk1sh2_distance", 20, (0, 40), "trk1-shr2 distance [cm]"),
]
bdtscore_variables = [
    # ('nonpi0_score',10,(0.,0.5),"BDT non-$\pi^0$ score", "low_bdt"),
    ("nonpi0_score", 10, (0.5, 1.0), "BDT non-$\pi^0$ score", "high_bdt"),
    ("nonpi0_score", 10, (0, 1.0), "BDT non-$\pi^0$ score"),
    # ('nonpi0_score',10,(0,1.0),"BDT non-$\pi^0$ score", "log", True),
    # ('pi0_score',10,(0.,0.5),"BDT $\pi^0$ score", "low_bdt"),
    ("pi0_score", 10, (0.5, 1.0), "BDT $\pi^0$ score", "high_bdt"),
    ("pi0_score", 10, (0, 1.0), "BDT $\pi^0$ score"),
    # ('pi0_score',10,(0,1.0),"BDT $\pi^0$ score", "log", True),
    # ('bkg_score',10,(0,1.0),"1e0p BDT score"),
    # ('bkg_score',10,(0,1.0),"1e0p BDT score", "log", True),
    ("bkg_score", 10, (0, 1.0), "1e0p BDT score"),
    # ('bkg_score',5,(0,0.5),"1e0p BDT score","low_bdt"),
    ("bkg_score", 5, (0.5, 1.0), "1e0p BDT score", "high_bdt"),
    # ('bkg_score',10,(0,1.0),"1e0p BDT score", "log", True),
]
energy_variables = [
    ("trk_energy_tot", 10, (0, 2), "trk energy (range, P) [GeV]"),
    ("shr_energy_tot_cali", 10, (0, 2), "shr energy (calibrated) [GeV]"),
    ("reco_e", 20, (0.15, 2.95), r"Reconstructed Energy [GeV]", "note"),
    # ('NeutrinoEnergy0', 20, (0,2000), r"Reconstructed Calorimetric Energy U [MeV]"),
    # ('NeutrinoEnergy1', 20, (0,2000), r"Reconstructed Calorimetric Energy V [MeV]"),
    # ('NeutrinoEnergy2', 20, (0,2000), r"Reconstructed Calorimetric Energy Y [MeV]"),
]

kinematic_variables = [
    ("protonenergy", 12, (0, 0.6), "proton kinetic energy [GeV]"),
    ("pt", 10, (0, 2), "pt [GeV]"),
    ("ptOverP", 20, (0, 1), "pt/p"),
    ("phi1MinusPhi2", 13, (-6.5, 6.5), "shr phi - trk phi"),
    ("theta1PlusTheta2", 13, (0, 6.5), "shr theta + trk theta"),
    ("trk_theta", 21, (0, 3.14), r"Track $\theta$"),
    ("trk_phi", 21, (-3.14, 3.14), r"Track $\phi$"),
    ("trk_len", 20, (0, 100), "Track length [cm]"),
    ("trk_len", 21, (0, 20), "Track length [cm]", "zoom"),
    ("shr_theta", 21, (0, 3.14), r"Shower $\theta$"),
    ("shr_phi", 21, (-3.14, 3.14), r"Shower $\phi$"),
    ("n_trks_gt10cm", 6, (-0.5, 5.5), "n tracks longer than 10 cm"),
    # ('n_trks_gt25cm',6,(-0.5, 5.5),"n tracks longer than 25 cm"),
]
other_variables = [
    ("slclustfrac", 20, (0, 1), "slice clustered fraction"),
    ("reco_nu_vtx_x", 10, (0, 260), "vertex x [cm]"),
    ("reco_nu_vtx_y", 10, (-120, 120), "vertex y [cm]"),
    ("reco_nu_vtx_z", 10, (0, 1000), "vertex z [cm]"),
    # ('slnhits',20,(0.,5000),"N total slice hits"),
    # ('hits_u',20,(0.,1000),"N clustered hits U plane"),
    # ('hits_v',20,(0.,1000),"N clustered hits V plane"),
    # ('hits_y',20,(0.,1000),"N clustered hits Y plane"),
    # ('trk_hits_tot',20,(0.,2000),"Total N hits in tracks"),
    # ('trk_hits_u_tot',20,(0.,700),"Total N hits in tracks (U)"),
    # ('trk_hits_v_tot',20,(0.,700),"Total N hits in tracks (V)"),
    # ('trk_hits_y_tot',20,(0.,700),"Total N hits in tracks (Y)"),
    # ('shr_hits_tot',20,(0.,2000),"Total N hits in showers"),
    # ('shr_hits_u_tot',20,(0.,800),"Total N hits in showers (U)"),
    # ('shr_hits_v_tot',20,(0.,800),"Total N hits in showers (V)"),
    # ('shr_hits_y_tot',20,(0.,800),"Total N hits in showers (Y)"),
    ("topological_score", 20, (0, 1), "topological score"),
    ("trk_score", 20, (0.5, 1.0), "trk score"),
    ("shr_tkfit_nhits_tot", 20, (0, 20), "shr tkfit nhits (tot, 0-4 cm) [MeV/cm]"),
    # ('shrsubclusters0',20,(0,20),"N sub-clusters in shower (U)"),
    # ('shrsubclusters1',20,(0,20),"N sub-clusters in shower (V)"),
    # ('shrsubclusters2',20,(0,20),"N sub-clusters in shower (Y)"),
    ("shrMCSMom", 20, (0, 200), "shr mcs mom [MeV]"),
    ("DeltaRMS2h", 20, (0, 10), "Median spread of spacepoints"),
    ("CylFrac2h_1cm", 20, (0, 1), "Frac. of spacepoints in 1cm cylinder (2nd half of shr)"),
    ("shrPCA1CMed_5cm", 10, (0.5, 1), "Median of 1st component of shr PCA (5cm window)"),
]
pi0_variables = [
    ("reco_e",20, (0.0,2.0), "Reconstructed Energy [GeV]"),
    ("pi0_gammadot", 20, (-1, 1), "$\pi^0$ $\gamma_{\\theta\\theta}$"),
    ("pi0energy", 20, (135, 1135), "$\pi^0$ Energy [MeV]"),
    ("pi0energyraw", 20, (0, 1135), "$\pi^0$ Calorimeric Energy $E_1 + E_2$ [MeV]"),
    ("pi0momentum", 20, (0, 1000), "$\pi^0$ Momentum [MeV]"),
    ("pi0beta", 40, (0, 1), "$\pi^0$ $\\beta$"),
    ("pi0momanglecos", 40, (0, 1), "$\pi^0$ $\cos\theta$"),
    ("epicospi", 40, (0, 800), "$\pi^0$ $\cos{\theta}$ $\times$ $E_{\pi}$"),
    ("asymm", 20, (0, 1), "$\pi^0$ asymmetry $\\frac{|E_1-E_2|}{E_1+E_2}$"),
    ("pi0thetacm", 20, (0, 1), "$\cos\\theta_{\gamma}^{CM} = \\frac{1}{\\beta_{\pi^0}} \\frac{|E_1-E_2|}{E_1+E_2}$"),
    ("pi0_mass_Y_corr", 49, (10, 500), "$\pi^0$ mass [MeV]"),
    ("reco_e", 19, (0.15, 2.15), "reconstructed energy [GeV]"),
    ("shr_energy_tot_cali", 20, (0.05, 1.50), "reconstructed shower energy [GeV]"),
    ("trk_energy_tot", 20, (0.05, 1.50), "reconstructed track energy [GeV]"),
    ("n_tracks_contained", 5, (0, 5), "number of contained tracks"),
    ("n_showers_contained", 5, (2, 7), "number of contained showers"),
    ("pi0_mass_U", 20, (10, 510), "$M_{\gamma\gamma}$ mass U plane [MeV]"),
    ("pi0_mass_V", 20, (10, 510), "$M_{\gamma\gamma}$ mass V plane [MeV]"),
    ("pi0_mass_Y", 20, (10, 510), "$M_{\gamma\gamma}$ mass Y plane [MeV]"),
    ("pi0_shrscore1", 20, (0, 1), "leading $\gamma$ shower score"),
    ("pi0_shrscore2", 20, (0, 1), "sub-leading $\gamma$ shower score"),
    ("pi0_radlen1", 20, (3, 103), "leading $\gamma$ shower conversion distance [cm]"),
    ("pi0_radlen2", 20, (3, 103), "sub-leading $\gamma$ shower conversion distance [cm]"),
    ("pi0_energy1_Y", 20, (60, 460), "leading $\gamma$ shower energy [MeV]"),
    ("pi0_energy2_Y", 20, (40, 240), "sub-leading $\gamma$ shower energy [MeV]"),
    ("pi0_dedx1_fit_Y", 20, (1.0, 11.0), "leading $\gamma$ shower dE/dx [MeV/cm]"),
    # ('pi0_radlen1',25,(0,100),"leading $\gamma$ shower conversion distance [cm]"),
    # ('pi0_radlen2',25,(0,100),"sub-leading $\gamma$ shower conversion distance [cm]"),
    # ('pi0_energy1_Y',25,(0,500),"leading $\gamma$ shower energy [MeV]"),
    # ('pi0_energy2_Y',25,(0,250),"sub-leading $\gamma$ shower energy [MeV]"),
    # ('pi0_dedx1_fit_Y',20,(1.0,11.0),"leading $\gamma$ shower dE/dx [MeV/cm]"),
]
pi0_truth_variables = [
    ("shr_bkt_pdg", 20, (5, 25), r"shower backtracked pdg"),
    ("trk_bkt_pdg", 20, (5, 2500), r"track backtracked pdg"),
    ("pi0truth_gamma1_dist", 20, (0, 100), r"leading photon conv. distance [cm]"),
    ("pi0truth_gamma2_dist", 20, (0, 100), r"sub-leading photon conv. distance [cm]"),
    ("pi0truth_gamma1_etot", 20, (0, 1000), r"leading photon true Energy [ MeV ]"),
    ("pi0truth_gamma2_etot", 20, (0, 500), r"sub-leading photon true Energy [ MeV ]"),
    ("pi0truth_gamma1_edep", 20, (0, 1000), r"leading photon deposited Energy [ MeV ]"),
    ("pi0truth_gamma2_edep", 20, (0, 500), r"sub-leading photon deposited Energy [ MeV ]"),
    ("pi0truth_gamma1_edep_frac", 20, (0, 1), r"leading photon deposited/total Energy"),
    ("pi0truth_gamma2_edep_frac", 20, (0, 1), r"sub-leading photon deposited/toal Energy"),
    ("true_nu_vtx_x", 12, (0, 252), "true vtx x [cm]"),
    ("true_nu_vtx_y", 12, (-120, 120), "true vtx y [cm]"),
    ("true_nu_vtx_z", 12, (0, 1200), "true vtx z [cm]"),
    # ('weightSplineTimesTune',20,(0,2),"event weight"),
    ("pi0truth_gammadot", 20, (-1, 1), "cos opening angle"),
    ("muon_e", 20, (0.0, 1.0), r"Muon Energy [ GeV ]"),
]

shr12_variables = [
    ("hitratio_shr12", 10, (0, 1), "hit ratio two showers"),
    ("min_tksh_dist", 20, (0, 40), "min tksh dist of two showers"),
    ("max_tksh_dist", 20, (0, 40), "max tksh dist of two showers"),
    ("tksh2_dist", 20, (0, 40), "tksh dist of second shower"),
    ("cos_shr12", 10, (-1, 1), "cos two showers"),
]
run_variables = [
    ("run", 100, (4500, 19500), "run number"),
]

numupresel_variables = [
    ("muon_candidate_start_x", 28, (0, 260), "muon candidate start x [cm]"),
    ("muon_candidate_start_y", 28, (-120, 120), "muon candidate start y [cm]"),
    ("muon_candidate_start_z", 28, (0, 1030), "muon candidate start z [cm]"),
    ("muon_candidate_end_x", 28, (0, 260), "muon candidate end x [cm]"),
    ("muon_candidate_end_y", 28, (-120, 120), "muon candidate end y [cm]"),
    ("muon_candidate_end_z", 28, (0, 1030), "muon candidate end z [cm]"),
    ("muon_candidate_score", 26, (0, 1), "muon candidate track score"),
    ("muon_candidate_pid", 26, (-1, 1), "muon candidate PID score"),
    ("muon_candidate_mcs", 26, (-1, 2.5), "muon candidate MCS consistency"),
    ("muon_candidate_length", 24, (0, 900), "muon candidate length [cm]"),
    ("muon_candidate_distance", 26, (0, 5), "muon candidate vtx distance [cm]"),
]


numusel_variables = [
    ("muon_energy", 14, (0.15, 1.55), "muon candidate reconstructed energy [GeV]"),
    ("neutrino_energy", 14, (0.15, 1.55), "neutrino reconstructed energy [GeV]"),
    ("muon_theta", 28, (-1, 1), r"muon candidate $\cos(\theta)$"),
]

bdt_common_variables_1eNp = [
    ("shr_score", 10, (0, 0.5), "shr score"),
    ("trkfit", 10, (0, 0.65), "Fraction of Track-fitted points", "zoomed"),
    ("subcluster", 10, (4, 44), "N sub-clusters in shower"),
    ("shrmoliereavg", 9, (0, 9), "average Moliere angle [degrees]"),
    ("CosmicIPAll3D", 10, (10, 200), "CosmicIPAll3D [cm]"),
    ("CosmicDirAll3D", 10, (-1, 1), "cos(CosmicDirAll3D)"),
    ("secondshower_Y_nhit", 10, (0, 200), "Nhit 2nd shower (Y)"),
    ("secondshower_Y_dot", 10, (0, 1), "cos(2nd shower direction wrt vtx) (Y)"),
    ("anglediff_Y", 10, (0, 350), "angle diff 1st-2nd shower (Y) [degrees]"),
    ("secondshower_Y_vtxdist", 10, (0.0, 200), "vtx dist 2nd shower (Y)"),
]

bdt_1enp_variables = [
    ("tksh_angle", 10, (-0.9, 1), "cos(trk-shr angle)"),
    ("trkshrhitdist2", 10, (0, 10), "2D trk-shr distance (Y)"),
    ("tksh_distance", 6, (0, 6), "trk-shr distance [cm]"),
    ("trkpid", 5, (-1, 0.02), "track LLR PID"),
    ("hits_ratio", 10, (0.5, 1.0), "shower hits/all hits"),
    ("shr_tkfit_dedx_max", 5, (0.5, 5.5), "shr tkfit dE/dx (max, 0-4 cm) [MeV/cm]"),
]

bdt_common_variables_1e0p = [
    ("shr_score", 10, (0, 0.5), "shr score"),
    ("trkfit", 10, (0, 0.65), "Fraction of Track-fitted points"),
    ("subcluster", 10, (4, 44), "N sub-clusters in shower"),
    ("shrmoliereavg", 9, (0, 15), "average Moliere angle [degrees]"),
    ("CosmicIPAll3D", 10, (10, 200), "CosmicIPAll3D [cm]"),
    ("CosmicDirAll3D", 10, (-1, 1), "cos(CosmicDirAll3D)"),
    ("secondshower_Y_nhit", 10, (0, 200), "Nhit 2nd shower (Y)"),
    ("secondshower_Y_dot", 10, (0, 1), "cos(2nd shower direction wrt vtx) (Y)"),
    ("anglediff_Y", 10, (0, 350), "angle diff 1st-2nd shower (Y) [degrees]"),
    ("secondshower_Y_vtxdist", 10, (0.0, 200), "vtx dist 2nd shower (Y)"),
]

bdt_1e0p_variables = [
    ("shr_tkfit_2cm_dedx_U", 10, (0, 10), "shr tkfit dE/dx (U, 0-2 cm) [MeV/cm]"),
    ("shr_tkfit_2cm_dedx_V", 10, (0, 10), "shr tkfit dE/dx (V, 0-2 cm) [MeV/cm]"),
    ("shr_tkfit_2cm_dedx_Y", 10, (0, 10), "shr tkfit dE/dx (Y, 0-2 cm) [MeV/cm]"),
    ("shr_tkfit_gap10_dedx_U", 10, (0, 10), "shr tkfit dE/dx (U, 1-5 cm) [MeV/cm]"),
    ("shr_tkfit_gap10_dedx_V", 10, (0, 10), "shr tkfit dE/dx (V, 1-5 cm) [MeV/cm]"),
    ("shr_tkfit_gap10_dedx_Y", 10, (0, 10), "shr tkfit dE/dx (Y, 1-5 cm) [MeV/cm]"),
    ("secondshower_U_nhit", 10, (0, 200), "Nhit 2nd shower (U)"),
    ("secondshower_U_dot", 10, (-1, 1), "cos(2nd shower direction wrt vtx) (U)"),
    ("anglediff_U", 10, (0, 350), "angle diff 1st-2nd shower (U) [degrees]"),
    ("secondshower_U_vtxdist", 10, (0.0, 200), "vtx dist 2nd shower (U)"),
    ("secondshower_V_nhit", 10, (0, 200), "Nhit 2nd shower (V)"),
    ("secondshower_V_dot", 10, (-1, 1), "cos(2nd shower direction wrt vtx) (V)"),
    ("anglediff_V", 10, (0, 350), "angle diff 1st-2nd shower (V) [degrees]"),
    ("secondshower_V_vtxdist", 10, (0.0, 200), "vtx dist 2nd shower (V)"),
    ("shrMCSMom", 10, (0, 200), "shr mcs mom [MeV]"),
    ("DeltaRMS2h", 10, (0, 10), "Median spread of spacepoints"),
    ("CylFrac2h_1cm", 10, (0, 1), "Frac. of spacepoints in 1cm cylinder (2nd half of shr)"),
    ("shrPCA1CMed_5cm", 10, (0.5, 1), "Median of 1st component of shr PCA (5cm window)"),
]

vtx_variables = [
    ("reco_nu_vtx_sce_x", 5, (0, 260), "reco neutrino vertex x [cm]"),
    ("reco_nu_vtx_sce_y", 5, (-120, 120), "reco neutrino vertex y [cm]"),
    ("reco_nu_vtx_sce_z", 5, (0, 1030), "reco neutrino vertex z [cm]"),
]

tki_truth_variables_1mu1p = [
    ("TrueDeltaPT_1mu1p", 10, (0.0,1.2), "delta pT"),
    ("TrueDeltaPhiT_1mu1p", 10, (0.0,3.142), "delta phiT"),
    ("TrueDeltaAlphaT_1mu1p", 10, (0.0,3.142), "delta alphaT"),
    ("TrueECal_1mu1p", 10, (0.1,2.0), "Ecal"),
    ("TruePL_1mu1p", 10, (0.0,0.75), "pL"),
    ("TruePN_1mu1p", 10, (0.0,1.0), "pn"),
    ("TrueAlpha3D_1mu1p", 10, (0.0,3.142), "alpha 3D"),
    ("TruePhi3D_1mu1p", 10, (0.0,3.142), "phi 3D"),
    ("TrueDeltaPTX_1mu1p", 10, (-0.5,0.5), "delta pTX"),
    ("TrueDeltaPTY_1mu1p", 10, (-0.5,0.5), "delta pTY"),
    ("TruePNTX_1mu1p", 10, (-0.3,0.3), "pnTx"),
    ("TruePNTY_1mu1p", 10, (-0.5,0.5), "pnTy"),
    ("TruePNT_1mu1p", 10, (0.0,0.5), "pnT"),
    ("TruePNII_1mu1p", 10, (-0.5,0.75), "pnII"),
]

tki_reco_variables_1mu1p = [
    ("RecoDeltaPT_1mu1p", 10, (0.0,1.2), "delta pT"),
    ("RecoDeltaPhiT_1mu1p", 10, (0.0,3.142), "delta phiT"),
    ("RecoDeltaAlphaT_1mu1p", 10, (0.0,3.142), "delta alphaT"),
    ("RecoECal_1mu1p", 10, (0.1,2.0), "Ecal"),
    ("RecoPL_1mu1p", 10, (0.0,0.75), "pL"),
    ("RecoPN_1mu1p", 10, (0.0,1.0), "pn"),
    ("RecoAlpha3D_1mu1p", 10, (0.0,3.142), "alpha 3D"),
    ("RecoPhi3D_1mu1p", 10, (0.0,3.142), "phi 3D"),
    ("RecoDeltaPTX_1mu1p", 10, (-0.5,0.5), "delta pTX"),
    ("RecoDeltaPTY_1mu1p", 10, (-0.5,0.5), "delta pTY"),
    ("RecoPNTX_1mu1p", 10, (-0.3,0.3), "pnTx"),
    ("RecoPNTY_1mu1p", 10, (-0.5,0.5), "pnTy"),
    ("RecoPNT_1mu1p", 10, (0.0,0.5), "pnT"),
    ("RecoPNII_1mu1p", 10, (-0.5,0.75), "pnII"),
]

# CT: Adding sideband variables
NP_far_sideband_variables = [
    ("shr_energy_tot_cali", 10, (0.05,0.7), "shr energy (calibrated) [GeV]"),
    ("reco_e", 10, (0.85, 2.05), r"Reconstructed Energy [GeV]", "note"),
    ("trk_energy_tot", 10, (0, 2), "trk energy (range, P) [GeV]"),
] + bdt_common_variables_1eNp

NP_near_sideband_variables = [
    ("shr_energy_tot_cali", 10, (0.0,0.7), "shr energy (calibrated) [GeV]"),
    ("reco_e", 10, (0.65, 0.85), r"Reconstructed Energy [GeV]", "note"),
    ("trk_energy_tot", 10, (0, 1), "trk energy (range, P) [GeV]"),
] + bdt_common_variables_1eNp

ZP_far_sideband_variables = [
    ("shr_energy_tot_cali", 10, (0.0, 0.7), "shr energy (calibrated) [GeV]"),
    ("reco_e", 10, (0.90, 2.50), r"Reconstructed Energy [GeV]", "note"),
] + bdt_common_variables_1e0p

ZP_near_sideband_variables = [
    ("shr_energy_tot_cali", 10, (0.5, 1.0), "shr energy (calibrated) [GeV]"),
    ("reco_e", 10, (0.65, 0.90), r"Reconstructed Energy [GeV]", "note"),
] + bdt_common_variables_1e0p


plot_variables = basic_variables + evtsel_variabls + shrsel_variables + bdtscore_variables
plot_variables += kinematic_variables
plot_variables += tki_truth_variables_1mu1p
