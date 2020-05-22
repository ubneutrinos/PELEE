"""
This script is a collection of redundant bits of code that don't exactly belong in the Plotter-NUMU notebook
Lots of these functions modify parameters that save files and modify plots
"""
#import management
import autoreload
#%load_ext autoreload
#%autoreload 2  # Autoreload all modules
import importlib
#standard imports
import os
from datetime import datetime
#scientific imports
import pandas as pd
import numpy as np
#user-defined modules
import plotter
import localSettings as ls

importlib.reload(plotter)

AVx = [-1.55,254.8]
AVy = [-115.53, 117.47]
AVz = [0.1, 1036.9]
FVx = [5,251]                      #[10,246]
FVy = [-110,110]                   #[-105,105]
FVz = [20,986]


presel_vars = ['nslice',
    'topological_score','reco_nu_vtx_sce_x', 'reco_nu_vtx_sce_y', 'reco_nu_vtx_sce_z',
]
muonsel_vars = [
    'trk_sce_start_x_v', 'trk_sce_start_y_v', 'trk_sce_start_z_v',
    'trk_sce_end_x_v', 'trk_sce_end_y_v', 'trk_sce_end_z_v',
    'trk_p_quality_v', 'trk_score_v', 'trk_llr_pid_score_v',
    'trk_len_v', 'pfp_generation_v', 'trk_distance_v',
]
other_vars = ['nslice','run',
    'trk_theta_v', 'trk_phi_v',
    'trk_cos_theta_v', 'trk_cos_phi_v',
    'reco_ntrack', 'reco_nproton',
    'reco_nu_e_range_v', 'trk_energy_proton_v', 'trk_range_muon_e_v',
    'trk_dx_v', 'trk_dy_v', 'trk_dz_v',
    'trk_range_proton_mom_v','trk_range_muon_mom_v',
    'trk_energy_proton_v',
    'NeutrinoEnergy2'
]
crt_vars = [
    'crtveto', 'crthitpe','_closestNuCosmicDist'
]
mc_vars = [
     'nu_e', 'theta', 'category', 'interaction','ccnc',
     'nmuon', 'nproton', 'npi0', 'npion', 'backtracked_pdg_v', 'nu_pdg',
     "weightSpline","weightTune","weightSplineTimesTune",
     #'phi',
]
weightsFlux = [
    "weightsGenie", "weightsFlux",
]
vars_agnostic = presel_vars + muonsel_vars + other_vars

load_vars = [
    "nslice", "selected", "nu_pdg",
    "slpdg", "trk_score_v","slclustfrac",
    #"contained_fraction",
    "backtracked_pdg","category",
    "topological_score",
    "run", "sub", "evt",
    "reco_nu_vtx_sce_x","reco_nu_vtx_sce_y","reco_nu_vtx_sce_z",
    "trk_sce_start_x_v","trk_sce_start_y_v","trk_sce_start_z_v",
    "trk_sce_end_x_v","trk_sce_end_y_v","trk_sce_end_z_v",
    "trk_mcs_muon_mom_v","trk_range_muon_mom_v", "trk_len_v",
    'trk_llr_pid_score_v',
    "pfp_generation_v","trk_distance_v","trk_theta_v","trk_phi_v",
    #"trk_energy_muon","trk_energy_tot","trk_energy",
    'trk_energy_muon_v','trk_energy_proton_v',
    "pfnhits","pfnunhits",
    'slnunhits','slnhits',
    'NeutrinoEnergy2',
]

#store a bunch of batches of plots to put into the "make a bunch of plots" cell
def get_plots(label):
    if label == "presel input":
        muon_vars = [
            'topological_score',
            'reco_nu_vtx_sce_x','reco_nu_vtx_sce_y','reco_nu_vtx_sce_z',
            'crtveto','crthitpe','_closestNuCosmicDist',

        ]
        muon_bins = [25]*len(muon_vars)
        muon_bins[4] = 2
        muon_ranges = [
            (0,1),
            (AVx[0],AVx[1]+1),(AVy[0]-1,AVy[1]+1),(AVz[0]-1,AVz[1]), #bunch of MC is slightly OOAV
            (-.5,1.5),(0,500),(0,5),
        ]
        muon_titles = [
            'Topological Score',
            r'Reco $\nu$ Vertex X [cm]', r'Reco $\nu$ Vertex Y [cm]', r'Reco $\nu$ Vertex Z [cm]',
            'CRT veto', 'CRT hitPE', r'Closest $\nu$-Cosmic Distance [cm]',
        ]
    if label == "muon input":
            #all the variables cut on in the muon selection
            muon_vars = [
                'trk_sce_start_x_v','trk_sce_start_y_v','trk_sce_start_z_v',
                'trk_sce_end_x_v','trk_sce_end_y_v','trk_sce_end_z_v',
                'trk_score_v','trk_llr_pid_score_v','trk_p_quality_v',
                'trk_len_v','trk_distance_v','pfp_generation_v'
            ]
            muon_bins = [26]*len(muon_vars)
            muon_bins[-1] = 5

            muon_ranges = [
                (AVx[0],AVx[1]+1),(AVy[0]-1,AVy[1]+1),(AVz[0]-1,AVz[1]),
                (AVx[0],AVx[1]+1),(AVy[0]-1.5,AVy[1]+1.5),(AVz[0]-1,AVz[1]+.5),
                (0,1),(-1,1),(-1,2.5),
                (0,1000),(0,5),(-0.5,4.5)
            ]
            muon_titles = [
                'Muon Candidate Start, X [cm]', 'Muon Candidate Start, Y [cm]', 'Muon Candidate Start, Z [cm]',
                'Muon Candidate End, X [cm]', 'Muon Candidate End, Y [cm]', 'Muon Candidate End, Z [cm]',
                'Track Score', 'Log-Likelihood PID Score', r'MCS Consistency $(\frac{P_{MCS}-P_{Range}}{P_{Range}})$',
                'Track Length [cm]', 'Track Distance from Reco Vertex [cm]', 'PFP Generation'
            ]
    if label == "edges high":
        muon_vars = [
            'trk_sce_start_x_v','trk_sce_start_y_v','trk_sce_start_z_v',
            'trk_sce_end_x_v','trk_sce_end_y_v','trk_sce_end_z_v',
            #'reco_nu_vtx_sce_x','reco_nu_vtx_sce_y','reco_nu_vtx_sce_z'
        ]
        muon_ranges = [
            (FVx[1]-5,AVx[1]+3),(FVy[1]-5,AVy[1]+3),(FVz[1]-5,AVz[1]+3),
            (FVx[1]-5,AVx[1]+3),(FVy[1]-5,AVy[1]+3),(FVz[1]-5,AVz[1]+3),
        ]
        muon_titles = [
            'Muon Candidate Start, X [cm]', 'Muon Candidate Start, Y [cm]', 'Muon Candidate Start, Z [cm]',
            'Muon Candidate End, X [cm]', 'Muon Candidate End, Y [cm]', 'Muon Candidate End, Z [cm]',
            #r'Reco Range-Based $\nu$ Vertex X [GeV]',r'Reco Range-Based $\nu$ Vertex Y [GeV]',r'Reco Range-Based $\nu$ Vertex Z [GeV]'
        ]
        muon_bins = [26]*len(muon_vars)
        muon_ranges = [
            (AVx[0]-3,FVx[0]+5),(AVy[0]-3,FVy[0]+5),(AVz[0]-3,FVz[0]+5),
            (AVx[0]-3,FVx[0]+5),(AVy[0]-3,FVy[0]+5),(AVz[0]-3,FVz[0]+5),
        ]
    if label == "edges low":
        muon_vars = [
            'trk_sce_start_x_v','trk_sce_start_y_v','trk_sce_start_z_v',
            'trk_sce_end_x_v','trk_sce_end_y_v','trk_sce_end_z_v',
            #'reco_nu_vtx_sce_x','reco_nu_vtx_sce_y','reco_nu_vtx_sce_z'
        ]
        muon_titles = [
            'Muon Candidate Start, X [cm]', 'Muon Candidate Start, Y [cm]', 'Muon Candidate Start, Z [cm]',
            'Muon Candidate End, X [cm]', 'Muon Candidate End, Y [cm]', 'Muon Candidate End, Z [cm]',
            #r'Reco Range-Based $\nu$ Vertex X [GeV]',r'Reco Range-Based $\nu$ Vertex Y [GeV]',r'Reco Range-Based $\nu$ Vertex Z [GeV]'
        ]
        muon_bins = [26]*len(muon_vars)
        muon_ranges = [
            (AVx[0]-3,FVx[0]+5),(AVy[0]-3,FVy[0]+5),(AVz[0]-3,FVz[0]+5),
            (AVx[0]-3,FVx[0]+5),(AVy[0]-3,FVy[0]+5),(AVz[0]-3,FVz[0]+5),
        ]
    if label == "proton multiplicity":
        muon_vars = [
            'reco_nproton', 'nproton'
        ]
        muon_bins = [5,5]
        muon_ranges = [
            (-0.5,4.5), (-0.5,4.5)
        ]
        muon_titles = ['Reco Protons ()']
    if label == 'CCQE':
        muon_vars = [
            'Q2','Mhad',
            'Xbj','Ybj',
            'pT','pL','pTransverseRatio',
            'opening_angle','phi_diff','theta_tot',
            'trk_len',
        ]
        muon_bins = [25]*len(muon_vars)
        muon_ranges = [
            (0,0.55),(0,0.55),
            (0,3),(0,0.8),
            (0,1.2),(0,2),(0,1),
            (0,3.14159),(-2*3.14159,2*3.14159),(0,5),
            (0,600),
        ]
        muon_titles = [
            r'Q$^2$', r'M$_{had}$ [GeV]',
            r"Bj$\"o$rken's Scaling X", r"Bj$\"o$rken's Scaling Y",
            'Total Transverse Momentum [GeV]', 'Total Linear Momentum [GeV$^2$]', 'Total Transverse Ratio',
            r'$\mu$-p Opening Angle', r'$\phi_{l} - \phi_{p}$', r'$\theta_{l} + \theta_{p}$',
            'Length of longest track [cm]'
        ]
    if label == 'CCQE_proton_kinematics':
        muon_vars = [
            'trk_theta_v', 'trk_phi_v',
            'E',
            'p', 'pT', 'pTransverseRatio',
            'trk_len',
        ]
        muon_bins = [25]*len(muon_vars)
        muon_ranges = [
            (0,3.14159), (-3.1415,3.1415),
            (0.15,0.55),
            (0.25,1),(0,0.8),(0,1),
            (0,120),
        ]
        muon_titles = [
            r'Track $\theta$', r'Track $\phi$',
            r'Reco Range-Based Track Energy [GeV]',
            'Track Range-Based Momentum [GeV]', 'Track Transverse Momentum [GeV]', r'$P_{T}$/P',
            'Track Length [cm]',
        ]
    if label == 'CCQE_muon_kinematics':
        muon_vars = [
            'trk_theta_v', 'trk_phi_v',
            'E',
            'p', 'pT', 'pTransverseRatio',
            'trk_len',
        ]
        muon_bins = [25]*len(muon_vars)
        muon_ranges = [
            (0,3.14159), (-3.1415,3.1415),
            (0.15,1.5),
            (0.15,1.5),(0,0.7),(0,1),
            (0,600),
        ]
        muon_titles = [
            r'Track $\theta$', r'Track $\phi$',
            r'Reco Range-Based Track Energy [GeV]',
            'Track Range-Based Momentum [GeV]', 'Track Transverse Momentum [GeV]', r'$P_{T}$/P',
            'Track Length [cm]',
        ]
    return muon_vars, muon_bins, muon_ranges, muon_titles

def draw_FV_AV(ax1, tag, VARIABLE):
    #if doing studies close to edge of detector, it's useful to draw active volume
    #this is a very crude way of going this. could be updated, but it's not used much
    if "low" in tag:
        idx = 0
    elif "high" in tag:
        idx = 1
    if '_x' in VARIABLE:
        ax1.plot([AVx[idx],AVx[idx]],[0,ax1.get_ylim()[1]],'r--', label='AVx: {}'.format(AVx[idx]))
        ax1.plot([FVx[idx],FVx[idx]],[0,ax1.get_ylim()[1]],'g--', label='FVx: {}'.format(FVx[idx]))
    if '_y' in VARIABLE:
        ax1.plot([AVy[idx],AVy[idx]],[0,ax1.get_ylim()[1]],'r--', label='AVy: {}'.format(AVy[idx]))
        ax1.plot([FVy[idx],FVy[idx]],[0,ax1.get_ylim()[1]],'g--', label='FVy: {}'.format(FVy[idx]))
    if '_z' in VARIABLE:
        ax1.plot([AVz[idx],AVz[idx]],[0,ax1.get_ylim()[1]],'r--', label='AVz: {}'.format(AVz[idx]))
        ax1.plot([FVz[idx],FVz[idx]],[0,ax1.get_ylim()[1]],'g--', label='FVz: {}'.format(FVz[idx]))
    ax1.legend()

def make_filename(VARIABLE, date_time, tag, detsys=None):
    # "consistency", just call this function instead of making up a new one
    fn = VARIABLE+"_"+date_time+"_"+tag
    if detsys:
        fn += "_detsys"
    fn += ".pdf"
    return fn

#scale factors for several runs/samples
def get_scaling(USECRT, ISG1, scaling=1):
    if USECRT:
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

#if both the data samples are loaded, have a way to switch between them
def update_data(data_sample, samples, USECRT, fullsel_samples=None):
    print("updating data sample to {}...".format(data_sample))
    data = samples[data_sample]
    if data_sample == "data_1e20": weights,pot = get_scaling(USECRT, True)
    elif data_sample == "data_7e18": weights,pot = get_scaling(USECRT, False)
    else: weights,pot =  get_scaling(USECRT, False)
    samples['data'] = data
    return samples, weights, pot

###############################################
# Calculate Purity and Efficiency
def Eff(df, df_cut, var,acceptance,bin_edges,absval=False):
    #print acceptance
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    bins = []
    bin_eff = []
    bin_err = []
    for i in range(len(bin_centers)):
        binmin = bin_edges[i]
        binmax = bin_edges[i+1]
        bincut = "{} > {} and {} < {}".format(var,binmin,var,binmax)
        if (absval == True):
            bincut = '(%s > %f and %s < %f) or (%s > -%f and %s < -%f)'%(var,binmin,var,binmax,var,binmax,var,binmin)
        #print bincut
        df_tmp =  df.query(bincut).query(acceptance) # denominator
        df_sub = df_cut.query(bincut).query(acceptance) # numerator

        if (df_tmp.shape[0] == 0): continue
        eff = df_sub.shape[0] / float( df_tmp.shape[0] )
        err = np.sqrt( eff*(1-eff)/df_tmp.shape[0] )
        bin_eff.append( eff )
        bin_err.append( err )
        bins.append(bin_centers[i])
        #print 'eff = %.02f @ bin = %.02f'%(eff,bin_centers[i])
    return np.array(bins),np.array(bin_eff),np.array(bin_err)

def Pur(samples, weights, var, acceptance, binedges):
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    bins = []
    bin_pur = []
    bin_err = []
    for i in range(len(bin_centers)):
        binmin = bin_edges[i]
        binmax = bin_edges[i+1]
        bincut = "{} > {} and {} < {}".format(var,binmin,var,binmax)
        df_num = samples['mc'].query(bincut).query(acceptance)
        num = df_num.shape[0]*weights['mc']
        den = 0
        for sample in ['mc','dirt','ext','nue']:
            den += samples[sample].query(bincut).shape[0]*weights[sample]
        pur = num * 1.0 / den
        err = np.sqrt(pur*(1-pur)/den)
        bin_pur.append(pur)
        bin_err.append(err)
        bins.append(bin_centers[i])
    return np.array(bins), np.array(bin_pur), np.array(bin_err)

#######################################
# Get DFs of final selection
# Will save time down the road since it takes FOREVER to implement track cuts
######################################
def get_longest_mask(df, mask):
    '''
    df: dataframe for sample
    variable: Series of values that pass cuts defined by mask
    mask: mask used to find variable

    returns
        list of values of variable corresponding to longest track in each slices
        boolean mask for longest tracks in df
    '''

    #print("selecting longest...")
    #print("mask", mask)
    trk_lens = (df['trk_len_v']*mask).apply(lambda x: x[x != False])#apply mask to track lengths
    trk_lens = trk_lens.apply(lambda x: x[~np.isnan(x)])#clean up nan vals
    trk_lens = trk_lens[trk_lens.apply(lambda x: len(x) > 0)]#clean up slices
    nan_mask = variable.apply(lambda x: np.nan in x or "nan" in x)
    longest_mask = trk_lens.apply(lambda x: x == x[list(x).index(max(x))])#identify longest
    return longest_mask

def make_mask(DF, query, track_cuts):
    df = DF.copy().query(query)
    mask = df['trk_len_v'].apply(lambda x: x==x) #all true mask
    #layer on each cut chipping down the all true mask
    for (var,op,val) in track_cuts:
        if type(op) == list:
            #relate the two operations with an 'or'
            or_mask1 = df[var].apply(lambda x: eval("x{}{}".format(op[0],val[0])))#or condition 1
            or_mask2 = df[var].apply(lambda x: eval("x{}{}".format(op[1],val[1])))#or condition 2
            mask *= (or_mask1 + or_mask2) #just add the booleans for "or"
        else:
            mask *= df[var].apply(lambda x: eval("x{}{}".format(op,val))) #layer on each cut mask

    return df, mask

def apply_mask(df, sample_name, mask, DETSYS, select_longest=False):
    #going to fresh new dataframe with only a few frames
    df_filtered = pd.DataFrame()
    common_vars = presel_vars + muonsel_vars + other_vars# + CCQE_vars
    temp = []
    [temp.append(str(x)) for x in common_vars if x not in temp] #remove redundancy
    common_vars = temp
    for var in common_vars:
        #apply mask to each variable
        VAR = (df[var]*mask).apply(lambda x: x[x != False])
        VAR = VAR.apply(lambda x: x[~np.isnan(x)])#clean up nan vals
        VAR = VAR[VAR.apply(lambda x: len(x) > 0)] #clean up empty slices
        if select_longest:
            trk_lens = (df['trk_len_v']*mask).apply(lambda x: x[x != False])#apply mask to track lengths
            trk_lens = trk_lens.apply(lambda x: x[~np.isnan(x)])#clean up nan vals
            trk_lens = trk_lens[trk_lens.apply(lambda x: len(x) > 0)]#clean up slices
            longest_mask = trk_lens.apply(lambda x: x == x[list(x).index(max(x))])#identify longest
            VAR = (VAR*longest_mask).apply(lambda x: x[x!=False])#apply mask
            VAR = VAR[VAR.apply(lambda x: len(x) > 0)] #clean up empty slices
        if var[-2:] != "_v":
            VAR = VAR.apply(lambda x: x[0])
        df_filtered[var] = VAR
    if sample_name in ['mc','nue','dirt']:
        uncommon_vars = mc_vars
        for var in uncommon_vars:
            try:
                VAR = (df[var]*mask).apply(lambda x: x[x != False])
                VAR = VAR.apply(lambda x: x[~np.isnan(x)])#clean up nan vals
                VAR = VAR[VAR.apply(lambda x: len(x) > 0)] #clean up empty slices
                if select_longest:
                    trk_lens = (df['trk_len_v']*mask).apply(lambda x: x[x != False])#apply mask to track lengths
                    trk_lens = trk_lens.apply(lambda x: x[~np.isnan(x)])#clean up nan vals
                    trk_lens = trk_lens[trk_lens.apply(lambda x: len(x) > 0)]#clean up slices
                    longest_mask = trk_lens.apply(lambda x: x == x[list(x).index(max(x))])#identify longest
                    VAR = (VAR*longest_mask).apply(lambda x: x[x!=False])#apply mask
                    VAR = VAR[VAR.apply(lambda x: len(x) > 0)] #clean up empty slices
            except:
                print(sample_name,var)
            try:
                VAR = VAR[VAR.apply(lambda x: len(x) > 0)] #clean up empty slices
            except:
                _=1
            if var[-2:] != "_v":
                VAR = VAR.apply(lambda x: x[0])
            df_filtered[var] = VAR
        if DETSYS:
            for var in weightsFlux:
                df_filtered[var] = df[var].loc[df_filtered.index]

    return df_filtered

def apply_muon_fullsel(DF, sample_name, USECRT, DETSYS):
    '''
    Returns dataframe will all cuts applied and longest track preselected
    '''
    query, track_cuts = get_NUMU_sel(USECRT, opfilter=False)
    df, mask = make_mask(DF, query, track_cuts)
    # need to protect the null values of certain variables
    if sample_name in ['mc','dirt','nue']:
        df['backtracked_pdg_v'] = df['backtracked_pdg_v'].apply(lambda x: [xx + 0.01 for xx in x])
        df['reco_nproton'] = df['reco_nproton'].apply(lambda x: x + 0.01)
    df_filtered = apply_mask(df.replace(0,-123456789), sample_name, mask, DETSYS, select_longest=True).replace(-123456789,0) #0's don't survive...
    if sample_name in ['mc','dirt','nue']:
        df_filtered['backtracked_pdg_v'] = df_filtered['backtracked_pdg_v'].apply(lambda x: [int(round(xx)) for xx in x])
        df_filtered['reco_nproton'] = df_filtered['reco_nproton'].apply(lambda x: int(round(x)))
    if sample_name == "mc":
        print("all vars: \n {}".format(list(df_filtered.keys())))
    return df_filtered

def apply_fullsel(DF, sample_name, USECRT, DETSYS):
    '''
    Returns dataframe will all cuts applied
    '''
    query, track_cuts = get_NUMU_sel(USECRT, opfilter=False)
    df, mask = make_mask(DF, query, track_cuts)
    # need to protect the null values of certain variables
    if sample_name in ['mc','dirt','nue']:
        df['backtracked_pdg_v'] = df['backtracked_pdg_v'].apply(lambda x: [xx + 0.01 for xx in x])
        df['reco_nproton'] = df['reco_nproton'].apply(lambda x: x + 0.01)
    df_filtered = apply_mask(df.replace(0,-123456789), sample_name, mask, DETSYS).replace(-123456789,0) #0's don't survive...
    if sample_name in ['mc','dirt','nue']:
        df_filtered['backtracked_pdg_v'] = df_filtered['backtracked_pdg_v'].apply(lambda x: [int(round(xx)) for xx in x])
        df_filtered['reco_nproton'] = df_filtered['reco_nproton'].apply(lambda x: int(round(x)))
    if sample_name == "mc":
        print("all vars: \n {}".format(list(df_filtered.keys())))
    return df_filtered


def apply_above105(DF, sample_name, USECRT, DETSYS):
    query, track_cuts = 'nslice == 1', [('reco_nu_e_range_v', '>', 1.05)]
    df, mask = make_mask(DF, query, track_cuts)
    # need to protect the null values of certain variables
    if sample_name in ['mc','dirt','nue']:
        df['backtracked_pdg_v'] = df['backtracked_pdg_v'].apply(lambda x: [xx + 0.01 for xx in x])
        df['reco_nproton'] = df['reco_nproton'].apply(lambda x: x + 0.01)
    df_filtered = apply_mask(df.replace(0,-123456789), sample_name, mask, DETSYS).replace(-123456789,0) #0's don't survive...
    if sample_name in ['mc','dirt','nue']:
        df_filtered['backtracked_pdg_v'] = df_filtered['backtracked_pdg_v'].apply(lambda x: [int(round(xx)) for xx in x])
        df_filtered['reco_nproton'] = df_filtered['reco_nproton'].apply(lambda x: int(round(x)))
    if sample_name == "mc":
        print("all vars: \n {}".format(list(df_filtered.keys())))
    return df_filtered

def apply_CCQE_presel(DF, sample_name, USECRT, DETSYS):
    '''
    Returns dataframe will all cuts applied
    '''
    query,_ = get_NUMU_sel(USECRT, opfilter=False)
    #exactly two tracks
    query += ' and reco_ntrack >= 2 and reco_ntrack < 3' #require two tracks (reco_ntrack==2.01 is actually 2)
    #one proton candidate must survive
    track_cuts = [
        ('trk_sce_start_x_v', '>', FVx[0]),
        ('trk_sce_start_x_v', '<', FVx[1]),
        ('trk_sce_start_y_v', '>', FVy[0]),
        ('trk_sce_start_y_v', '<', FVy[1]),
        ('trk_sce_start_z_v', '>', FVz[0]),
        ('trk_sce_start_z_v', '<', FVz[1]),
        ('trk_sce_end_x_v', '>', FVx[0]),
        ('trk_sce_end_x_v', '<', FVx[1]),
        ('trk_sce_end_y_v', '>', FVy[0]),
        ('trk_sce_end_y_v', '<', FVy[1]),
        ('trk_sce_end_z_v', '>', FVz[0]),
        ('trk_sce_end_z_v', '<', FVz[1]),
        ('trk_score_v', '>', 0.5),
        ('pfp_generation_v', '==', 2),
        ('trk_distance_v', '<', 4)
    ]
    df, mask = make_mask(DF, query, track_cuts)
    # need to protect the null values of certain variables
    if sample_name in ['mc','dirt','nue']: df['backtracked_pdg_v'] = df['backtracked_pdg_v'].apply(lambda x: [xx + 0.01 for xx in x])
    df_filtered = apply_mask(df.replace(0,-123456789), sample_name, mask, DETSYS).replace(-123456789,0) #0's don't survive...
    if sample_name in ['mc','dirt','nue']: df_filtered['backtracked_pdg_v'] = df_filtered['backtracked_pdg_v'].apply(lambda x: [int(round(xx)) for xx in x])
    if sample_name == "mc":
        print("all vars: \n {}".format(list(df_filtered.keys())))

    return df_filtered

def select_protons(DF, sample_name, USECRT, DETSYS):
    '''
    Returns dataframe will all cuts applied
    '''
    query = 'nslice == 1'
    track_cuts = [
        ('trk_llr_pid_score_v', '<', -0.2),
        ('trk_llr_pid_score_v', '>=', -1)
    ]
    df, mask = make_mask(DF, query, track_cuts)
    # need to protect the null values of certain variables
    if sample_name in ['mc','dirt','nue']: df['backtracked_pdg_v'] = df['backtracked_pdg_v'].apply(lambda x: [xx + 0.01 for xx in x])
    df_filtered = apply_mask(df.replace(0,-123456789), sample_name, mask, DETSYS).replace(-123456789,0) #0's don't survive...
    if sample_name in ['mc','dirt','nue']: df_filtered['backtracked_pdg_v'] = df_filtered['backtracked_pdg_v'].apply(lambda x: [int(round(xx)) for xx in x])

    return df_filtered

def select_muons(DF, sample_name, USECRT, DETSYS):
    '''
    Returns dataframe will all cuts applied
    '''
    query = 'nslice == 1'
    track_cuts = [
        ('trk_p_quality_v', '>', -0.5),
        ('trk_p_quality_v', '<', 0.5),
        ('trk_llr_pid_score_v', '>', 0.2),
        ('trk_score_v', '>', 0.8),
        ('trk_len_v', '>', 10),
    ]
    df, mask = make_mask(DF, query, track_cuts)
    # need to protect the null values of certain variables
    if sample_name in ['mc','dirt','nue']: df['backtracked_pdg_v'] = df['backtracked_pdg_v'].apply(lambda x: [xx + 0.01 for xx in x])
    df_filtered = apply_mask(df.replace(0,-123456789), sample_name, mask, DETSYS).replace(-123456789,0) #0's don't survive...
    if sample_name in ['mc','dirt','nue']: df_filtered['backtracked_pdg_v'] = df_filtered['backtracked_pdg_v'].apply(lambda x: [int(round(xx)) for xx in x])
    return df_filtered

def apply_fullsel_noopfilter(DF, sample_name, ISRUN3, DETSYS):
    query, track_cuts = get_NUMU_sel(ISRUN3, opfilter=False)
    df, mask = make_mask(DF, query, track_cuts)
    df_filtered = apply_mask(df.replace(0,-123456789), sample_name, mask, DETSYS).replace(-123456789,0)
    if sample_name == "mc":
        print("all vars: \n {}".format(list(df_filtered.keys())))

    return df_filtered

def apply_fullsel_noMCS(DF, sample_name, ISRUN3, DETSYS):
    '''
    Returns dataframe will all cuts applied
    '''
    query, track_cuts = get_NUMU_sel(ISRUN3)
    track_cuts.remove(('trk_p_quality_v', '>', -0.5))
    track_cuts.remove(('trk_p_quality_v', '<', 0.5))
    df, mask = make_mask(DF, query, track_cuts)
    df_filtered = apply_mask(df.replace(0,-123456789), sample_name, mask, DETSYS).replace(-123456789,0)
    if sample_name == "mc":
        print("all vars: \n {}".format(list(df_filtered.keys())))

    return df_filtered

def apply_contained(DF, sample_name, ISRUN3, DETSYS):
    '''
    Returns dataframe will all cuts applied
    '''
    query, track_cuts = get_NUMU_sel(ISRUN3)
    query = 'nslice == 1'
    track_cuts.remove(('trk_p_quality_v', '>', -0.5))
    track_cuts.remove(('trk_p_quality_v', '<', 0.5))
    track_cuts.remove(('trk_score_v', '>', 0.8))
    track_cuts.remove(('trk_llr_pid_score_v', '>', 0.2))
    track_cuts.remove(('trk_len_v', '>', 10))
    track_cuts.remove(('pfp_generation_v', '==', 2))
    track_cuts.remove(('trk_distance_v', '<', 4))
        #print("{} {} {}".format(cut[0]),cut[1],cut[2])
    df, mask = make_mask(DF, query, track_cuts)
    # need to protect the null values of certain variables
    if sample_name in ['mc','dirt','nue']:
        df['backtracked_pdg_v'] = df['backtracked_pdg_v'].apply(lambda x: [xx + 0.01 for xx in x])
        df['reco_nproton'] = df['reco_nproton'].apply(lambda x: x + 0.01)
        df['reco_ntrack'] = df['reco_ntrack'].apply(lambda x: x + 0.01)
    df_filtered = apply_mask(df.replace(0,-123456789), sample_name, mask, DETSYS).replace(-123456789,0) #0's don't survive...
    if sample_name in ['mc','dirt','nue']:
        df_filtered['backtracked_pdg_v'] = df_filtered['backtracked_pdg_v'].apply(lambda x: [int(round(xx)) for xx in x])
        df_filtered['reco_nproton'] = df_filtered['reco_nproton'].apply(lambda x: int(round(x)))
        df_filtered['reco_ntrack'] = df_filtered['reco_nproton'].apply(lambda x: int(round(x)))
    if sample_name == "mc":
        print("all vars: \n {}".format(list(df_filtered.keys())))
    return df_filtered

def apply_onetrackcut(DF, sample_name, ISRUN3, DETSYS):
    '''
    Returns dataframe will all cuts applied
    '''
    query = 'nslice == 1'
    track_cuts = [('trk_score_v', '>', 0.5)]
        #print("{} {} {}".format(cut[0]),cut[1],cut[2])
    df, mask = make_mask(DF, query, track_cuts)
    # need to protect the null values of certain variables
    if sample_name in ['mc','dirt','nue']: df['backtracked_pdg_v'] = df['backtracked_pdg_v'].apply(lambda x: [xx + 0.01 for xx in x])
    df_filtered = apply_mask(df.replace(0,-123456789), sample_name, mask, DETSYS).replace(-123456789,0) #0's don't survive...
    if sample_name in ['mc','dirt','nue']: df_filtered['backtracked_pdg_v'] = df_filtered['backtracked_pdg_v'].apply(lambda x: [xx - 0.01 for xx in x])
    if sample_name == "mc":
        print("all vars: \n {}".format(list(df_filtered.keys())))

    return df_filtered

#####################################################

def get_true_protonsel_index(DF, ISRUN3):
    df = DF.copy()
    mask = df['trk_len_v'].apply(lambda x: x==x) #all true mask
    mask *= df['backtracked_pdg_v'].apply(lambda x: eval("abs(x) == 2212"))
    #layer on each cut chipping down the all true mask
    df
    VAR = (df['trk_len_v']*mask).apply(lambda x: x[x != False]) #apply mask to dummy var
    VAR = VAR[VAR.apply(lambda x: len(x) > 0)] #clean up empty slices
    return VAR.index

def get_fullsel_invertMCS_index(DF, ISRUN3):
    '''
    Returns dataframe will all cuts applied
    '''
    FVx = [5,251]                      #[10,246]
    FVy = [-110,110]                   #[-105,105]
    FVz = [20,986]
    query, track_cuts = get_NUMU_sel(ISRUN3)
    track_cuts.remove(('trk_p_quality_v', '>', -0.5))
    track_cuts.remove(('trk_p_quality_v', '<', 0.5))
    track_cuts.append(('trk_p_quality_v', ['<','>'], [-0.5,0.5]))
    df = DF.copy().query(query)
    mask = df['trk_len_v'].apply(lambda x: x==x) #all true mask
    #layer on each cut chipping down the all true mask
    for (var,op,val) in track_cuts:
        if type(op) == list:
            #relate the two operations with an 'or'
            or_mask1 = df[var].apply(lambda x: eval("x{}{}".format(op[0],val[0])))#or condition 1
            or_mask2 = df[var].apply(lambda x: eval("x{}{}".format(op[1],val[1])))#or condition 2
            mask *= (or_mask1 + or_mask2) #just add the booleans for "or"
        else:
            mask *= df[var].apply(lambda x: eval("x{}{}".format(op,val))) #layer on each cut mask
    VAR = (df['trk_len_v']*mask).apply(lambda x: x[x != False]) #apply mask to dummy var
    VAR = VAR[VAR.apply(lambda x: len(x) > 0)] #clean up empty slices
    return VAR.index

def get_presel_index(DF):
    '''
    Returns dataframe will all cuts applied
    '''
    query, track_cuts = get_NUMU_sel()
    df = DF.copy().query(query)
    VAR = df['trk_len_v'] #any variable will do
    return VAR.index

################################################
# Useful for cleaning up parameter management in notebook
# use the tag to figure out what type of selection should be applied
# tag is also atteched to the filename so will create redundancy
def get_NUMU_sel(CRTVARS, opfilter=True, verbose=False):
    #updated for SCE
    #will swap numerical vals at end   systematic boundaries
    FVx = [5,251]                      #[10,246]
    FVy = [-110,110]                   #[-105,105]
    FVz = [20,986]

    # muon selection
    QUERY = 'nslice == 1'
    QUERY += ' and topological_score > 0.06'
    QUERY += ' and reco_nu_vtx_sce_x > FVx[0] and reco_nu_vtx_sce_x < FVx[1]'
    QUERY += ' and reco_nu_vtx_sce_y > FVy[0] and reco_nu_vtx_sce_y < FVy[1]'
    QUERY += ' and reco_nu_vtx_sce_z > FVz[0] and reco_nu_vtx_sce_z < FVz[1]'
    QUERY += ' and ( (reco_nu_vtx_sce_z < 675) or (reco_nu_vtx_sce_z > 775) )' #avoid dead wire region
    if CRTVARS:
        QUERY += ' and (crtveto!=1 or crthitpe < 100.) and (_closestNuCosmicDist > 5.)' # Note: CRTs weren't installed for until after Run 1
    if opfilter:
        QUERY += ' and (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20)'# or bnbdata == 1 or extdata == 1)'

    #okay, so the track level cuts are a bit more complicated
    #can't just use query. Gotta do each cut one at a time
    #(variable (must end in _v), operation, value)
    track_cuts = [
        ('trk_sce_start_x_v', '>', FVx[0]),
        ('trk_sce_start_x_v', '<', FVx[1]),
        ('trk_sce_start_y_v', '>', FVy[0]),
        ('trk_sce_start_y_v', '<', FVy[1]),
        ('trk_sce_start_z_v', '>', FVz[0]),
        ('trk_sce_start_z_v', '<', FVz[1]),
        ('trk_sce_end_x_v', '>', FVx[0]),
        ('trk_sce_end_x_v', '<', FVx[1]),
        ('trk_sce_end_y_v', '>', FVy[0]),
        ('trk_sce_end_y_v', '<', FVy[1]),
        ('trk_sce_end_z_v', '>', FVz[0]),
        ('trk_sce_end_z_v', '<', FVz[1]),
        ('trk_p_quality_v', '>', -0.5),
        ('trk_p_quality_v', '<', 0.5),
        ('trk_llr_pid_score_v', '>', 0.2),
        ('trk_score_v', '>', 0.8),
        ('trk_len_v', '>', 10),
        ('pfp_generation_v', '==', 2),
        ('trk_distance_v', '<', 4)
    ]

    QUERY = QUERY.replace('FVx[0]',str(FVx[0]))
    QUERY = QUERY.replace('FVy[0]',str(FVy[0]))
    QUERY = QUERY.replace('FVz[0]',str(FVz[0]))
    QUERY = QUERY.replace('FVx[1]',str(FVx[1]))
    QUERY = QUERY.replace('FVy[1]',str(FVy[1]))
    QUERY = QUERY.replace('FVz[1]',str(FVz[1]))

    if verbose:
        print ("QUERY:\n {}\n".format(QUERY))
        print("track_cuts:\n{}".format(track_cuts))

    return QUERY,track_cuts

#this function might be obsolete
def get_Cuts(tag, ISRUN3, verbose=True):
    QUERY, track_cuts = get_NUMU_sel(ISRUN3)
    muon_fid = track_cuts[:12] #cuts associated with fiducializing the muon
    #track_cuts = muon_fid
    if "noopfilter" in tag.lower():
        QUERY = 'nslice == 1'
        QUERY += ' and topological_score > 0.06'
        QUERY += ' and reco_nu_vtx_sce_x > 5 and reco_nu_vtx_sce_x < 251'
        QUERY += ' and reco_nu_vtx_sce_y > -110 and reco_nu_vtx_sce_y < 110'
        QUERY += ' and reco_nu_vtx_sce_z > 20 and reco_nu_vtx_sce_z < 986'
        QUERY += ' and ( (reco_nu_vtx_sce_z < 675) or (reco_nu_vtx_sce_z > 775) )' #avoid dead wire region
        if ISRUN3:
            QUERY += ' and (crtveto!=1 or crthitpe < 100.) and (_closestNuCosmicDist > 5.)' # Note: CRTs weren't installed for until after
    if "nocrtcuts" in tag.lower():
        QUERY = 'nslice == 1'
        QUERY += ' and topological_score > 0.06'
        QUERY += ' and reco_nu_vtx_sce_x > 5 and reco_nu_vtx_sce_x < 251'
        QUERY += ' and reco_nu_vtx_sce_y > -110 and reco_nu_vtx_sce_y < 110'
        QUERY += ' and reco_nu_vtx_sce_z > 20 and reco_nu_vtx_sce_z < 986'
        QUERY += ' and ( (reco_nu_vtx_sce_z < 675) or (reco_nu_vtx_sce_z > 775) )' #avoid dead wire region
        QUERY += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'
    if "presel" in tag.lower(): track_cuts = None
    if "samples" in tag.lower(): QUERY = 'nslice == 1'
    if "fid" in tag.lower(): track_cuts = muon_fig
    if "justsliceid" in tag.lower():
        QUERY,track_cuts = 'nslice == 1', None
        QUERY += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'
    if "withoutmcs" in tag.lower():
        track_cuts.remove(('trk_p_quality_v', '>', -0.5))
        track_cuts.remove(('trk_p_quality_v', '<', 0.5))
    if "fullsel_sample" in tag.lower(): QUERY,track_cuts = 'nslice == 1', None
    elif "fullsel_notopo_samples" in tag.lower() or "fullsel_nomcs_samples" in tag.lower() or "fullsel_invertmcs_samples" in tag.lower() or "fullsel_ccqe" in tag.lower() or "presel_contained" in tag.lower() or "ccqe_sample" in tag.lower() or "ccqe_contained_sample" in tag.lower() or "ccqe_tracktester" in tag.lower():
        QUERY,track_cuts = 'nslice == 1', None
    if "above105" in tag.lower():
        try:
            track_cuts.append(('reco_nu_e_range_v', '>', 1.05))
        except:
            track_cuts = [('reco_nu_e_range_v', '>', 1.05)]
    elif "below105" in tag.lower():
        try:
            track_cuts.append(('reco_nu_e_range_v', '<=', 1.05))
        except:
            track_cuts = [('reco_nu_e_range_v', '<=', 1.05)]
    if "nomcs" in tag.lower():
        try:
            track_cuts.remove(('trk_p_quality_v', '>', -0.5))
            track_cuts.remove(('trk_p_quality_v', '<', 0.5))
        except:
            print('not able to remove mcs cut')
    elif 'invertmcs' in tag.lower():
        try:
            track_cuts.remove(('trk_p_quality_v', '>', -0.5))
            track_cuts.remove(('trk_p_quality_v', '<', 0.5))
            track_cuts.append(('trk_p_quality_v', ['<','>'], [-0.5,0.5]))
        except:
            print('not able to remove mcs cut')
    if 'crtgt100' in tag.lower():
        QUERY,track_cuts = 'nslice == 1 and crthitpe > 100', None
        QUERY += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'
    elif 'crtlt100' in tag.lower():
        QUERY,track_cuts = 'nslice == 1 and crthitpe < 100', None
        QUERY += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'
    elif 'invertcrt' in tag.lower():
        QUERY,track_cuts = 'nslice == 1 and crthitpe > 100 and crtveto==1', None
    if "protonsel" in tag.lower():
        QUERY,track_cuts = 'nslice == 1 and reco_nproton >= 1', [('trk_llr_pid_score_v', '<', 0.5)]
    if "true2212" in tag.lower():
        QUERY,track_cuts = 'nslice == 1', None
        track_cuts = [('backtracked_pdg_v', '==', 2212)]
    if "firstfew" in tag.lower():
        QUERY,track_cuts = 'nu_e <= 0.55', None
    if "nopion" in tag.lower():
        QUERY += ' and npi0 == 0 and npion == 0'
    if "trueqe" in tag.lower():
        QUERY += ' and interaction == 0'
    #print for user-varification
    if verbose:
        print("QUERY: ", QUERY)
        if track_cuts:
            print("track_cuts: ")
            for cut in track_cuts:
                try: print("{} {} {}".format(cut[0],cut[1],cut[2]))
                except: print("{} {} {} or {} {} {}".format(cut[0],cut[1][0],cut[2][0],cut[0],cut[1][1],cut[2][1]))

    #if "opfilter" not in QUERY: QUERY += ' and ( (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20) or bnbdata == 1 or extdata == 1)'
    return QUERY, track_cuts

def get_plotter(tag, samples, pot, ISG1, USECRT):
    if "7e18" in tag.lower(): samples, weights, pot = update_data("data_7e18", samples, USECRT)
    elif "g1" in tag.lower(): samples, weights, pot = update_data("data_1e20", samples, USECRT)
    elif "fullsel_sample" in tag.lower():
        print("using prefiltered dataframe")
        weights, pot = get_scaling(USECRT, True)
    else:
        #neither is specified, go with safe bet
        if ISG1 and USECRT: weights, pot = get_scaling(USECRT, True)
        elif USECRT: weights, pot = get_scaling(USECRT, False)
        else: print("need a way to deal with this when ISRUN3 = {} <- that should be false".format(ISRUN3))

    if "allopen" in tag.lower():
        print('using all open data with Run 3 MC')
        weights = {
            "data": 1,
            "mc": 0.441,
            "nue": 0.441,#should be identical to numu weight, since parsed from same sample
            "ext": 1.89, #G1
            "dirt": 1.82,
        }
        pot = 5.9e20

    importlib.reload(plotter)
    my_plotter = plotter.Plotter(samples, weights, pot=pot)
    return my_plotter

def get_Detsys(tag,DETSYS,VARIABLE,RANGE,BINS):
    # For Systematic Errors
    #unpickle the most up-to-date systematic files, if available
    #must have pickled data of this format available
    PICKLEPATH = ls.main_path+ls.pickle_path+'NUMU-constr\\04072020\\'
    pickle_tag = tag
    pickle_name = "{}_{}_{}-{}-{}.pickle".format(VARIABLE,pickle_tag,RANGE[0],RANGE[1],BINS+1)
    if DETSYS:
        try:
            df_detsys_mc = pd.read_pickle(PICKLEPATH+pickle_name)
            detsys = {
                'mc' : df_detsys_mc['sum_noRecomb'] #'sum_noRecomb' or 'sum_nodEdX'
            }
        except:
            detsys = None
            print("could not load detsys pickle...")
            print("...tried {}".format(PICKLEPATH+pickle_name))
    else: detsys = None

    return detsys

def deJargoner(s):
    if "theta" in s and "trk_theta_v" not in s:
        s = s.replace("theta",r"True $\theta$")
    s = s.replace("trk_theta_v",r"Reco $\theta$")
    s = s.replace("trk_cos_theta_v",r"Reco cos($\theta$)")
    s = s.replace("reco_nu_e_range_v",r"Reco Range-Based $\nu$ Energy [GeV]")
    s = s.replace("nu_e",r"True $\nu$ Energy [GeV]")

    return s

def get_current_time(format="%m%d%H&M"):
    now = datetime.now()
    current_time = now.strftime(format)
    return current_time
