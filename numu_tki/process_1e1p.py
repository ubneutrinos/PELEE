import numpy as np
from numu_tki import tki_calculators

# Function to calculate the reco TKI variables using CT/SG's code and also using my code to cross check. 
# Author: M Moudgalya

################################################################################

def calculate_trk_pi(row, vec_name):
    """Function to calculate the proton momentum component variables, where i=x,y,z.
        Parameters:
        -----------
        vec_name : str
            The direction variable. Can be either 'trk_dir_x_v', 'trk_dir_y_v' or 'trk_dir_z_v'.

        Returns:
        --------
        trk_pi : float
            The momentum component of the specified direction.
    """
    trk_id = row['trk_id'] -1
    trk_vec = row[vec_name]
    trk_p = row['mod_trk_p']
    
    # Access the 'trk_id'-th element of the 'trk_vec' vector and multiply it by 'trk_p'
    # But first check if trk_id is within the valid range of 'trk_vec'
    if 0 <= trk_id < len(trk_vec):
        trk_pi = trk_p * trk_vec[trk_id]
    else:
        trk_pi = np.nan  # Set to NaN for out-of-bounds indices
    
    return trk_pi

################################################################################

def calculate_pt(row, particle):
    """Function to calculate the transverse projection vector of the electron and proton momenta. 
        This will be used in calculating the TKI variable delta_pt.
        Parameters:
        -----------
        particle : str
            The particle for which the transverse momentum needs to be calculated. Can be either 'electron' or 'proton'.

        Returns:
        --------
        pt: numpy array of floats
            The transverse projection of the particle momentum.
    """
    
    if particle == 'electron':
        px = row['shr_px']
        py = row['shr_py']
        
    elif particle == 'proton':
        px = row['trk_px']
        py = row['trk_py']
        
    pt = np.array([px, py])
    return pt

################################################################################

def calculate_delta_alpha(row):
    """Function to calculate the TKI variable delta_alpha."""
    return np.degrees(np.arccos(np.inner(-1 * row['shr_pt'], row['delta_pt']) / (np.linalg.norm(row['shr_pt']) * row['mod_delta_pt'])))

################################################################################

def process_1e1p_tki(up,df):

    # Load the extra branches needed 
    df["trk_dir_x_v"] = up.array("trk_dir_x_v")
    df["trk_dir_y_v"] = up.array("trk_dir_y_v")
    df["trk_dir_z_v"] = up.array("trk_dir_z_v")
    df['trk_energy'] = up.array('trk_energy')
    df['trk_id'] = up.array('trk_id')
    df['shr_energy_cali'] = up.array('shr_energy_cali')
    df['shr_energy'] = up.array('shr_energy')
    df['shr_px'] = up.array('shr_px')
    df['shr_py'] = up.array('shr_py')
    df['shr_pz'] = up.array('shr_pz')

    # Calculating the modulus of the proton momentum using trk_energy as the kinetic energy
    m_p = 0.939 # GeV (natural units c=1)
    gamma = (df['trk_energy'] / m_p) + 1
    beta = np.sqrt(1 - (1/gamma**2))
    df['mod_trk_p'] = m_p * gamma * beta # c=1

    df['trk_px'] = df.apply(calculate_trk_pi, axis=1, vec_name='trk_dir_x_v')
    df['trk_py'] = df.apply(calculate_trk_pi, axis=1, vec_name='trk_dir_y_v')
    df['trk_pz'] = df.apply(calculate_trk_pi, axis=1, vec_name='trk_dir_z_v')

    # Making corrections to the electron energy and momentum variables
    df['shr_px'] = df['shr_px'] * df['shr_energy_cali'] / df['shr_energy'] / 0.83
    df['shr_py'] = df['shr_py'] * df['shr_energy_cali'] / df['shr_energy'] / 0.83
    df['shr_pz'] = df['shr_pz'] * df['shr_energy_cali'] / df['shr_energy'] / 0.83
        
    df['shr_energy_cali'] = df['shr_energy_cali'] * 1/0.83
        
    # Calculating the modulus of the electron momentum
    df['mod_shr_p'] = np.sqrt((df['shr_px'])**2 + (df['shr_py'])**2 + (df['shr_pz'])**2)

    # Calculating the transverse momentum projection of the electron and proton
    df['shr_pt'] = df.apply(calculate_pt, axis=1, particle='electron')
    df['trk_pt'] = df.apply(calculate_pt, axis=1, particle='proton')
        
    # Calculating TKI delta_pt
    df['delta_pt'] = df['shr_pt'] + df['trk_pt']
    df['mod_delta_pt'] = df['delta_pt'].apply(lambda vec: np.linalg.norm(vec))
        
    # Calculating TKI delta_alpha
    df['delta_alpha'] = df.apply(calculate_delta_alpha, axis=1)


    # Calculate the TKI variables for 1e1p using CT/SG's code
    print("Calc reco TKI variables for leading proton only")

    df["RecoDeltaPT"] = df.apply(lambda x: (tki_calculators.delta_pT(x["shr_px"],x["shr_py"],x["shr_pz"],x["trk_px"],x["trk_py"],x["trk_pz"])),axis=1)
    #df["RecoDeltaPhiT"] = df.apply(lambda x: (tki_calculators.delta_phiT(x["shr_px"],x["shr_py"],x["shr_pz"],x["trk_px"],x["trk_py"],x["trk_pz"])),axis=1)
    df["RecoDeltaAlphaT"] = df.apply(lambda x: (tki_calculators.delta_alphaT(x["shr_px"],x["shr_py"],x["shr_pz"],x["trk_px"],x["trk_py"],x["trk_pz"])),axis=1)
    df['RecoDeltaAlphaT'] = np.degrees(df['RecoDeltaAlphaT'])

    # Drop temporary data from dataframes
    df.drop(["trk_dir_x_v", "trk_dir_y_v", "trk_dir_z_v"], inplace=True, axis=1)

    return df