import numpy as np
from scipy.stats import binned_statistic

def Eff_vectorial(df,weights,var,num_query,den_query,bin_edges,absval=False):
    #print acceptance
    if len(weights.shape) == 1:
        weights = weights[np.newaxis, ...]
    
    den_mask = df.eval(den_query)[np.newaxis, ...]
    num_mask = df.eval(den_query + " and " + num_query)[np.newaxis, ...]
    
    den_counts = binned_statistic(df[var], weights*den_mask, statistic='sum', bins=bin_edges, range=[bin_edges[0], bin_edges[-1]])
    num_counts = binned_statistic(df[var], weights*num_mask, statistic='sum', bins=bin_edges, range=[bin_edges[0], bin_edges[-1]])

    return num_counts[0]/den_counts[0]

def covMatrix(df, function, var_weight_sys, var_weight_cv='weightSplineTimesTune', n_max_universes=None, **kwargs):
    '''function must be in the form f(df, weights, ...)'''

    weights_cv = df[var_weight_cv] # shape = (len(df),)
    print(f"weights_cv shape = {weights_cv.shape}")
    cv = function(df, weights_cv, **kwargs) # shape = (1, n_bin)
    print(f"cv shape = {cv.shape}")

    weights_sys = np.stack(df[var_weight_sys].values, axis=1)
    if n_max_universes is not None:
        weights_sys = weights_sys[:n_max_universes]
    print(f"weights_sys shape = {weights_sys.shape}")
    sys_variations = function(df, weights_sys, **kwargs) # shape = (n_max_universes, n_bin)
    print(f"sys_variations shape = {sys_variations.shape}")
    delta_sys = (sys_variations - cv)[..., np.newaxis] # shape = (n_max_universes, n_bin, 1)
    print(f"delta_sys shape = {delta_sys.shape}")

    cov = (delta_sys.transpose(0, 2, 1) * delta_sys).mean(axis=0)
    print(f"cov shape = {cov.shape}")
    return cov

if __name__ == "__main__":
    cov = covMatrix(pi0s, 
            Eff_vectorial,  
            "weightsGenie", 
            var_weight_cv='weightSplineTimesTune', 
            var=VART,
            num_query=QUERY,
            den_query=ACCEPTANCE+' and '+q,
            bin_edges=bin_edges)

    Eff_vectorial(pi0s,pi0s["weightSplineTimesTune"],VART,QUERY,ACCEPTANCE+' and '+q,bin_edges)

    np.sqrt(np.diag(cov))