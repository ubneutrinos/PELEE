import numpy as np
from scipy.stats import binned_statistic

def Eff_vectorial(df,weights,var,num_query,den_query,bin_edges,with_uncertainties=False):
    if len(weights.shape) == 1:
        if type(weights) != np.ndarray:
            weights = weights.values
        weights = weights[np.newaxis, ...]
    
    den_mask = df.eval(den_query).values[np.newaxis, ...]
    num_mask = df.eval(den_query + " and " + num_query).values[np.newaxis, ...]
    
    den_counts = binned_statistic(df[var], weights*den_mask, statistic='sum', bins=bin_edges, range=[bin_edges[0], bin_edges[-1]])
    num_counts = binned_statistic(df[var], weights*num_mask, statistic='sum', bins=bin_edges, range=[bin_edges[0], bin_edges[-1]])

    eff = num_counts[0]/den_counts[0]
    if not with_uncertainties:
        return eff
    else:
        eff_err = np.sqrt(eff * (1 - eff) / den_counts[0])
        return eff, eff_err

def Eff_ccnc(df,weights,var,num_query,den_query,bin_edges, num_dem=False):
    if len(weights.shape) == 1:
        if type(weights) != np.ndarray:
            weights = weights.values
        weights = weights[np.newaxis, ...]
    out = []
    for pi0_type in ['ccnc==0','ccnc==1']:
        den_mask = df.eval(" and ".join([den_query, pi0_type])).values[np.newaxis, ...]
        num_mask = df.eval(" and ".join([den_query, num_query, pi0_type])).values[np.newaxis, ...]

        den_counts = binned_statistic(df[var], weights*den_mask, statistic='sum', bins=bin_edges, range=[bin_edges[0], bin_edges[-1]])
        num_counts = binned_statistic(df[var], weights*num_mask, statistic='sum', bins=bin_edges, range=[bin_edges[0], bin_edges[-1]])
        if num_dem:
            out.append(num_counts[0])
            out.append(den_counts[0])
        out.append(num_counts[0]/den_counts[0])
    out.append(out[5]/out[2])
    return np.concatenate(out, axis=1)

def sampleSystematics(df, function, var_weight_sys, var_weight_cv='weightSplineTimesTune', n_max_universes=None, **kwargs):
    '''function must be in the form f(df, weights, ...), and kwargs are passed to f'''

    weights_cv = df[var_weight_cv] # shape = (len(df),)
    print(f"weights_cv shape = {weights_cv.shape}")
    cv = function(df, weights_cv, **kwargs) # shape = (1, n_bin)
    print(f"cv shape = {cv.shape}")
    
    if var_weight_sys == "weightsGenie":
        weights_sys = np.stack(df["weightSpline"] * df[var_weight_sys].values, axis=1)
    else:
        weights_sys = np.stack(df[var_weight_sys].values, axis=1)
    weights_sys = weights_sys.astype(float)
    weights_sys /= 1000.
    
    if n_max_universes is not None:
        weights_sys = weights_sys[:n_max_universes]
    print(f"weights_sys shape = {weights_sys.shape}")
    sys_variations = function(df, weights_sys, **kwargs) # shape = (n_max_universes, n_bin)
    print(f"sys_variations shape = {sys_variations.shape}")
    return cv, sys_variations

def covMatrix(df, function, var_weight_sys, var_weight_cv='weightSplineTimesTune', n_max_universes=None, **kwargs):
    '''function must be in the form f(df, weights, ...), and kwargs are passed to f'''
    cv, sys_variations = sampleSystematics(df, function, var_weight_sys, var_weight_cv, n_max_universes=None, **kwargs)
    
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