# %%
# %load_ext autoreload
# %autoreload 2
# %%
import sys
import os

from matplotlib import pyplot as plt
import numpy as np
import toml
# %%
sys.path.append("../")
import logging
from microfit.fileio import from_json, to_json
from microfit.analysis import MultibandAnalysis
from microfit.parameters import Parameter, ParameterSet
from unblinding_functions import (
    plot_chi2_distribution,
)
from microfit.histogram import MultiChannelHistogram
from microfit.category_definitions import get_category_label
logging.basicConfig(level=logging.INFO)

# %%
config_file = "../config_files/old_model_ana_with_detvars.toml"
output_dir = "../old_model_ana_output_with_3a/"

analysis = MultibandAnalysis.from_toml(
    config_file,
    logging_level=logging.DEBUG,
    overwrite_cached_df=False,
    overwrite_cached_df_detvars=False,
    output_dir=output_dir,
)
# %%
h0_params = ParameterSet([Parameter("signal_strength", 0.0)])
h1_params = ParameterSet([Parameter("signal_strength", 1.0)])
analysis.set_parameters(h0_params)
data_hist = analysis.get_data_hist()
ext_hist = analysis.get_data_hist(type="ext")
assert ext_hist is not None
assert data_hist is not None
total_mc_hist_h0_constrained = analysis.get_mc_hist(
    include_multisim_errors=True,
    use_sideband=True,
    add_precomputed_detsys=True
)
total_pred_hist_h0_constrained = total_mc_hist_h0_constrained + ext_hist
total_mc_hist_unconstrained = analysis.get_mc_hist(
    include_multisim_errors=True,
    use_sideband=False,
    add_precomputed_detsys=True
)
total_pred_hist_h0_unconstrained = total_mc_hist_unconstrained + ext_hist

analysis.set_parameters(h1_params)

mc_hists = analysis.get_mc_hists(category_column="paper_category")

total_mc_hist_h1 = analysis.get_mc_hist(
    include_multisim_errors=True,
    use_sideband=True,
    add_precomputed_detsys=True,
)
total_pred_hist_h1 = total_mc_hist_h1 + ext_hist
# %%
print("Total H0 pred (constrained):")
assert isinstance(total_pred_hist_h0_constrained, MultiChannelHistogram)
for channel_hist in total_pred_hist_h0_constrained:
    print(f"{channel_hist.binning.label}: {channel_hist.sum():.3f} +/- {channel_hist.sum_std():.3f}")
print("Total H0 pred (unconstrained):")
assert isinstance(total_pred_hist_h0_unconstrained, MultiChannelHistogram)
for channel_hist in total_pred_hist_h0_unconstrained:
    print(f"{channel_hist.binning.label}: {channel_hist.sum():.3f} +/- {channel_hist.sum_std():.3f}")
print("Split by category:")
for category, category_mc_hist in mc_hists.items():
    category_label = get_category_label(category_column="paper_category", category=category)
    for channel_hist in category_mc_hist:
        print(f"{category_label} {channel_hist.binning.label}: {channel_hist.sum():.3f}")
# treat EXT also as a category
for channel_hist in ext_hist:
    print(f"EXT {channel_hist.binning.label}: {channel_hist.sum():.3f}")
print("Total H1 pred:")
assert isinstance(total_pred_hist_h1, MultiChannelHistogram)
for channel_hist in total_pred_hist_h1:
    print(f"{channel_hist.binning.label}: {channel_hist.sum():.3f} +/- {channel_hist.sum_std():.3f}")
print("Data counts:")
assert isinstance(data_hist, MultiChannelHistogram)
for channel_hist in data_hist:
    # No errors because this is data
    print(f"{channel_hist.binning.label}: {channel_hist.sum():.0f}")
# %%
# Turn the printout into a table, with one column for each channel
# and one row for each of the following:
# - Data counts
# - Total pred. H0 counts (constrained), with uncertainties
# - Total pred. H0 counts (unconstrained), with uncertainties
# - Total pred. H1 counts, with uncertainties
# - Total pred. H0 counts split by category (no uncertainties)
# - Data counts (no uncertainties)

# First, get the channel labels
channel_labels = [channel_hist.binning.label for channel_hist in data_hist]
print("&" + "&".join(channel_labels) + r"\\")  # type: ignore
# Data counts
print("Data counts", end="")
for channel_hist in data_hist:
    print(f"&{channel_hist.sum():.0f}", end="")
print(r"\\")
# MC H0 counts (constrained)
print("Total $H_0$ pred. (constrained)", end="")
for channel_hist in total_pred_hist_h0_constrained:
    print(f"&{channel_hist.sum():.1f} $\\pm$ {channel_hist.sum_std():.1f}", end="")
print(r"\\")
# MC H0 counts (unconstrained)
print("Total $H_0$ pred. (unconstrained)", end="")
for channel_hist in total_pred_hist_h0_unconstrained:
    print(f"&{channel_hist.sum():.1f} $\\pm$ {channel_hist.sum_std():.1f}", end="")
print(r"\\")
# MC H1 counts
print("Total $H_1$ pred.", end="")
for channel_hist in total_pred_hist_h1:
    print(f"&{channel_hist.sum():.1f} $\\pm$ {channel_hist.sum_std():.1f}", end="")
print(r"\\")
# Predicted H0 counts split by category.
print("$H_0$ pred. split by category&&\\\\")

# treat EXT also as a category
print("EXT", end="")
for channel_hist in ext_hist:
    print(f"&{channel_hist.sum():.1f}", end="")
print(r"\\")
for category, category_mc_hist in mc_hists.items():
    category_label = get_category_label(category_column="paper_category", category=category)
    print(category_label, end="")
    for channel_hist in category_mc_hist:
        print(f"&{channel_hist.sum():.1f}", end="")
    print(r"\\")

# %%
# Now we slighly hack the configuration to use the signal model 2
ana_config = toml.load(config_file)
for gen_config in ana_config["generator"]:
    if gen_config["load_runs"].get("load_lee", False):
        gen_config["load_runs"]["use_new_signal_model"] = True

analysis = MultibandAnalysis(
    configuration=ana_config,
    output_dir=output_dir,
    logging_level=logging.DEBUG,
    overwrite_cached_df=False,
    overwrite_cached_df_detvars=False,
)
# %%
analysis.plot_signals()
# %%
analysis.set_parameters(h1_params)
total_mc_hist_h1_new_model = analysis.get_mc_hist(
    include_multisim_errors=True,
    use_sideband=True,
    add_precomputed_detsys=True,
)
total_pred_hist_h1_new_model = total_mc_hist_h1_new_model + ext_hist
mc_hists_new_model = analysis.get_mc_hists(category_column="paper_category")
# %%
print("Total H1 pred (new model, constrained):")
assert isinstance(total_pred_hist_h1_new_model, MultiChannelHistogram)
for channel_hist in total_pred_hist_h1_new_model:
    print(f"{channel_hist.binning.label}: {channel_hist.sum():.3f} +/- {channel_hist.sum_std():.3f}")
print("Split by category:")
for category, category_mc_hist in mc_hists_new_model.items():
    category_label = get_category_label(category_column="paper_category", category=category)
    for channel_hist in category_mc_hist:
        print(f"{category_label} {channel_hist.binning.label}: {channel_hist.sum():.3f}")
# %%
