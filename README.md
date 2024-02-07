# PELEE
## Prerequisites
### Conda Environment
(1) Install free conda from miniforge: https://github.com/conda-forge/miniforge

(2) follow the setup below:

```
conda create -n python3LEE python=3.7
conda activate python3LEE
conda install scipy
conda install scikit-learn
conda install jupyter
conda install pandas==1.0.5
conda install matplotlib
conda install -c conda-forge uproot==3.11.6
conda install dask-core
conda install -c conda-forge xgboost==0.90
conda install -c conda-forge shap
conda install -c conda-forge uncertainties
pip install unitpy
```
(the xgboost version needs to be specified to be compatible with the stored BDTs)

(3) To activate this local python setup type from the terminal:

```
conda activate python3LEE
```

### Local Settings
 You will need a file localSettings.py in the root directory of this repository that defines the root directory where all the data lives that contains:
```
main_path = "/Users/cerati/Notebooks/PELEE/"
ntuple_path = "/Users/cerati/Notebooks/PELEE/root_files/1013/"
pickle_path = "/Users/cerati/Notebooks/PELEE/pickles/"
plots_path = "/Users/cerati/Notebooks/PELEE/plots/"
dataframe_cache_path = "/Users/cerati/Notebooks/PELEE/cached_dataframes/
```
The path `dataframe_cache_path` is used to store loaded dataframes to disk for faster loading in the future when `enable_cache=True` is set for the data loading functions. 

## Unit Tests
Run unit tests to make sure core functionality is working as intended.
These tests are also automatically run on GitHub for every PR to `master`.
Since all tests are defined in source files that match the `test_*.py` pattern, they are automatically discovered and run by `unittest`.
```
python -m unittest discover
```

## Tutorial
Run the `MicroFit Tutorial.ipynb` notebook to familiarize yourself with the MicroFit framework and how to run a basic two-hypothesis test.

## Technote plots

### Plots of the signal and sideband spectra with correlations
Run the following to make the spectra plots, plots of the covariance matrices, and print a table of the fractional error contributions:
```
python scripts/plot_analysis_histograms_correlations.py --configuration /nashome/a/atrettin/PELEE/config_files/first_round_analysis_runs_1-5.toml --output-dir ana_output_runs_1-5 --print-tables
```

### First sensitivities with all runs
To run the two-hypothesis test: 
```
python scripts/two_hypothesis_test.py run_analysis --configuration config_files/first_round_
analysis_runs_1-5.toml --output-dir ana_output_runs_1-5 --sensitivity-only
```
To make the plots:
```
python scripts/two_hypothesis_test.py plot_results --configuration config_files/first_round_
analysis_runs_1-5.toml --output-dir ana_output_runs_1-5 --sensitivity-only
```
This should run the two-hypothesis test and estimate the median sensitivity.

To run the sensitivity scan for the signal strength fit: 
```
python scripts/fc_scan_signal_sensitivity.py run_analysis --configuration conf
ig_files/first_round_analysis_runs_1-5.toml --output-dir ana_output_runs_1-5
```
To make the plot
```
python scripts/fc_scan_signal_sensitivity.py plot_results --configuration conf
ig_files/first_round_analysis_runs_1-5.toml --output-dir ana_output_runs_1-5 --title "Runs 1-5, inclusive $\nu_\mu$ constraints"
```
