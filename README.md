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

The following scripts make the technote plots for each sub-analysis. These include the histograms of the signal and sideband channels,
the covariance matrices (total and split by error source), tables with error budgets and constraint updates in TeX format,
chi-square distributions, two-hypothesis tests, and the FC corrected Asimov sensitivity scan. These scripts
can take up to 2-3 hours to run if they are run from scratch without already existing cached dataframes.

### Technote plots for the new signal analysis

All technote plots for the analysis with the new signal model can be produced by running 
```
python new_signal_ana_technote_plots.py
```
while inside the `scripts` directory. Make sure to make a directory called `full_ana_remerged_crt_output` before running the script.
The script is fully automated and loads all necessary data and produces the detector variations it needs. 
All plots that are produced will be placed into `full_ana_remerged_crt_output`.

### Technote plots for the old signal analysis

All technote plots for the analysis with the old signal model can be produced by running
```
python old_signal_ana_technote_plots.py
```
while inside the `scripts` directory. Make sure to make a directory called `old_model_ana_remerged_crt_output` before running the script.
The script is fully automated and loads all necessary data and produces the detector variations it needs. 
All plots that are produced will be placed into `old_model_ana_remerged_crt_output`.