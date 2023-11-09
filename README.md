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
```

## Unit Tests
Run unit tests to make sure core functionality is working as intended
```
python -m unittest microfit.parameters microfit.statistics microfit.test_histogram
```

## Tutorial
Run the `MicroFit Tutorial.ipynb` notebook to familiarize yourself with the MicroFit framework and how to run a basic two-hypothesis test.
