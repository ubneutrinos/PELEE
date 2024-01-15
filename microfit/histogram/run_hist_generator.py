import os
from typing import Dict, Optional, Union

from microfit import selections
from microfit.category_definitions import get_category_color, get_category_label
from microfit.statistics import chi_square
from microfit.histogram import Binning
from microfit.histogram import Histogram, HistogramGenerator
from microfit.parameters import ParameterSet
from microfit.fileio import from_json

import logging
import pandas as pd
import numpy as np

class RunHistGenerator:
    """Histogram generator for data and simulation runs."""

    def __init__(
        self,
        rundata_dict: Dict[str, pd.DataFrame],
        binning: Binning,
        selection: Optional[str] = None,
        preselection: Optional[str] = None,
        data_pot: Optional[float] = None,
        sideband_generator: Optional["RunHistGenerator"] = None,
        uncertainty_defaults: Optional[Dict[str, bool]] = None,
        parameters: Optional[ParameterSet] = None,
        detvar_data_path: Optional[str] = None,
        mc_hist_generator_cls: Optional[type] = None,
        showdata = True,
        **mc_hist_generator_kwargs,
    ) -> None:
        """Create a histogram generator for data and simulation runs.

        This combines data and MC appropriately for the given run. It assumes also that,
        if truth-filtered samples are present, that the corresponding event types have
        already been removed from the 'mc' dataframe. It also assumes that the background sets
        have been scaled to the same POT as the data.

        Parameters
        ----------
        rundata_dict : Dict[str, pd.DataFrame]
            Dictionary containing the dataframes for this run. The keys are the names of the
            datasets and the values are the dataframes. This must at least contain the keys
            "data", "mc", and "ext". This dictionary should be returned by the data_loader.
        binning : Binning
            Binning object containing the binning of the histogram.
        selection : str, optional
            Query to be applied to the dataframe before generating the histogram.
        preselection : str, optional
            Query to be applied to the dataframe before the selection is applied.
        data_pot : float, optional
            POT of the data sample. Required to reweight to a different target POT.
        sideband_generator : RunHistGenerator, optional
            Histogram generator for the sideband data. If provided, the sideband data will be
            used to constrain multisim uncertainties.
        uncertainty_defaults : Dict[str, bool], optional
            Dictionary containing default configuration of the uncertainty calculation, i.e.
            whether to use the sideband, include multisim errors etc.
        parameters : ParameterSet, optional
            Set of parameters for the analysis. These parameters will be passed through to the
            histogram generator for MC. The default HistogramGenerator ignores all parameters.
            The parameters are passed through by reference, so any changes to the parameters
            will be reflected in the histogram generator automatically.
        detvar_data_path : str, optional
            Path to the JSON file containing histograms of detector variations. If provided,
            the detector variations will be treated as unisim variations for additional uncertainty.
        mc_hist_generator_cls : type, optional
            Class to use for the MC histogram generator. If None, the default HistogramGenerator
            class is used.
        showdata : bool, optional 
            Whether to show data in the plot. If False, only MC is shown. Internally, this removes the
            dataframe for the real data entirely.
        **mc_hist_generator_kwargs
            Additional keyword arguments that are passed to the MC histogram generator on initialization.
        """
        self.data_pot = data_pot
        query = self.get_selection_query(selection, preselection)
        self.selection = selection
        self.preselection = preselection
        self.binning = binning
        self.logger = logging.getLogger(__name__)
        self.detvar_data = None
        if detvar_data_path is not None:
            # check that path exists
            if not os.path.exists(detvar_data_path):
                raise ValueError(f"Path {detvar_data_path} does not exist.")
            self.detvar_data = from_json(detvar_data_path)
            if not self.detvar_data["binning"] == self.binning:
                raise ValueError(
                    "Binning of detector variations does not match binning of main histogram."
                )
            if not self.detvar_data["selection"] == self.selection:
                raise ValueError(
                    "Selection of detector variations does not match selection of main histogram."
                )
            if not self.detvar_data["preselection"] == self.preselection:
                raise ValueError(
                    "Preselection of detector variations does not match preselection of main histogram."
                )

        # ensure that the necessary keys are present
        if "data" not in rundata_dict.keys():
            raise ValueError("data key is missing from rundata_dict (may be None).")
        if "mc" not in rundata_dict.keys():
            raise ValueError("mc key is missing from rundata_dict.")
        if "ext" not in rundata_dict.keys():
            raise ValueError("ext key is missing from rundata_dict (may be None).")

        for k, df in rundata_dict.items():
            if df is None:
                continue
            df["dataset_name"] = k
            df["dataset_name"] = df["dataset_name"].astype("category")
        mc_hist_generator_cls = (
            HistogramGenerator if mc_hist_generator_cls is None else mc_hist_generator_cls
        )
        # make one dataframe for all mc events
        df_mc = pd.concat([df for k, df in rundata_dict.items() if k not in ["data", "ext"]])
        df_ext = rundata_dict["ext"]
        df_data = rundata_dict["data"] if showdata else None
        if query is not None:
            # The Python engine is necessary because the queries tend to have too many inputs
            # for numexpr to handle.
            df_mc = df_mc.query(query, engine="python")
            if df_ext is not None:
                df_ext = df_ext.query(query, engine="python")
            if df_data is not None:
                df_data = df_data.query(query, engine="python")
            # the query has already been applied, so we can set it to None
            query = None

        self.parameters = parameters
        if self.parameters is None:
            self.parameters = ParameterSet([])  # empty parameter set
        else:
            assert isinstance(self.parameters, ParameterSet), "parameters must be a ParameterSet."
        self.mc_hist_generator = mc_hist_generator_cls(
            df_mc,
            binning,
            parameters=self.parameters,
            detvar_data=self.detvar_data,
            **mc_hist_generator_kwargs,
        )
        if df_ext is not None:
            self.ext_hist_generator = HistogramGenerator(df_ext, binning, enable_cache=False)
        else:
            self.ext_hist_generator = None
        if df_data is not None:
            self.data_hist_generator = HistogramGenerator(df_data, binning, enable_cache=False)
        else:
            self.data_hist_generator = None
        self.sideband_generator = sideband_generator
        self.uncertainty_defaults = dict() if uncertainty_defaults is None else uncertainty_defaults

    @classmethod
    def get_selection_query(cls, selection, preselection, extra_queries=None):
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
            return None
        presel_query = selections.preselection_categories[preselection]["query"]
        sel_query = selections.selection_categories[selection]["query"]

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

    def get_data_hist(
        self, type="data", add_error_floor=None, scale_to_pot=None, smooth_ext_histogram=False
    ) -> Union[None, Histogram]:
        """Get the histogram for the data (or EXT).

        Parameters
        ----------
        type : str, optional
            Type of data to return. Can be "data" or "ext".
        add_error_floor : bool, optional
            Add a minimum error of 1.4 to empty bins. This is motivated by a Bayesian
            prior of a unit step function as documented in
            https://microboone-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=32714&filename=Monte_Carlo_Uncertainties.pdf&version=1
            and is currently only applied to EXT events.
        scale_to_pot : float, optional
            POT to scale the data to. Only applicable to EXT data.

        Returns
        -------
        data_hist : Histogram
            Histogram of the data.
        """

        assert type in ["data", "ext"]
        add_error_floor = (
            self.uncertainty_defaults.get("add_ext_error_floor", False)
            if add_error_floor is None
            else add_error_floor
        )
        if smooth_ext_histogram:
            assert not add_error_floor, "Cannot smooth and add error floor at the same time."
        # The error floor is never added for data, overriding anything else
        if type == "data":
            add_error_floor = False
            smooth_ext_histogram = False
        scale_factor = 1.0
        if scale_to_pot is not None:
            if type == "data":
                raise ValueError("Cannot scale data to POT.")
            assert self.data_pot is not None, "Must provide data POT to scale EXT data."
            scale_factor = scale_to_pot / self.data_pot
        hist_generator = self.get_hist_generator(which=type)
        if hist_generator is None:
            return None
        data_hist = hist_generator.generate(use_kde_smoothing=smooth_ext_histogram)
        if add_error_floor:
            prior_errors = np.ones(data_hist.n_bins) * 1.4 ** 2
            prior_errors[data_hist.nominal_values > 0] = 0
            data_hist.add_covariance(np.diag(prior_errors))
        data_hist *= scale_factor
        data_hist.label = {"data": "Data", "ext": "EXT"}[type]
        data_hist.color = "k"
        data_hist.hatch = {"data": None, "ext": "///"}[type]
        return data_hist

    def get_mc_hists(
        self,
        category_column="dataset_name",
        include_multisim_errors=None,
        scale_to_pot=None,
        channel=None,
    ) -> Dict[str, Histogram]:
        """Get MC histograms that are split by event category.

        Parameters
        ----------
        category_column : str, optional
            Name of the column containing the event categories.
        include_multisim_errors : bool, optional
            Whether to include the systematic uncertainties from the multisim 'universes' for
            GENIE, flux and reintegration. Overrides the default setting.
        scale_to_pot : float, optional
            POT to scale the MC histograms to. If None, no scaling is performed.

        Returns
        -------
        mc_hists : dict
            Dictionary containing the histograms for each event category. Keys are the
            category names and values are the histograms.
        """

        if scale_to_pot is not None:
            assert self.data_pot is not None, "data_pot must be set to scale to a different POT."
        include_multisim_errors = (
            self.uncertainty_defaults.get("include_multisim_errors", False)
            if include_multisim_errors is None
            else include_multisim_errors
        )
        mc_hists = {}  # type: Dict[str, Histogram]
        other_categories = []
        for category in self.mc_hist_generator.dataframe[category_column].unique():
            extra_query = f"{category_column} == '{category}'"
            hist = self.get_mc_hist(
                include_multisim_errors=include_multisim_errors,
                extra_query=extra_query,
                scale_to_pot=scale_to_pot,
            )
            if category_column == "dataset_name":
                hist.label = str(category)
            else:
                hist.label = get_category_label(category_column, category)
                hist.color = get_category_color(category_column, category)
                if hist.label == "Other":
                    other_categories.append(category)
            mc_hists[category] = hist
        # before we return the histogram dict, we want to sum all categories together
        # that were labeled as "Other"
        if len(other_categories) > 0:
            summed_other_hist = sum([mc_hists.pop(cat) for cat in other_categories])
            assert isinstance(summed_other_hist, Histogram)
            mc_hists["Other"] = summed_other_hist
            mc_hists["Other"].label = "Other"
            mc_hists["Other"].color = "gray"
        return mc_hists

    def get_hist_generator(self, which):
        assert which in ["mc", "data", "ext"]
        hist_generator = {
            "mc": self.mc_hist_generator,
            "data": self.data_hist_generator,
            "ext": self.ext_hist_generator,
        }[which]
        return hist_generator

    def get_mc_hist(
        self,
        include_multisim_errors: Optional[bool] = None,
        extra_query: Optional[str] = None,
        scale_to_pot: Optional[float] = None,
        use_sideband: Optional[bool] = None,
        add_precomputed_detsys: bool = False,
    ) -> Histogram:
        """Produce a histogram from the MC dataframe.

        Parameters
        ----------
        include_multisim_errors : bool, optional
            Whether to include the systematic uncertainties from the multisim 'universes' for
            GENIE, flux and reintegration. Overrides the default setting.
        extra_query : str, optional
            Additional query to apply to the dataframe before generating the histogram.
        scale_to_pot : float, optional
            POT to scale the MC histograms to. If None, no scaling is performed.
        use_sideband : bool, optional
            If True, use the sideband MC and data to constrain multisim uncertainties.
            Overrides the default setting.
        add_precomputed_detsys : bool, optional
            Whether to add the precomputed detector systematics to the histogram covariance.
        """

        scale_factor = 1.0
        if scale_to_pot is not None:
            assert self.data_pot is not None, "data_pot must be set to scale to a different POT."
            scale_factor = scale_to_pot / self.data_pot
        include_multisim_errors = (
            self.uncertainty_defaults.get("include_multisim_errors", False)
            if include_multisim_errors is None
            else include_multisim_errors
        )
        use_sideband = (
            self.uncertainty_defaults.get("use_sideband", False)
            if use_sideband is None
            else use_sideband
        )
        hist_generator = self.get_hist_generator(which="mc")
        use_sideband = use_sideband and self.sideband_generator is not None
        if use_sideband:
            assert isinstance(self.sideband_generator, RunHistGenerator)
            sideband_generator = self.sideband_generator.get_hist_generator(which="mc")
            sideband_total_prediction = self.sideband_generator.get_total_prediction(
                include_multisim_errors=True
            )
            sideband_observed_hist = self.sideband_generator.get_data_hist(type="data")
            if sideband_observed_hist is None:
                raise RuntimeError(
                    "The sideband generator contains no data. Make sure to set `blinded=False` when loading the sideband data."
                )
        else:
            sideband_generator = None
            sideband_total_prediction = None
            sideband_observed_hist = None
        hist = hist_generator.generate(
            include_multisim_errors=include_multisim_errors,
            use_sideband=use_sideband,
            sideband_generator=sideband_generator,
            sideband_total_prediction=sideband_total_prediction,
            sideband_observed_hist=sideband_observed_hist,
            extra_query=extra_query,
            add_precomputed_detsys=add_precomputed_detsys,
        )
        hist.label = "MC"

        hist *= scale_factor
        return hist

    def get_total_prediction(
        self,
        include_multisim_errors=None,
        extra_query=None,
        scale_to_pot=None,
        use_sideband=None,
        add_ext_error_floor=None,
        add_precomputed_detsys=False,
        smooth_ext_histogram=False,
    ):
        """Get the total prediction from MC and EXT.

        Parameters
        ----------
        include_multisim_errors : bool, optional
            Whether to include the systematic uncertainties from the multisim 'universes' for
            GENIE, flux and reintegration. Overrides the default setting.
        extra_query : str, optional
            Additional query to apply to the dataframe before generating the histogram.
        scale_to_pot : float, optional
            POT to scale the MC histograms to. If None, no scaling is performed.
        use_sideband : bool, optional
            If True, use the sideband MC and data to constrain multisim uncertainties.
            Overrides the default setting.
        add_ext_error_floor : bool, optional
            Whether to add an error floor to the histogram in bins with zero entries.
        add_precomputed_detsys : bool, optional
            Whether to add the precomputed detector systematics to the histogram covariance.
        smooth_ext_histogram : bool, optional
            Whether to smooth the EXT histogram using a KDE. This is useful for low-statistics
            EXT samples.
        """
        mc_prediction = self.get_mc_hist(
            include_multisim_errors=include_multisim_errors,
            extra_query=extra_query,
            scale_to_pot=scale_to_pot,
            use_sideband=use_sideband,
            add_precomputed_detsys=add_precomputed_detsys,
        )
        if self.ext_hist_generator is not None:
            ext_prediction = self.get_data_hist(
                type="ext", scale_to_pot=scale_to_pot, add_error_floor=add_ext_error_floor, smooth_ext_histogram=smooth_ext_histogram
            )
            total_prediction = mc_prediction + ext_prediction
        else:
            total_prediction = mc_prediction
        return total_prediction

    def get_chi_square(self, **kwargs):
        """Get the chi square between the data and the total prediction.

        Returns
        -------
        chi_square : float
            Chi square between the data and the total prediction.
        """
        data_hist = self.get_data_hist(type="data")
        if data_hist is None:
            return np.nan
        total_prediction = self.get_total_prediction(**kwargs)
        chi_sq = chi_square(
            data_hist.nominal_values,
            total_prediction.nominal_values,
            total_prediction.covariance_matrix,
        )
        return chi_sq
