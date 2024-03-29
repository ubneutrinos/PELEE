from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Union
import warnings
from microfit.selections import find_common_selection, get_selection_query, get_selection_title

import numpy as np


@dataclass
class Binning:
    """Binning for a variable

    Attributes:
    -----------
    variable : str
        Name of the variable being binned
    bin_edges : np.ndarray
        Array of bin edges
    label : str
        Label of the binning. In a multi-dimensional binning, this should be
        a unique key.
    variable_tex : str, optional
        LaTeX representation of the variable (default is None) that can be used
        in plots.
    is_log : bool, optional
        Whether the binning is logarithmic or not (default is False)
    selection_query : str, optional
        Query to be applied to the dataframe before generating the histogram.
        This can be used to define a selection on the variable being binned.
    selection_key : str, optional
        Key of the selection in the selection dictionary. This is used to
        identify the selection in the MultiChannelBinning.
    preselection_key : str, optional
        Key of the preselection in the selection dictionary. This is used to
        identify the preselection in the MultiChannelBinning.
    selection_tex : str, optional
        LaTeX representation of the selection (default is None) that can be used
        in plots.
    selection_tex_short : str, optional
        Short LaTeX representation of the selection (default is None) that can be used
        in plots.
    """

    variable: str
    bin_edges: np.ndarray
    label: Optional[str] = None
    variable_tex: Optional[str] = None
    variable_tex_short: Optional[str] = None
    is_log: bool = False
    selection_query: Optional[str] = None
    selection_key: Optional[str] = None
    preselection_key: Optional[str] = None
    selection_tex: Optional[str] = None
    selection_tex_short: Optional[str] = None

    def is_compatible(self, other):
        """Check that two binnings are compatible.

        This function is a relaxed version of the __eq__ method. It checks that the
        binning is compatible with another binning, i.e. that the bin edges are the
        same, but it does not check other properties such as the selection. This is useful when
        adding and subtracting histograms that may originate from different selections.
        """

        # The properties that must match are the bin edges, is_log, and the variable.
        for field in fields(self):
            if field.name not in ["variable", "bin_edges", "is_log"]:
                continue
            attr_self = getattr(self, field.name)
            attr_other = getattr(other, field.name)
            if isinstance(attr_self, np.ndarray) and isinstance(attr_other, np.ndarray):
                if not np.array_equal(attr_self, attr_other):
                    return False
            else:
                if attr_self != attr_other:
                    return False
        return True

    def __eq__(self, other) -> bool:
        for field in fields(self):
            attr_self = getattr(self, field.name)
            attr_other = getattr(other, field.name)
            # The "label" property *does* have to match now, because we use it to
            # uniquely identify the channel in the MultiChannelBinning.
            # But the TeX strings don't have to match, because they are only used
            # for plotting.
            if field.name in [
                "variable_tex",
                "selection_tex",
                "selection_tex_short",
                "variable_tex_short",
            ]:
                # It really doesn't matter if the variable_tex is different, as it is
                # only used for plotting. So we can just skip it.
                continue
            if isinstance(attr_self, np.ndarray) and isinstance(attr_other, np.ndarray):
                if not np.array_equal(attr_self, attr_other):
                    return False
            else:
                if attr_self != attr_other:
                    return False
        return True

    def __post_init__(self):
        if isinstance(self.bin_edges, list):
            self.bin_edges = np.array(self.bin_edges)
        if self.variable_tex is None:
            self.variable_tex = self.variable
        assert self.variable_tex is not None

    def __len__(self):
        return self.n_bins

    def set_selection(
        self,
        selection=None,
        preselection=None,
        query=None,
        selection_tex=None,
        selection_tex_short=None,
    ):
        """Set the selection query of the binning given the preselection and the selection."""
        if selection is not None or preselection is not None:
            assert query is None, "Cannot set both query and selection/preselection"
            self.selection_query = get_selection_query(selection, preselection)
            self.selection_key = selection
            self.preselection_key = preselection
            self.selection_tex = selection_tex or get_selection_title(selection, preselection)
            self.selection_tex_short = selection_tex_short or get_selection_title(
                selection, preselection, short=True
            )
        elif query is not None:
            self.selection_query = query
            self.selection_key = None
            self.preselection_key = None
            self.selection_tex = selection_tex or query
            self.selection_tex_short = selection_tex_short or query
        else:
            raise ValueError("Must specify either selection/preselection or query")

    @classmethod
    def from_config(
        cls,
        variable: str,
        n_bins: Optional[int],
        limits: Optional[Tuple[float, float]],
        variable_tex: Optional[str] = None,
        is_log: bool = False,
        label: Optional[str] = None,
        bin_edges: Optional[Union[np.ndarray, List[float]]] = None,
        **kwargs,
    ):
        """Create a Binning object from a typical binning configuration

        Parameters:
        -----------
        variable : str
            Name of the variable being binned
        n_bins : int, optional
            Real of bins. If not provided, then bin_edges must be provided.
        limits : tuple, optional
            Tuple of lower and upper limits. If not provided, then bin_edges must be provided.
        variable_tex : str, optional
            Label for the binned variable. This will be used to label the x-axis in plots.
        is_log : bool, optional
            Whether the binning is logarithmic or not (default is False)
        label : str, optional
            Label of the binning. In a multi-dimensional binning, this should be
            a unique key.
        bin_edges : np.ndarray, optional
            Array of bin edges. If this is provided, the n_bins and limits are ignored.

        Returns:
        --------
        Binning
            A Binning object with the specified bounds
        """

        label = variable if label is None else label
        if bin_edges is not None:
            return cls(variable, np.array(bin_edges), label, variable_tex, is_log=is_log, **kwargs)
        else:
            assert n_bins is not None, "Must provide either n_bins or bin_edges"
            assert limits is not None, "Must provide either limits or bin_edges"
        if is_log:
            bin_edges = np.geomspace(*limits, n_bins + 1)
        else:
            bin_edges = np.linspace(*limits, n_bins + 1)
        
        return cls(variable, bin_edges, label, variable_tex, is_log=is_log, **kwargs)

    @property
    def n_bins(self):
        """Real of bins"""
        return len(self.bin_edges) - 1

    @property
    def bin_centers(self):
        """Array of bin centers"""
        if self.is_log:
            return np.sqrt(self.bin_edges[1:] * self.bin_edges[:-1])
        else:
            return (self.bin_edges[1:] + self.bin_edges[:-1]) / 2

    def to_dict(self):
        """Return a dictionary representation of the binning."""
        return self.__dict__

    @classmethod
    def from_dict(cls, state):
        """Create a Binning object from a dictionary representation of the binning."""
        return cls(**state)

    def copy(self):
        """Create a copy of the binning."""
        return Binning(**self.to_dict())

    @property
    def channels(self) -> List[str]:
        assert self.label is not None, "Binning must have a label"
        return [self.label]


@dataclass
class MultiChannelBinning:
    """Binning for multiple channels.

    This can be used to define multi-channel binnings of the same data as well as
    binnings for different selections. For every channel, we may have a different
    Binning that may have its own selection query. Using this binning definition,
    the HistogramGenerator can make multi-channel histograms that may include
    correlations between bins and channels.

    Note that this is different from a multi-dimensional binning. The binnings
    defined in this class are still 1-dimensional, but the HistogramGenerator
    can create an unrolled 1-dimensional histogram from all the channels that
    includes correlations between bins of different channels.
    """

    binnings: List[Binning]
    is_log: bool = False

    def __post_init__(self):
        assert [b.label is not None for b in self.binnings], "All binnings must have a label"
        self.ensure_unique_labels()

    def ensure_unique_labels(self):
        """Ensure that the `label` properties of the binnings in a MultiChannelBinning are unique."""
        used_labels = set()
        counter = 0
        for binning in self.binnings:
            if binning.label in used_labels:
                binning.label = f"{binning.label}_{counter}"
                counter += 1
            used_labels.add(binning.label)

    def reduce_selection(self):
        """Extract the common selection query from all channels and reduce the
        selection queries of all channels to only include the unique selections.

        Returns
        -------
        str
            A single selection query that can be applied to the dataframe before
            making the histogram.
        """
        selection_queries = [b.selection_query for b in self.binnings]
        selection_queries = [s if s is not None else "" for s in selection_queries]
        common_selection, unique_selections = find_common_selection(selection_queries)
        for binning, unique_selection in zip(self.binnings, unique_selections):
            binning.selection_query = unique_selection if len(unique_selection) > 0 else None
        return common_selection if len(common_selection) > 0 else None

    def to_dict(self):
        """Return a dictionary representation of the binning."""
        return {"binnings": [b.to_dict() for b in self.binnings], "is_log": self.is_log}

    @classmethod
    def from_dict(cls, state):
        """Create a MultiChannelBinning object from a dictionary representation of the binning."""
        state["binnings"] = [Binning.from_dict(b) for b in state["binnings"]]
        return cls(**state)

    @property
    def labels(self) -> List[str]:
        """Labels of all channels."""
        # Deprecation warning: We are renaming "labels" to "channels"
        warnings.warn(
            "The 'labels' property is deprecated. Use 'channels' instead.",
        )
        assert [b.label is not None for b in self.binnings], "All binnings must have a label"
        # The filter seems superfluous, but it is needed for the type checker to
        # understand that the labels are not None.
        return [b.label for b in self.binnings if b.label is not None]

    @property
    def channels(self) -> List[str]:
        """Labels of all channels."""
        assert [b.label is not None for b in self.binnings], "All binnings must have a label"
        # The filter seems superfluous, but it is needed for the type checker to
        # understand that the labels are not None.
        return [b.label for b in self.binnings if b.label is not None]

    @property
    def n_channels(self):
        """Real of channels."""
        return len(self.binnings)

    @property
    def n_bins(self):
        """Real of bins in all channels."""
        return sum([len(b) for b in self.binnings])

    @property
    def consecutive_bin_edges(self) -> List[np.ndarray]:
        """Get the bin edges of all channels.

        The bin edges of all channels are concatenated
        into a single array, which can be passed to np.histogramdd as follows:
        >>> bin_edges = consecutive_bin_edges
        >>> hist, _ = np.histogramdd(data, bins=bin_edges)

        Returns
        -------
        List[np.ndarray]
            List of arrays of bin edges of all channels.
        """
        return [b.bin_edges for b in self.binnings]

    @property
    def variables(self) -> List[str]:
        """Get the variables of all channels.

        Returns
        -------
        List[str]
            List of variables of all channels.
        """
        return [b.variable for b in self.binnings]

    @property
    def selection_queries(self) -> List[Union[str, None]]:
        """Get the selection queries of all channels.

        Returns
        -------
        List[str]
            List of selection queries of all channels.
        """
        return [b.selection_query for b in self.binnings]

    def delete_channel(self, key: Union[int, str]):
        """Delete a channel from the binning.

        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        """
        idx = self._idx_channel(key)
        del self.binnings[idx]

    def get_unrolled_binning(self) -> Binning:
        """Get an unrolled binning of all channels.

        The bins will just be labeled as 'Global bin number' and the
        variable will be 'none'. An unrolled binning cannot be used to
        create a histogram directly, but it can be used for plotting.
        """
        bin_edges = np.arange(self.n_bins + 1)
        return Binning("none", bin_edges, label="unrolled", variable_tex="Global bin number")

    def _idx_channel(self, key: Union[int, str]) -> int:
        """Get the index of a channel from the key.
        The key may be a numeric index or a string label.

        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        """
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            return self.channels.index(key)
        else:
            raise ValueError(f"Invalid key {key}. Must be an integer or a string.")

    def _channel_bin_idx(self, key: Union[int, str]) -> List[int]:
        """Get the indices of bins in the unrolled binning belonging to a channel.

        The bin counts from the channel can be extracted from the full bin counts
        using these indices as follows:
        >>> idx = _channel_bin_idx(key)
        >>> channel_bin_counts = bin_counts[idx]

        Parameters
        ----------
        key : int or str
            Index or label of the channel.
        bin_idx : int
            Index of the bin in the channel.

        Returns
        -------
        List[int]
            List of indices of bins belonging to the channel.
        """
        idx = self._idx_channel(key)
        binning = self.binnings[idx]
        start_idx = sum([len(b) for b in self.binnings[:idx]])
        return list(range(start_idx, start_idx + len(binning)))

    def __getitem__(self, key: Union[int, str]) -> Binning:
        """Get the binning of a given channel.

        Parameters
        ----------
        key : int or str
            Index or label of the channel.

        Returns
        -------
        Binning
            Binning of the channel.
        """
        return self.binnings[self._idx_channel(key)]

    def __iter__(self):
        for binning in self.binnings:
            yield binning

    def __len__(self):
        return self.n_channels

    def copy(self):
        """Create a copy of the binning."""
        return MultiChannelBinning.from_dict(self.to_dict())

    def _roll_list(self, lst, shift):
        return lst[-shift:] + lst[:-shift]

    def roll_channels(self, shift: int):
        """Roll the channels by a given number of steps.

        Parameters
        ----------
        shift : int
            Number of steps to roll the channels.
        """
        self.binnings = self._roll_list(self.binnings, shift)
        self.ensure_unique_labels()

    def roll_to_first(self, label: str):
        """Roll the channels such that the channel with the given label is first.

        Parameters
        ----------
        label : str
            Label of the channel to be rolled to first.
        """
        idx = self.channels.index(label)
        self.roll_channels(-idx)

    @classmethod
    def join(cls, *args):
        """Join multiple MultiChannelBinning or Binning objects into a single MultiChannelBinning.

        Parameters
        ----------
        *args : MultiChannelBinning or Binning
            Multiple MultiChannelBinning or Binning objects to be joined.

        Returns
        -------
        MultiChannelBinning
            A single MultiChannelBinning object containing all the binnings.
        """
        binnings = []
        for mcb in args:
            if not isinstance(mcb, MultiChannelBinning):
                binnings.append(mcb)
            else:
                binnings.extend(mcb.binnings)
        return cls(binnings)

    def is_compatible(self, other):
        """Check that two MultiChannelBinning objects are compatible.

        This function is a relaxed version of the __eq__ method. It checks that the
        binnings are compatible with another MultiChannelBinning, i.e. that the bin edges are the
        same, but it does not check other properties such as the selection. This is useful when
        adding and subtracting histograms that may originate from different selections.
        """
        for b1, b2 in zip(self.binnings, other.binnings):
            if not b1.is_compatible(b2):
                return False
        return True
