"""Definitions of event categories and their associated labels and colors."""


def get_category_label(category_column, category):
    """Get the appropriate label for a given category depending on which category column was used."""

    if category_column == "category":
        return category_labels.get(category, "Other")
    elif category_column == "paper_category":
        return paper_labels.get(category, "Other")
    elif category_column == "paper_category_xsec":
        return paper_labels_xsec.get(category, "Other")
    elif category_column == "category_1e1p":
        return category_labels_1e1p.get(category, "Other")
    else:
        raise ValueError("Invalid category column: {}".format(category_column))

def get_categories(category_column):
    """Get the list of categories for a given category column."""

    if category_column == "category":
        return list(category_labels.keys())
    elif category_column == "paper_category":
        return list(paper_labels.keys())
    elif category_column == "paper_category_xsec":
        return list(paper_labels_xsec.keys())
    elif category_column == "category_1e1p":
        return list(category_labels_1e1p.keys())
    else:
        raise ValueError("Invalid category column: {}".format(category_column))

def get_category_color(category_column, category):
    """Get the appropriate color for a given category depending on which category column was used."""

    # always use the same colors for now
    return category_colors[category]

# Whenever you define categories, be sure to define all the categories that can appear in the data.
# Missing categories can lead to events not showing up in histograms.

paper_labels_numu = {
    11: r"$\nu_e$ CC",
    111: r"MiniBooNE LEE",
    2: r"$\nu_{\mu}$ CC",
    22: r"$\nu_{\mu}$ CC 0p",
    23: r"$\nu_{\mu}$ CC 1p",
    24: r"$\nu_{\mu}$ CC 2p",
    25: r"$\nu_{\mu}$ CC 3+p",
    3: r"NC $\nu$",
    5: r"Dirt",  # (Outside TPC)
}

paper_labels = {
    11: r"$\nu_e$ CC",
    111: r"MiniBooNE LEE",
    2: r"$\nu$ other",
    31: r"$\nu$ with $\pi^{0}$",
    5: r"Dirt",  # (Outside TPC)
}

paper_labels_xsec = {
    1: r"$\nu_e$ CC with $\pi$",
    10: r"$\nu_e$ CC 0p0$\pi$",
    11: r"$\nu_e$ CC Np0$\pi$",
    111: r"MiniBooNE LEE",
    2: r"$\nu$ other",
    31: r"$\nu$ with $\pi^{0}$",
    5: r"Dirt",  # (Outside TPC)
}


category_labels = {
    1: r"$\nu_e$ CC",
    10: r"$\nu_e$ CC0$\pi$0p",
    11: r"$\nu_e$ CC0$\pi$Np",
    111: r"MiniBooNE LEE",
    2: r"$\nu_{\mu}$ CC",
    222: r"$\nu_{\mu}$ CC w/ Michel",
    21: r"$\nu_{\mu}$ CC $\pi^{0}$",
    22: r"$\nu_{\mu}$ CC 0p",
    23: r"$\nu_{\mu}$ CC 1p",
    24: r"$\nu_{\mu}$ CC 2p",
    25: r"$\nu_{\mu}$ CC 3+p",
    3: r"$\nu$ NC",
    31: r"$\nu$ NC $\pi^{0}$",
    4: r"Cosmic",
    5: r"Out. fid. vol.",
    # eta categories start with 80XX
    801: r"$\eta \rightarrow$ other",
    802: r"$\nu_{\mu} \eta \rightarrow \gamma\gamma$",
    803: r"1 $\pi^0$",
    804: r"2 $\pi^0$",
    807: r"3+ $\pi^0$",
    805: r"$\nu$ other",
    806: r"out of FV",
    6: r"other",
    0: r"No slice",
}

# These labels are pretty much the same as category_labels except for a couple of edits that are required for the 1e1p study
category_labels_1e1p = {
    1: r"$\nu_e$ CC",
    10: r"$\nu_e$ CC0$\pi$0p",
    #11: r"$\nu_e$ CC0$\pi$Np",
    12: r"$\nu_e$ CC0$\pi$1p",
    13: r"$\nu_e$ CC0$\pi$2+p",
    111: r"MiniBooNE LEE",
    2: r"$\nu_{\mu}$ CC",
    222: r"$\nu_{\mu}$ CC w/ Michel",
    21: r"$\nu_{\mu}$ CC $\pi^{0}$",
    22: r"$\nu_{\mu}$ CC 0p",
    23: r"$\nu_{\mu}$ CC 1p",
    24: r"$\nu_{\mu}$ CC 2p",
    25: r"$\nu_{\mu}$ CC 3+p",
    3: r"$\nu$ NC",
    31: r"$\nu$ NC $\pi^{0}$",
    4: r"Cosmic",
    5: r"Out. fid. vol.",
    # eta categories start with 80XX
    801: r"$\eta \rightarrow$ other",
    802: r"$\nu_{\mu} \eta \rightarrow \gamma\gamma$",
    803: r"1 $\pi^0$",
    804: r"2 $\pi^0$",
    807: r"3+ $\pi^0$",
    805: r"$\nu$ other",
    806: r"out of FV",
    6: r"other",
    0: r"No slice",
}


flux_labels = {1: r"$\pi$", 10: r"K", 111: r"MiniBooNE LEE", 0: r"backgrounds"}

sample_labels = {
    0: r"data",
    1: r"mc",
    2: r"nue",
    3: r"ext",
    4: r"lee",
    5: r"dirt",
    6: r"ccnopi",
    7: r"cccpi",
    8: r"ncnopi",
    9: r"nccpi",
    10: r"ncpi0",
    11: r"ccpi0",
    802: r"eta",
}

flux_colors = {
    0: "xkcd:cerulean",
    111: "xkcd:goldenrod",
    10: "xkcd:light red",
    1: "xkcd:purple",
}


pdg_labels = {
    2212: r"$p$",
    13: r"$\mu$",
    11: r"$e$",
    111: r"$\pi^0$",
    -13: r"$\mu$",
    -11: r"$e$",
    211: r"$\pi^{\pm}$",
    -211: r"$\pi$",
    2112: r"$n$",
    22: r"$\gamma$",
    321: r"$K$",
    -321: r"$K$",
    0: "Cosmic",
}

int_labels = {
    0: "QE",
    1: "Resonant",
    2: "DIS",
    3: "Coherent",
    4: "Coherent Elastic",
    5: "Electron scatt.",
    6: "IMDAnnihilation",
    7: r"Inverse $\beta$ decay",
    8: "Glashow resonance",
    9: "AMNuGamma",
    10: "MEC",
    11: "Diffractive",
    12: "EM",
    13: "Weak Mix",
}


int_colors = {
    0: "bisque",
    1: "darkorange",
    2: "goldenrod",
    3: "lightcoral",
    4: "forestgreen",
    5: "turquoise",
    6: "teal",
    7: "deepskyblue",
    8: "steelblue",
    80: "steelblue",
    81: "steelblue",
    82: "steelblue",
    9: "royalblue",
    10: "crimson",
    11: "mediumorchid",
    12: "magenta",
    13: "pink",
    111: "black",
}

category_colors = {
    4: "xkcd:light red",
    5: "xkcd:brick",
    8: "xkcd:cerulean",
    2: "xkcd:cyan",
    21: "xkcd:cerulean",
    222: "pink",
    22: "xkcd:khaki",  # "xkcd:lightblue",
    23: "xkcd:maroon",  # "xkcd:cyan",
    24: "xkcd:teal",  # "steelblue",
    25: "xkcd:coral",  # "blue",
    3: "xkcd:cobalt",
    31: "xkcd:sky blue",
    1: "xkcd:green",
    10: "xkcd:mint green",
    11: "xkcd:lime green",
    12: "xkcd:soft green",
    13: "xkcd:bright lime",
    111: "xkcd:goldenrod",
    6: "xkcd:grey",
    0: "xkcd:black",
    # eta categories
    803: "xkcd:cerulean",
    804: "xkcd:blue",
    807: "xkcd:blurple",
    801: "xkcd:purple",
    802: "xkcd:lavender",
    806: "xkcd:crimson",
    805: "xkcd:cyan",
}

pdg_colors = {
    2212: "#a6cee3",
    22: "#1f78b4",
    13: "#b2df8a",
    211: "#33a02c",
    111: "#137e6d",
    0: "#e31a1c",
    11: "#ff7f00",
    321: "#fdbf6f",
    2112: "#cab2d6",
}
