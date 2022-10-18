"""
closuretest/plots.py

Plots of statistical estimators for closure tests
"""
import matplotlib.pyplot as plt

from reportengine.figure import figure
from validphys import plotutils


@figure
def plot_biases(biases_table):
    """
    Plot the bias of each experiment for all fits with bars. For information on
    how biases is calculated see `bias_experiment`
    """
    fig, ax = plotutils.barplot(
        biases_table.values.T,
        collabels=biases_table.index.values,
        datalabels=biases_table.columns.droplevel(1).values,
    )
    ax.set_title("Biases per experiment for each fit")
    ax.legend()
    return fig


@figure
def plot_delta_chi2(delta_chi2_bootstrap, fits):
    """Plots distributions of delta chi2 for each fit in `fits`.
    Distribution is generated by bootstrapping. For more information
    on delta chi2 see `delta_chi2_bootstrap`
    """
    delta_chi2 = delta_chi2_bootstrap.T
    labels = [fit.label for fit in fits]
    fig, ax = plt.subplots()
    for i, label in enumerate(labels):
        ax.hist(delta_chi2[:, i], alpha=0.3, label=label, zorder=100)
    plt.xlabel(r"$\Delta_{\chi^{2}}$")
    l = ax.legend()
    l.set_zorder(1000)
    ax.set_title(r"Total $\Delta_{\chi^{2}}$ for each fit")
    return fig


def errorbar_figure_from_table(df):
    """Given a table with even columns as central values as odd columns as errors
    plot an errorbar plot"""
    fig, ax = plotutils.plot_horizontal_errorbars(
        df.values[:, ::2].T,
        df.values[:, 1::2].T,
        df.index.values,
        df.columns.unique(0),
        xlim=0,
    )
    return fig, ax


@figure
def plot_fits_bootstrap_variance(fits_bootstrap_variance_table):
    """Plot variance as error bars, with mean and central value calculated
    from bootstrap sample
    """
    fig, ax = errorbar_figure_from_table(fits_bootstrap_variance_table)
    ax.set_title("Variance by experiment for closure fits")
    return fig


@figure
def plot_fits_bootstrap_bias(fits_bootstrap_bias_table):
    """Plot the bias for each experiment for all `fits` as a point with an error bar,
    where the error bar is given by bootstrapping the bias across replicas

    The number of bootstrap samples can be controlled by the parameter `bootstrap_samples`
    """
    fig, ax = errorbar_figure_from_table(fits_bootstrap_bias_table)
    ax.set_title("Bias by experiment for closure fits")
    return fig

# NOTE: this is the same as in theorycovariance/output.py -> find a way to use that
def matrix_plot_labels(df):
    """Returns the tick locations and labels, and the starting
    point values for each category,  based on a dataframe
    to be plotted. The dataframe is assumed to be multiindexed by
    (process, dataset, points) or else (dataset, points). The tick
    location is in the centre of the dataset, and labelling is by
    the outermost index of the multiindex."""
    if len(df.index[0]) == 3:
        proclabels = [x[0] for x in df.index]
        points = [x[2] for x in df.index]
        labels = proclabels
    elif len(df.index[0]) == 2:
        dslabels = [x[0] for x in df.index]
        points = [x[1] for x in df.index]
        labels = dslabels
    unique_ds = []
    unique_ds.append([labels[0], 0])
    for x in range(len(labels) - 1):
        if labels[x + 1] != labels[x]:
            unique_ds.append([labels[x + 1], x + 1])
    ticklabels = [unique_ds[x][0] for x in range(len(unique_ds))]
    startlocs = [unique_ds[x][1] for x in range(len(unique_ds))]
    startlocs += [len(labels)]
    ticklocs = [0 for x in range(len(startlocs) - 1)]
    for i in range(len(startlocs) - 1):
        ticklocs[i] = 0.5 * (startlocs[i + 1] + startlocs[i])
    return ticklocs, ticklabels, startlocs

@figure
def plot_diagonal_vs_manipulated(
    procs_covmat,
    procs_manip_covmat,
    procs_data_values,
):
    """Plot of sqrt(cov_ii)/|data_i| for cov = exp, theory, exp+theory"""
    import pandas as pd
    import numpy as np
    data = np.abs(procs_data_values)
    plot_index = procs_covmat.index
    sqrtdiags_manip = np.sqrt(np.diag(procs_manip_covmat)) / data
    sqrtdiags_manip = pd.DataFrame(sqrtdiags_manip.values, index=plot_index)
    sqrtdiags_manip.sort_index(0, inplace=True)
    oldindex = sqrtdiags_manip.index.tolist()
    sqrtdiags = np.sqrt(np.diag(procs_covmat)) / data
    sqrtdiags = pd.DataFrame(sqrtdiags.values, index=plot_index)
    sqrtdiags.sort_index(0, inplace=True)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(sqrtdiags_manip.values, ".", label="Manipulated", color="red")
    ax.plot(sqrtdiags.values, ".", label="Original", color="blue")
    ticklocs, ticklabels, startlocs = matrix_plot_labels(sqrtdiags_manip)
    plt.xticks(ticklocs, ticklabels, rotation=45, fontsize=20)
    startlocs_lines = [x - 0.5 for x in startlocs]
    ax.vlines(startlocs_lines, 0, len(data), linestyles="dashed")
    ax.set_ylabel(r"$\frac{\sqrt{cov_{ii}}}{|D_i|}$", fontsize=30)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_ylim([0, 0.5])
    ax.set_title(
        f"Square diagonal of covariance matrix before and after manipulation normalised to absolute value of data",
        fontsize=20,
    )
    ax.legend(fontsize=20)
    ax.margins(x=0)
    return fig

_procorder = ("DIS NC", "DIS CC", "DY", "JETS", "TOP")

_dsorder = (
    "BCDMSP",
    "BCDMSD",
    "SLACP",
    "SLACD",
    "NMC",
    "NMCPD",
    "HERAF2CHARM",
    "HERACOMBNCEP460",
    "HERACOMBNCEP575",
    "HERACOMBNCEP820",
    "HERACOMBNCEP920",
    "HERACOMBNCEM",
    "CHORUSNU",
    "CHORUSNB",
    "NTVNUDMN",
    "NTVNBDMN",
    "HERACOMBCCEP",
    "HERACOMBCCEM",
    "CDFZRAP",
    "D0ZRAP",
    "D0WEASY",
    "D0WMASY",
    "ATLASWZRAP36PB",
    "ATLASZHIGHMASS49FB",
    "ATLASLOMASSDY11EXT",
    "ATLASWZRAP11",
    "ATLASZPT8TEVMDIST",
    "ATLASZPT8TEVYDIST",
    "CMSWEASY840PB",
    "CMSWMASY47FB",
    "CMSWCHARMRAT",
    "CMSDY2D11",
    "CMSWMU8TEV",
    "CMSWCHARMTOT",
    "CMSZDIFF12",
    "LHCBZ940PB",
    "LHCBWZMU7TEV",
    "LHCBWZMU8TEV",
    "LHCBZEE2FB",
    "ATLAS1JET11",
    "CMSJETS11",
    "CDFR2KT",
    "ATLASTTBARTOT",
    "ATLASTOPDIFF8TEVTRAPNORM",
    "CMSTTBARTOT",
    "CMSTOPDIFF8TEVTTRAPNORM",
)


def _get_key(element):
    """The key used to sort covariance matrix dataframes according to
    the ordering of processes and datasets specified in _procorder and
    _dsorder."""
    from math import inf
    x1, y1, z1 = element
    x2 = _procorder.index(x1) if x1 in _procorder else inf
    y2 = _dsorder.index(y1) if y1 in _dsorder else inf
    z2 = z1
    newelement = (x2, y2, z2)
    return newelement

# NOTE: this is the same as in theorycovariance/output.py -> find a way to use that
def plot_covmat_heatmap(covmat, title):
    from matplotlib import cm, colors as mcolors
    """Matrix plot of a covariance matrix."""
    df = covmat
    df.sort_index(0, inplace=True)
    df.sort_index(1, inplace=True)
    oldindex = df.index.tolist()
    newindex = sorted(oldindex, key=_get_key)
    # reindex index
    newdf = df.reindex(newindex)
    # reindex columns by transposing, reindexing, then transposing back
    newdf = (newdf.T.reindex(newindex)).T
    matrix = newdf.values
    fig, ax = plt.subplots(figsize=(15, 15))
    matrixplot = ax.matshow(
        100 * matrix,
        cmap=cm.Spectral_r,
        norm=mcolors.SymLogNorm(
            linthresh=0.01,
            linscale=10,
            vmin=-100 * matrix.max(),
            vmax=100 * matrix.max(),
        ),
    )
    cbar = fig.colorbar(matrixplot, fraction=0.046, pad=0.04)
    cbar.set_label(label="% of data", fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(title, fontsize=25)
    ticklocs, ticklabels, startlocs = matrix_plot_labels(newdf)
    plt.xticks(ticklocs, ticklabels, rotation=30, ha="right", fontsize=20)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(ticklocs, ticklabels, fontsize=20)
    # Shift startlocs elements 0.5 to left so lines are between indexes
    startlocs_lines = [x - 0.5 for x in startlocs]
    ax.vlines(startlocs_lines, -0.5, len(matrix) - 0.5, linestyles="dashed")
    ax.hlines(startlocs_lines, -0.5, len(matrix) - 0.5, linestyles="dashed")
    ax.margins(x=0, y=0)
    return fig

@figure
def plot_sampling_original_cov_heatmap(procs_sampling_covmat):
    fig = plot_covmat_heatmap(procs_sampling_covmat, "Original Sampling Covmat")
    return fig

@figure
def plot_sampling_manip_cov_heatmap(procs_sampling_covmat_manip):
    fig = plot_covmat_heatmap(procs_sampling_covmat_manip, "Manipulated Sampling Covmat")
    return fig