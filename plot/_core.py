#!/usr/bin/env python

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import histogram

from matplotlib.ticker import LogFormatterExponent
from matplotlib.ticker import NullFormatter

# import utils

CENTIMETER = 1./2.54

MNRAS_COLUMN_WIDTH = 8.45*CENTIMETER
MNRAS_GUTTER_WIDTH = 0.8*CENTIMETER
MNRAS_COLUMN_HEIGHT = 23.9*CENTIMETER

LEFT_MARGIN = 1.2*CENTIMETER
RIGHT_MARGIN = 1.2*CENTIMETER
BOTTOM_MARGIN = 1.*CENTIMETER
TOP_MARGIN = 1.*CENTIMETER
CBAR_WIDTH = 0.4*CENTIMETER

def plot(aspect=1.4142135623730951, left_margin=0.4724409448818897,
         right_margin=0.4724409448818897, bottom_margin=0.39370078740157477,
         top_margin=0.39370078740157477, cbar=False):
    """Return a single blank plot

    The width of a figure is always that of a column in MNRAS (8.45
    cm). Within the margins of the figure is drawn either:
    
    - a single plot, or
    - a plot and a color bar on a left-hand side.

    """    
    if cbar:
        plot_width = (
            MNRAS_COLUMN_WIDTH - left_margin - right_margin - 2.*CBAR_WIDTH
        )
        n_rows = 1        
        n_cols = 3
        width_ratios = (plot_width, CBAR_WIDTH, CBAR_WIDTH)
    else:
        plot_width = MNRAS_COLUMN_WIDTH - left_margin - right_margin
        n_rows = 1
        n_cols = 1
        width_ratios = (1.,)
    
    plot_height = plot_width/aspect
    fig_width = MNRAS_COLUMN_WIDTH
    fig_height = bottom_margin + plot_height + top_margin

    left = left_margin/fig_width
    right = (fig_width - right_margin)/fig_width
    bottom = bottom_margin/fig_height
    top = (fig_height - top_margin)/fig_height

    # Make figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        width_ratios=width_ratios,
        wspace=0.,
        left=left,
        bottom=bottom,
        right=right,
        top=top
    )
    ax = fig.add_subplot(gs[0])
    if cbar:
        cbar = fig.add_subplot(gs[2])
        res = fig, ax, cbar
    else:
        res = fig, ax

    return res

def array(nrows=1, ncols=1, aspect=1.4142135623730951,
          left_margin=0.4724409448818897, right_margin=0.4724409448818897,
          bottom_margin=0.39370078740157477, top_margin=0.39370078740157477,
          sharex=False, sharey=False, wspace=0.4724409448818897,
          hspace=0.39370078740157477, cbar=False, subplot_kw=None,
          gridspec_kw=None):
    """Return an array of blank plots

    If `ncols = 1` then the width of a figure is always that of a
    column in MNRAS (8.45 cm). Otherwise it is the width of two
    columns (16.9 cm) plus the gutter (0.8 cm) in MNRAS (17.7
    cm). Within the margins of the figure is drawn either:
    
    - an array of plots, or
    - an array of plots and a color bar on the left-hand side.

    """
    cbar_padding = wspace
    if ncols == 1:
        fig_width = MNRAS_COLUMN_WIDTH
    else:
        fig_width = 2.*MNRAS_COLUMN_WIDTH + MNRAS_GUTTER_WIDTH
    if cbar:
        plot_width = (
            (fig_width - left_margin - right_margin - (ncols - 1)*wspace
             - 2.*CBAR_WIDTH)
            /ncols
        )
        width_ratio = (
            (ncols - 1)*(plot_width, wspace) + (plot_width,)
            + (0.5*cbar_padding,) + (CBAR_WIDTH,)
        )
        ncols_ = 2*ncols + 1
    else:
        plot_width = (
            (fig_width - left_margin - right_margin - (ncols - 1)*wspace)
            /ncols
        )
        width_ratio = (ncols - 1)*(plot_width, wspace) + (plot_width,)
        ncols_ = 2*ncols - 1

    plot_height = plot_width/aspect
    
    if nrows == 1:
        height_ratio = (1,)
        nrows_ = nrows
    else:
        height_ratio = (nrows - 1)*(plot_height, hspace) + (plot_height,)
        nrows_ = 2*nrows - 1

    fig_height = (
        nrows*plot_height
        + (nrows - 1)*hspace
        + bottom_margin
        + top_margin
    )
    left = left_margin/fig_width
    right = (fig_width - right_margin)/fig_width
    bottom = bottom_margin/fig_height
    top = (fig_height - top_margin)/fig_height
    
    # Make figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        nrows_,
        ncols_,
        width_ratios=width_ratio,
        height_ratios=height_ratio,
        wspace=0.,
        hspace=0.,
        left=left,
        bottom=bottom,
        right=right,
        top=top
    )
    ax = np.array(
        [[fig.add_subplot(gs[i, j]) for i in np.arange(0, 2*nrows, 2)]
         for j in np.arange(0, 2*ncols, 2)]
    )
    if sharex:
        for j in np.arange(0, ncols):
            for i, ax_i in enumerate(ax[j, :-1]):
                # Share x axes for each column
                ax_i.sharex(ax[j, -1])
                # Tick labels on bottom axes only
                ax_i.xaxis.set_tick_params(labelbottom=False)
    if sharey:
        for i in np.arange(0, nrows):
            for ax_j in ax[1:, i]:
                # Share y axes for each row
                ax_j.sharey(ax[0, i])
                # Tick labels on left axes only
                ax_j.yaxis.set_tick_params(labelleft=False)
    # if xlabel is not None:
    #     xlabel = np.atleast_1d(xlabel)
    #     for i, ax_i in enumerate(ax[:, -1]):
    #         ax_i.set_xlabel(xlabel[i])
    # if ylabel is not None:
    #     ylabel = np.atleast_1d(ylabel)
    #     for i, ax_i in enumerate(ax[0, :]):
    #         ax_i.set_ylabel(ylabel[i])

    # fig.align_ylabels(ax[0,:])
    
    if cbar:
        cbar = fig.add_subplot(gs[0:,-1])
        res = fig, ax, cbar
    else:
        res = fig, ax

    return res

# def corner(n=2):
#     """Return blank corner plot

#     Margins are fixed, and within those margins are drawn either
#     (1) the axes of a corner plot, or 
#     (2) the axes of a corner plot and a colour bar.

#     """
#     fig_width = 16.8*CENTIMETER
#     left_margin = 1.2*CENTIMETER
#     right_margin = 1.2*CENTIMETER
#     bottom_margin = 1.*CENTIMETER
#     top_margin = 0.5*CENTIMETER

#     plot_width = fig_width - left_margin - right_margin    

#     n_rows = n_cols = n
#     width_ratios = [1.]*n_cols
#     aspect = n_cols/n_rows
#     plot_height = plot_width/aspect

#     fig_height= plot_height + bottom_margin + top_margin
#     left = left_margin/fig_width
#     right = (left_margin + plot_width)/fig_width
#     bottom = bottom_margin/fig_height
#     top = (bottom_margin + plot_height)/fig_height

#     n_rows = n_cols = n
#     # Make figure
#     gridspec_kw = {
#         "width_ratios": width_ratios, 
#         "wspace": 0.15,
#         "hspace": 0.15,
#         "left": left,
#         "bottom": bottom,
#         "right": right,
#         "top": top,
#     }

#     fig, ax = plt.subplots(
#         n_rows,
#         n_cols,
#         figsize=(fig_width, fig_height),
#         gridspec_kw=gridspec_kw
#     )

#     for i in range(n):
#         for j in range(i + 1, n):
#             ax[i][j].set_axis_off()

#     return fig, ax
    
# def histogram3d(sample,
#                 bins=10,
#                 range=None, # As for histogramdd but D = 3
#                 normed=None, # As for histogramdd but D = 3
#                 weights=None, # As for histogramdd
#                 density=False, # As for histogramdd
#                 norm=None,
#                 coord_sys="cartesian",
#                 xscale=None,
#                 yscale=None,
#                 vmin_1d=None,
#                 vmax_1d=None,
#                 xlabel=None,
#                 ylabel=None):
#     sample = np.asarray(sample)
#     N, D = sample.shape

#     if (isinstance(xscale, str)):
#         xscale = 3*[xscale]
#     if (isinstance(ylabel, str)):
#         ylabel = 3*[ylabel]

#     # Compute marginalized histograms
#     if coord_sys == "cartesian":
#         coord_diag = [["x"], ["y"], ["z"]]
#         coord_offdiag = [["x", "y"], ["x", "z"], ["y", "z"]]
#     elif coord_sys == "cylindrical":
#         coord_diag = [["rho"], ["phi"], ["z"]]
#         coord_offdiag = [["rho", "phi"], ["rho", "z"], ["phi", "z"]]
#     elif coord_sys == "spherical":
#         coord_diag = [["r"], ["theta"],["phi"]]
#         coord_offdiag = [["r", "theta"], ["r", "phi"], ["theta", "phi"]]
#     else:
#         raise ValueError("'coord_sys' must be one of 'cartesian', "
#                          "'cylindrical', or 'spherical'.")

#     hist = histogram.Histogram3d(
#         sample, bins, range, normed, weights, density, coord_sys
#     )
#     hist_diag = [hist.marginalized_histogram(coord) for coord in coord_diag]
#     hist_offdiag = [
#         hist.marginalized_histogram(coord) for coord in coord_offdiag
#     ]

#     # Find common limits for on- and off-diagonal plots
#     if vmin_1d is None:
#         vmin_1d = np.nanmin([hist[0] for hist in hist_diag])
#     if vmax_1d is None:
#         vmax_1d = np.nanmax([hist[0] for hist in hist_diag])
#     vmin_2d = np.nanmin([hist[0] for hist in hist_offdiag])
#     vmax_2d = np.nanmax([hist[0] for hist in hist_offdiag])

#     if norm is None:
#         norm = colors.Normalize()
#     elif norm is colors.LogNorm or "log":
#         # Minimum cannot be zero
#         if vmin_1d == 0.:
#             vmin_1d = None
#         if vmin_2d == 0.:
#             vmin_2d = None
#     # norm = norm(vmin_2d, vmax_2d)    

#     # Create plot axes
#     fig, ax = corner(D)

#     # Populate plot axes
#     n = 0 # Keep track of off-diagonal histogram index
#     for i in np.arange(D):
#         for j in np.arange(i + 1):
#             if j < i:
#                 im = ax[i][j].pcolormesh(
#                     *hist_offdiag[n][1], hist_offdiag[n][0].T, norm=norm,
#                     rasterized=True, cmap="Greys",
#                     vmin=vmin_2d, vmax=vmax_2d
#                 )
#                 ax[i][j].set_aspect("auto")
#                 n += 1
#             else:
#                 ax[i][j].stairs(hist_diag[i][0], hist_diag[i][1])
#                 ax[i][j].set_ylim(vmin_1d, vmax_1d)
#                 if yscale:
#                     ax[i][j].set_yscale(yscale)
#             if xscale:
#                 ax[i][j].set_yscale(xscale[i])
#                 ax[i][j].set_xscale(xscale[j])
#             if i < D - 1:
#                 # Command ax.set_xticklabels() does not work with log x-scale
#                 ax[i][j].xaxis.set_major_formatter(NullFormatter())
#                 ax[i][j].xaxis.set_minor_formatter(NullFormatter())
#             else:
#                 if xlabel:
#                     ax[i][j].set_xlabel(xlabel[j])
#             if j > 0:
#                 # Command ax.set_yticklabels() does not work with log y-scale
#                 ax[i][j].yaxis.set_major_formatter(NullFormatter())
#                 ax[i][j].yaxis.set_minor_formatter(NullFormatter())
#             else:
#                 if ylabel:
#                     ax[i][j].set_ylabel(ylabel[i])
#     # fig.colorbar(im, cbar)
#     # cbar.set_ylabel(ylabel[0])

#     return fig

# if __name__ == "__main__":
#     plt.style.use("sm")

#     x_min = 1.e-2
#     x_max = 1.
#     y_min = 0.
#     y_max = 1.
#     z_min = 0.
#     z_max = 1.

#     sample = sp.stats.multivariate_normal([0., 0., 0.], np.eye(3)).rvs(100_000)
#     n_bins = 25
#     bins = [
#         np.logspace(np.log10(x_min), np.log10(x_max), n_bins),
#         np.linspace(y_min, y_max, n_bins),
#         np.linspace(z_min, z_max, n_bins)
#     ]
#     fig = histogram3d(sample,
#                       bins,
#                       coord_sys="cartesian",
#                       xscale=["log", "linear", "linear"],
#                       yscale="log",
#                       norm="log",
#                       xlabel=[r"$x$/kpc", r"$y$/kpc", r"$z$/kpc"],
#                       ylabel=[r"$\nu$", r"$y$/kpc", r"$z$/kpc"]
#                       )
#     plt.savefig("test.pdf")
#     plt.show()

#     R_min = 1.e-2
#     R_max = 10.
#     phi_min = -np.pi
#     phi_max = np.pi
#     z_min = 1.e-2
#     z_max = 10.

#     sample = sp.stats.multivariate_normal(np.zeros(6), np.eye(6)).rvs(100_000)
#     sample = utils.cartesian_to_cylindrical(sample).T[:3].T
#     n_bins = 25
#     bins = np.array([
#         np.logspace(np.log10(R_min), np.log10(R_max), n_bins),
#         np.linspace(phi_min, phi_max, n_bins),
#         # np.linspace(z_min, z_max, n_bins)
#         np.logspace(np.log10(z_min), np.log10(z_max), n_bins)
#     ])
#     fig = histogram3d(
#         sample,
#         bins,
#         coord_sys="cylindrical",                      
#         xscale=["log", "linear", "log"],
#         yscale="log",
#         norm="log",
#         xlabel=[r"$R$/kpc", r"$\phi$", r"$|z|$/kpc"],
#         ylabel=[r"$\hat{f}$", r"$\phi$", r"$|z|$/kpc"],
#         vmax_1d=1.e1,
#         vmin_1d=1.e-8,
#         density=True
#     )
#     plt.savefig("test.pdf")
#     plt.show()
