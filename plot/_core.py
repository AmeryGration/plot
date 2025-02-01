#!/usr/bin/env python

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

from matplotlib.ticker import LogFormatterExponent
from matplotlib.ticker import NullFormatter

CENTIMETER = 1./2.54

MNRAS_COLUMN_WIDTH = 8.45*CENTIMETER
MNRAS_GUTTER_WIDTH = 0.8*CENTIMETER
# MNRAS_COLUMN_HEIGHT = 23.9*CENTIMETER

def plot(aspect=1.4142135623730951, left_margin=0.4724409448818897,
         right_margin=0.4724409448818897, bottom_margin=0.39370078740157477,
         top_margin=0.39370078740157477):
    """Return a single blank plot

    The width of a figure is always that of a column in MNRAS (8.45
    cm). 

    """    
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
    res = fig, ax

    return res

def plot_with_colorbar(aspect=1.4142135623730951,
                       left_margin=0.4724409448818897,
                       right_margin=0.4724409448818897,
                       bottom_margin=0.39370078740157477,
                       top_margin=0.39370078740157477,
                       cbar_width=0.15748031496062992,
                       cbar_padding=0.15748031496062992):
    """Return a single blank plot with a color bar

    The width of a figure is always that of a column in MNRAS (8.45
    cm). 

    """    
    plot_width = (
        MNRAS_COLUMN_WIDTH - left_margin - right_margin - cbar_width
        - cbar_padding
    )
    n_rows = 1        
    n_cols = 3
    width_ratios = (plot_width, cbar_width, cbar_width)
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
    cbar = fig.add_subplot(gs[2])
    res = fig, ax, cbar

    return res

def array(nrows=1, ncols=1, aspect=1.4142135623730951,
          left_margin=0.4724409448818897, right_margin=0.4724409448818897,
          bottom_margin=0.39370078740157477, top_margin=0.39370078740157477,
          sharex=False, sharey=False, wspace=0.4724409448818897,
          hspace=0.39370078740157477):
    """Return an array of blank plots

    If `ncols = 1` then the width of a figure is always that of a
    column in MNRAS (8.45 cm). Otherwise it is the width of two
    columns (16.9 cm) plus the gutter (0.8 cm) in MNRAS (17.7
    cm).

    """
    if ncols == 1:
        fig_width = MNRAS_COLUMN_WIDTH
    else:
        fig_width = 2.*MNRAS_COLUMN_WIDTH + MNRAS_GUTTER_WIDTH
    plot_width = (
        (fig_width - left_margin - right_margin - (ncols - 1)*wspace)/ncols
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
    res = fig, ax

    return res

def array_with_colorbar(nrows=1, ncols=1, aspect=1.4142135623730951,
                        left_margin=0.4724409448818897,
                        right_margin=0.4724409448818897,
                        bottom_margin=0.39370078740157477,
                        top_margin=0.39370078740157477, sharex=False,
                        sharey=False, wspace=0.4724409448818897,
                        hspace=0.39370078740157477,
                        cbar_width=0.15748031496062992,
                        cbar_padding=0.15748031496062992):
    """Return an array of blank plots along with a global colorbar

    If `ncols = 1` then the width of a figure is always that of a
    column in MNRAS (8.45 cm). Otherwise it is the width of two
    columns (16.9 cm) plus the gutter (0.8 cm) in MNRAS (17.7
    cm). 

    """
    if ncols == 1:
        fig_width = MNRAS_COLUMN_WIDTH
    else:
        fig_width = 2.*MNRAS_COLUMN_WIDTH + MNRAS_GUTTER_WIDTH
    plot_width = (
        (fig_width - left_margin - right_margin - (ncols - 1)*wspace
         - cbar_width - cbar_padding)
        /ncols
    )
    width_ratio = (
        (ncols - 1)*(plot_width, wspace) + (plot_width,)
        + (0.5*cbar_padding,) + (cbar_width,)
    )
    ncols_ = 2*ncols + 1
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
    
    cbar = fig.add_subplot(gs[0:,-1])
    res = fig, ax, cbar

    return res

def array_with_colorbars(nrows=1, ncols=1, aspect=1.4142135623730951,
                         left_margin=0.4724409448818897,
                         right_margin=0.4724409448818897,
                         bottom_margin=0.39370078740157477,
                         top_margin=0.39370078740157477, sharex=False,
                         sharey=False, wspace=0.9448818897637794,
                         hspace=0.39370078740157477,
                         cbar_width=0.15748031496062992,
                         cbar_padding=0.15748031496062992):
    """Return an array of blank plots each with a global colorbar

    
    If `ncols = 1` then the width of a figure is always that of a
    column in MNRAS (8.45 cm). Otherwise it is the width of two
    columns (16.9 cm) plus the gutter (0.8 cm) in MNRAS (17.7
    cm).
    
    """
    if ncols == 1:
        fig_width = MNRAS_COLUMN_WIDTH
    else:
        fig_width = 2.*MNRAS_COLUMN_WIDTH + MNRAS_GUTTER_WIDTH
    plot_width = (
        (fig_width - left_margin - right_margin - (ncols - 1)*wspace)
        /ncols
        - cbar_width - cbar_padding
    )
    width_ratio = (
        (ncols - 1)*(plot_width, cbar_padding, cbar_width, wspace)
        + (plot_width, cbar_padding, cbar_width)
    )
    ncols_ = 4*ncols - 1
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
         for j in np.arange(0, 4*ncols, 4)]
    )
    cbar = np.array(
        [[fig.add_subplot(gs[i, j]) for i in np.arange(0, 2*nrows, 2)]
         for j in np.arange(2, ncols_, 4)]
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
    res = fig, ax, cbar

    return res

if __name__ == "__main__":
    # Plot
    x = np.linspace(-2.*np.pi, 2.*np.pi)
    y = np.sin(x)

    fig, ax = plot()
    ax.plot(x, y)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")    
    fig.savefig("plot.pdf")
    plt.show()

    # Plot with colorbar
    x = np.linspace(0., 1.)
    y = np.linspace(0., 1.)
    xx, yy = np.meshgrid(x, y)
    zz = xx*yy
    
    fig, ax, cbar = plot_with_colorbar()
    im = ax.pcolormesh(xx, yy, zz, rasterized=True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")    
    fig.colorbar(im, cax=cbar)
    cbar.set_ylabel(r"$z$")
    fig.savefig("plot_cbar.pdf")
    plt.show()

    # Array
    x = np.linspace(-2.*np.pi, 2.*np.pi)
    y_0 = np.sin(x)
    y_1 = np.cos(x)
    
    fig, ax = array(2, 1)
    ax[0][0].plot(x, y_0)
    ax[0][0].set_xlabel(r"$x$")
    ax[0][0].set_ylabel(r"$\sin(x)$")
    ax[0][1].plot(x, y_1)
    ax[0][1].set_xlabel(r"$x$")
    ax[0][1].set_ylabel(r"$\cos(x)$")
    # fig.align_ylabels(ax[0,:])
    fig.savefig("array.pdf")
    plt.show()

    # Array with colorbar
    x = np.linspace(0., 1.)
    y = np.linspace(0., 1.)
    xx, yy = np.meshgrid(x, y)
    zz_0 = xx*yy
    zz_1 = xx/yy
    
    fig, ax, cbar = array_with_colorbar(2, 1)
    im = ax[0][0].pcolormesh(xx, yy, zz_0, rasterized=True)
    ax[0][0].set_xlabel(r"$x$")
    ax[0][0].set_ylabel(r"$y$")
    ax[0][1].pcolormesh(xx, yy, zz_1, rasterized=True)
    ax[0][1].set_xlabel(r"$x$")
    ax[0][1].set_ylabel(r"$y$")
    fig.colorbar(im, cax=cbar)
    cbar.set_ylabel(r"$z$")
    fig.savefig("array_cbar.pdf")
    plt.show()
 
    fig, ax, cbar = array_with_colorbars(2, 1)
    plt.show()

    fig, ax, cbar = array_with_colorbars(2, 3)
    plt.show()
