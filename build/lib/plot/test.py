
#############################################################################
#############################################################################
#############################################################################

def mycorner(
        sample,
        bins=10, # As for histogramdd but D = 3
        range=None, # As for histogramdd but D = 3
        normed=None, # As for histogramdd but D = 3
        weights=None, # As for histogramdd
        density=False, # As for histogramdd
        # data=None, # indexable object, optional
        # hist2d kwargs
        # cmin=None,
        # cmax=None,
        cmap=None,
        norm=colors.Normalize(),
        alpha=None, # (3,) array_like
        # **pcolormesh_kwargs=None,
        # # hist kwargs
        # **stairs_kwargs=None,
        # # axis keywords
        vmin_1d=None, # float
        vmax_1d=None, # float
        vmin_2d=None, # float
        vmax_2d=None, # float
        #
        coord_sys="cartesian",
        xscale="linear",
        yscale="linear",
        #
        axis_labels=None, # (3,) array_like
        ticks=None,
        tick_labels=None,
        #
        cbar=False,
        cbar_label=None,
        cbar_ticks=None,
        width=3.307
):
    """Plot histograms for a three-dimensional sample

    Plot all one- and two-dimensional histograms for a data set
    These are approximations of all one- and two-dimensional
    marginalizations of a multivariable PDF, arranged as a corner
    plot.

    Follows the API of Matplotlib function :func:`pyplot.hist2d`.

    Parameters
    ----------

    Returns
    -------

    h: ???

        The bi-dimensional histogram of samples x and y. Values in x
        are histogrammed along the first dimension and values in y are
        histogrammed along the second dimension.

    edges: ???

        The bin edges along the x axis.

    image: matplotlib.figure

        An array of histograms, as a :func:`matplotlib.figure` object.

    See also
    --------

    Examples
    --------

    """
    # Condition data
    sample = np.asarray(sample)
    _, dim_sample = sample.shape
    # if isinstance(bins, int):
    #     bins = 3*[bins]

    # Compute data
    if coord_sys == "cartesian":
        coords = [
            ["x"],
            ["x", "y"], ["y"],
            ["x", "z"], ["y", "z"], ["z"]
        ]
    elif coord_sys == "cylindrical":
        coords = [
            ["rho"],
            ["rho", "phi"], ["phi"],
            ["rho", "z"], ["phi", "z"], ["z"]
        ]
    elif coord_sys == "spherical":
        coords = [
            ["r"],
            ["r", "theta"], ["theta"],
            ["r", "phi"], ["theta", "phi"], ["phi"]
        ]
    else:
        raise ValueError("'coord_sys' must be one of 'cartesian', "
                         "'cylindrical', or 'spherical'.")
    hist = histogram.Histogram3d(sample, bins, range, normed, weights,
                                 density, coord_sys)
    hists = [hist.marginalized_histogram(coords_i) for coords_i in coords]

    vmin = 0.
    vmax = np.max([np.max(hist[0]) for hist in hists])
    
    if (isinstance(xscale, str)) or (xscale is None):
        xscale = 3*[xscale]

    # Figure specification
    # width = 8.4/2.54
    top_margin = 1./2.54
    bottom_margin = 1./2.54
    left_margin = 1./2.54
    right_margin = 1./2.54
    plot_width = width - left_margin - right_margin
    plot_height = plot_width
    height = bottom_margin + top_margin + plot_height

    # Column and row specification
    n_rows = dim_sample
    # TODO Without colour bar wspace and hspace are equal. But with
    # the colour bar they are not.
    if cbar:
        n_cols = n_rows + 1
        width_ratios = np.append(n_rows*[1.], [0.15])
        # n_cols = n_rows + 2
        # width_ratios = np.append(n_rows*[1.], [0.1, 0.1])
    else:
        n_cols = n_rows
        width_ratios = n_rows*[1.]
    # Make figure
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(n_rows,
                           n_cols,
                           width_ratios=width_ratios,
                           wspace=0.15,
                           hspace=0.15,
                           left=left_margin/width,
                           bottom=bottom_margin/height,
                           right=1. - left_margin/width,
                           top=1. - top_margin/height)
    hist_number = 0
    for i in np.arange(n_rows):
        for j in np.arange(i + 1):
            ax = fig.add_subplot(gs[i, j])
            if j < i:
                # Off-diagonal plots
                hist_ij, edges_ij = hists[hist_number]
                ax.pcolormesh(*edges_ij, hist_ij.T, norm=norm,
                              rasterized=True)
                # ax.pcolormesh(*edges_ij, hist_ij.T, rasterized=True)
                ax.set_xscale(xscale[j])
                # ax.set_aspect("auto")
                hist_number += 1
            else:
                # On-diagonal plots
                hist_ij = hists[hist_number]
                ax.stairs(hist_ij[0], hist_ij[1])
                ax.set_xscale(xscale[j])
                ax.set_yscale(yscale)
                # ax.set_aspect("auto")
                hist_number += 1
            if j == i:
                ax.set_ylim(vmin, vmax)
            if ticks:
                # Set x-ticks for off-diagonal plots
                ax.set_xticks(ticks[j])
                if j < i:
                    # Set y-ticks to match x-ticks for off-diagonal plots
                    ax.set_yticks(ticks[i])
                # Set y-ticks for on-diagonal plots
                if j == i:
                    try:
                        ax.set_yticks(ticks[3])
                    except IndexError:
                        pass
            if i == n_rows - 1:
                # Abscissa labels on bottom panels only
                if tick_labels:
                    ax.set_xticklabels(tick_labels[j])
                if axis_labels:
                    ax.set_xlabel(axis_labels[j])
            else:
                # Abscissa tick labels on bottom panels only
                ax.set_xticklabels([])
            if j == 0:
                # Ordinate labels on leftmost panels only
                if i == 0:
                    if tick_labels:
                        ax.set_yticklabels(tick_labels[3])
                        # try:
                        #     ax.set_yticklabels(tick_labels[3])
                        # except IndexError:
                        #     pass
                    if axis_labels:
                        ax.set_ylabel(axis_labels[3])
                        ax.yaxis.set_label_coords(-0.35, 0.5)
                        # try:
                        #     ax.set_ylabel(axis_labels[3])
                        #     ax.yaxis.set_label_coords(-0.35, 0.5)
                        # except IndexError:
                        #     pass
                    # if norm == "linear":
                    #     ax.ticklabel_format(axis="y", style="sci",
                    #                         scilimits=(-3, 3))
                    # elif norm == "log":
                    #     ax.yaxis.set_major_formatter(LogFormatterExponent())
                    # ax.ticklabel_format(axis="y", style="sci",
                    #                     scilimits=(-3, 3))
                    ax.ticklabel_format(axis="y", style="sci",
                                        scilimits=(-3, 3))
                # else:
                #     if tick_labels:
                #         ax.set_yticklabels(tick_labels[i])
                #     if axis_labels:
                #         ax.set_ylabel(axis_labels[i])
                #         ax.yaxis.set_label_coords(-0.35, 0.5)
                # # TODO Scientific notation
           # else:
           #     ax.annotate("hello", (0., 0.))
           #     # Ordinate tick labels on bottom panels only
           #     ax.set_yticklabels([])

    # # Modify axes properties
    # ax = np.array(fig.get_axes())
    # # Find min, max of on-diagonal plots (triangular numbers)
    # if not (vmin_1d and vmax_1d):
    #     ind_diag = [i*(i + 1)//2  - 1 for i in np.arange(1, dim_sample + 1)]
    #     ax_ylim = [ax_i.get_ylim() for ax_i in ax[ind_diag]]
    #     vmin_1d = np.min(ax_ylim)
    #     vmax_1d = np.max(ax_ylim)
    #     print("vmin_1d")
    #     print(vmin_1d)
    #     print("vmax_1d")
    #     print(vmax_1d)
    # # Find min, max of off-diagonal plots
    # if not (vmin_2d and vmax_2d):
    #     ind_diag = [i*(i + 1)//2  - 1 for i in np.arange(1, dim_sample + 1)]
    #     mask = np.ones(dim_sample*(dim_sample + 1)//2, dtype=bool)
    #     mask[ind_diag] = False
    #     clim = [ax_i.collections[0].properties()["clim"] for ax_i in ax[mask]]
    #     vmin_2d = np.min(clim)
    #     vmax_2d = np.max(clim)
    #     print("vmin_2d")
    #     print(vmin_2d)
    #     print("vmax_2d")
    #     print(vmax_2d)
    # # Set color limits and y limits
    for ax in fig.get_axes():
        try:
            # Off-diagonal plots
            ax.collections[0].set_clim(vmin_2d, vmax_2d)
        except IndexError:
            # On-diagonal plots
            if not ticks or len(ticks) != 4:
                ax.set_ylim(vmin_1d, vmax_1d)
    # # Colour bar
    # if cbar:
    #     # TO DO Scientific notation
    #     ax_cbar = plt.subplot(gs[0:,-1])
    #     cbar = Colorbar(ax=ax_cbar,
    #                     mappable=fig.get_axes()[1].collections[0],
    #                     label=cbar_label
    #                     )
    fig.align_ylabels(fig.axes)

    return fig

#############################################################################
#############################################################################
#############################################################################

def array(n_rows=1,
          n_cols=1,
          aspect=1.414214,
          cbar=False,
          sharex=True,
          sharey=True,
          gridspec_kw=None,
          figure_kw=None):
    # Set margins
    left_margin = 1.2*CENTIMETER
    right_margin = 1.2*CENTIMETER
    bottom_margin = 1.*CENTIMETER
    top_margin = 0.5*CENTIMETER

    # Compute figure width and height
    fig_width = 8.4*CENTIMETER
    if cbar:
        n_rows = n_rows
        n_cols = n_cols
        cbar_width = 0.4*CENTIMETER
        width_ratios = (
            n_cols*[(plot_width - 2.*cbar_width)/n_cols] + 2*[cbar_width]
        )
        aspect = n_cols*aspect/n_rows
        plot_width = fig_width - left_margin - right_margin - 2.*cbar
        plot_height = plot_width/aspect
        n_cols = n_cols + 2
    else:
        width_ratios = n_cols*[1.]
        aspect = n_cols*aspect/n_rows
        plot_width = fig_width - left_margin - right_margin
        plot_height = plot_width/aspect
    fig_height= plot_height + bottom_margin + top_margin

    # Gridspec kwargs
    left = left_margin/fig_width
    right = (left_margin + plot_width)/fig_width
    bottom = bottom_margin/fig_height
    top = (bottom_margin + plot_height)/fig_height

    # Make figure
    fig = plt.figure(**figure_kw)

    # Specify gridspec 
    if not gridspec_kw:
        gridspec_kw = dict(
            width_ratios=width_ratios,
            wspace=0.,
            hspace=0.,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            figure=fig
        )    
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        **gridspec_kw
    )

    # Return figure and axes
    if cbar:
        ax = [[fig.add_subplot(gs[j, i]) for i in range(n_cols - 2)]
              for j in range(n_rows)]
        cbar = fig.add_subplot(gs[0:, -1])
    
        return fig, ax, cbar
    else:
        ax = [[fig.add_subplot(gs[j, i]) for i in range(n_cols)]
              for j in range(n_rows)]
        return fig, ax

#############################################################################
#############################################################################
#############################################################################

#############################################################################
#############################################################################
#############################################################################

def array(n_rows=1,
          n_cols=1,
          aspect=1.414214,
          cbar=False,
          sharex=True,
          sharey=True,
          gridspec_kw=None,
          figure_kw=None):
    # Set margins
    left_margin = 1.2*CENTIMETER
    right_margin = 1.2*CENTIMETER
    bottom_margin = 1.*CENTIMETER
    top_margin = 0.5*CENTIMETER

    # Compute figure width and height
    fig_width = 8.4*CENTIMETER
    if cbar:
        n_rows = n_rows
        n_cols = n_cols
        cbar_width = 0.4*CENTIMETER
        width_ratios = (
            n_cols*[(plot_width - 2.*cbar_width)/n_cols] + 2*[cbar_width]
        )
        aspect = n_cols*aspect/n_rows
        plot_width = fig_width - left_margin - right_margin - 2.*cbar
        plot_height = plot_width/aspect
        n_cols = n_cols + 2
    else:
        width_ratios = n_cols*[1.]
        aspect = n_cols*aspect/n_rows
        plot_width = fig_width - left_margin - right_margin
        plot_height = plot_width/aspect
    fig_height= plot_height + bottom_margin + top_margin

    # Gridspec kwargs
    left = left_margin/fig_width
    right = (left_margin + plot_width)/fig_width
    bottom = bottom_margin/fig_height
    top = (bottom_margin + plot_height)/fig_height

    # Make figure
    fig = plt.figure(**figure_kw)

    # Specify gridspec 
    if not gridspec_kw:
        gridspec_kw = dict(
            width_ratios=width_ratios,
            wspace=0.,
            hspace=0.,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            figure=fig
        )    
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        **gridspec_kw
    )

    # Return figure and axes
    if cbar:
        ax = [[fig.add_subplot(gs[j, i]) for i in range(n_cols - 2)]
              for j in range(n_rows)]
        cbar = fig.add_subplot(gs[0:, -1])
    
        return fig, ax, cbar
    else:
        ax = [[fig.add_subplot(gs[j, i]) for i in range(n_cols)]
              for j in range(n_rows)]
        return fig, ax

#############################################################################
#############################################################################
#############################################################################
