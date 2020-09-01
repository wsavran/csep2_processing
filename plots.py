
def plot_probability_test(evaluation_result, axes=None, plot_args=None, show=True):
    """ Plots the 


    Args:
        evaluation_result:

    Returns:

    """
    plot_args = plot_args or {}
    # handle plotting
    if axes:
        chained = True
    else:
        chained = False
    # supply fixed arguments to plots
    # might want to add other defaults here
    filename = plot_args.get('filename', None)
    fixed_plot_args = {'obs_label': evaluation_result.obs_name,
                       'sim_label': evaluation_result.sim_name}
    plot_args.update(fixed_plot_args)

    bins = plot_args.get('bins', 'auto')
    percentile = plot_args.get('percentile', 95)
    ax = plot_histogram(evaluation_result.test_distribution, evaluation_result.observed_statistic,
                        catalog=evaluation_result.obs_catalog_repr,
                        plot_args=plot_args,
                        bins=bins,
                        axes=axes,
                        percentile=percentile)

    # annotate plot with p-values
    if not chained:
        ax.annotate('$\gamma = P(X \leq x) = {:.2f}$\n$\omega = {:.2f}$'
                    .format(evaluation_result.quantile, evaluation_result.observed_statistic),
                    xycoords='axes fraction',
                    xy=(0.2, 0.6),
                    fontsize=14)

    title = plot_args.get('title', 'CSEP2 Probability Test')
    ax.set_title(title, fontsize=14)

    if filename is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    # func has different return types, before release refactor and remove plotting from evaluation.
    # plotting should be separated from evaluation.
    # evaluation should return some object that can be plotted maybe with verbose option.
    if show:
        pyplot.show()

    return ax


def plot_global_forecast(forecast, catalog=None, name=None):
    """
    Creates global plot from a forecast using a Robin projection. This should be used as a quick and dirty plot and probably
    will not suffice for publication quality figures.

    Args:
        forecast (csep.core.forecasts.MarkedGriddedDataSet): marked gridded data set
        catalog (csep.core.data.AbstractBaseCatalog):  data base class
        name (str): name of the data

    Returns:
        axes

    """
    fig, ax = pyplot.subplots(figsize=(18,11))
    m = Basemap(projection='robin', lon_0= -180, resolution='c')
    m.drawcoastlines(color = 'lightgrey', linewidth = 1.5)
    m.drawparallels(numpy.arange(-90.,120.,15.), labels=[1,1,0,1], linewidth= 0.0, fontsize = 13)
    m.drawmeridians(numpy.arange(0.,360.,40.), labels=[1,1,1,1], linewidth= 0.0, fontsize = 13)
    x, y = m(forecast.get_longitudes(), forecast.get_longitudes())
    cbar = ax.scatter(x, y, s = 2, c = numpy.log10(forecast.spatial_counts()), cmap = 'inferno', edgecolor='')
    a = fig.colorbar(cbar, orientation = 'horizontal', shrink = 0.5, pad = 0.01)
    if catalog is not None:
        x, y = m(catalog.get_longitudes(), catalog.get_latitudes())
        ax.scatter(x, y, color='black')
    a.ax.tick_params(labelsize = 14)
    a.ax.tick_params(labelsize = 14)
    if name is None:
        name='Global Forecast'
    a.set_label('{}\nlog$_{{10}}$(EQs / (0.1$^o$ x 0.1$^o$)'.format(name), size = 18)
    return ax