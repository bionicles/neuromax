"""Plotting functions."""
import sys
from itertools import count
from functools import partial

# For plot tests, matplotlib must be set to headless mode early
if 'pytest' in sys.modules:
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.optimize import OptimizeResult

from skopt.space import Categorical


def plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces.

    Parameters
    ----------
    * `args[i]` [`OptimizeResult`, list of `OptimizeResult`, or tuple]:
        The result(s) for which to plot the convergence trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in transparency, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `true_minimum` [float, optional]:
        The true minimum value of the function, if known.

    * `yscale` [None or string, optional]:
        The scale for the y-axis.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            mins = [np.min(results.func_vals[:i])
                    for i in range(1, n_calls + 1)]
            ax.plot(range(1, n_calls + 1), mins, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            mins = [[np.min(r.func_vals[:i]) for i in iterations]
                    for r in results]

            for m in mins:
                ax.plot(iterations, m, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(mins, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    if true_minimum:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum or name:
        ax.legend(loc="best")

    return ax


def plot_regret(*args, **kwargs):
    """Plot one or several cumulative regret traces.

    Parameters
    ----------
    * `args[i]` [`OptimizeResult`, list of `OptimizeResult`, or tuple]:
        The result(s) for which to plot the cumulative regret trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding cumulative
        regret traces in transparency, along with the average cumulative regret
        trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `true_minimum` [float, optional]:
        The true minimum value of the function, if known.

    * `yscale` [None or string, optional]:
        The scale for the y-axis.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Cumulative regret plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\sum_{i=0}^n(f(x_i) - optimum)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    if true_minimum is None:
        results = []
        for res in args:
            if isinstance(res, tuple):
                res = res[1]

            if isinstance(res, OptimizeResult):
                results.append(res)
            elif isinstance(res, list):
                results.extend(res)
        true_minimum = np.min([np.min(r.func_vals) for r in results])

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            regrets = [np.sum(results.func_vals[:i] - true_minimum)
                       for i in range(1, n_calls + 1)]
            ax.plot(range(1, n_calls + 1), regrets, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            regrets = [[np.sum(r.func_vals[:i] - true_minimum) for i in
                        iterations] for r in results]

            for cr in regrets:
                ax.plot(iterations, cr, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(regrets, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    if name:
        ax.legend(loc="best")

    return ax


def _format_scatter_plot_axes(ax, space, ylabel, dim_labels=None):
    # Work out min, max of y axis for the diagonal so we can adjust
    # them all to the same value
    diagonal_ylim = (np.min([ax[i, i].get_ylim()[0]
                             for i in range(space.n_dims)]),
                     np.max([ax[i, i].get_ylim()[1]
                             for i in range(space.n_dims)]))

    if dim_labels is None:
        dim_labels = ["$X_{%i}$" % i if d.name is None else d.name
                      for i, d in enumerate(space.dimensions)]
    # Axes for categorical dimensions are really integers; we have to
    # label them with the category names
    iscat = [isinstance(dim, Categorical) for dim in space.dimensions]

    # Deal with formatting of the axes
    for i in range(space.n_dims):  # rows
        for j in range(space.n_dims):  # columns
            ax_ = ax[i, j]

            if j > i:
                ax_.axis("off")
            elif i > j:        # off-diagonal plots
                # plots on the diagonal are special, like Texas. They have
                # their own range so do not mess with them.
                if not iscat[i]:  # bounds not meaningful for categoricals
                    ax_.set_ylim(*space.dimensions[i].bounds)
                if iscat[j]:
                    # partial() avoids creating closures in a loop
                    ax_.xaxis.set_major_formatter(FuncFormatter(
                            partial(_cat_format, space.dimensions[j])))
                else:
                    ax_.set_xlim(*space.dimensions[j].bounds)
                if j == 0:      # only leftmost column (0) gets y labels
                    ax_.set_ylabel(dim_labels[i])
                    if iscat[i]:    # Set category labels for left column
                        ax_.yaxis.set_major_formatter(FuncFormatter(
                            partial(_cat_format, space.dimensions[i])))
                else:
                    ax_.set_yticklabels([])

                # for all rows except ...
                if i < space.n_dims - 1:
                    ax_.set_xticklabels([])
                # ... the bottom row
                else:
                    [l.set_rotation(45) for l in ax_.get_xticklabels()]
                    ax_.set_xlabel(dim_labels[j])

                # configure plot for linear vs log-scale
                if space.dimensions[j].prior == 'log-uniform':
                    ax_.set_xscale('log')
                else:
                    ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[j]))

                if space.dimensions[i].prior == 'log-uniform':
                    ax_.set_yscale('log')
                else:
                    ax_.yaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[i]))

            else:       # diagonal plots
                ax_.set_ylim(*diagonal_ylim)
                ax_.yaxis.tick_right()
                ax_.yaxis.set_label_position('right')
                ax_.yaxis.set_ticks_position('both')
                ax_.set_ylabel(ylabel)

                ax_.xaxis.tick_top()
                ax_.xaxis.set_label_position('top')
                ax_.set_xlabel(dim_labels[j])

                if space.dimensions[i].prior == 'log-uniform':
                    ax_.set_xscale('log')
                else:
                    ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both',
                                                            integer=iscat[i]))
                    if iscat[i]:
                        ax_.xaxis.set_major_formatter(FuncFormatter(
                            partial(_cat_format, space.dimensions[i])))

    return ax


def partial_dependence(space, model, i, j=None, sample_points=None,
                       n_samples=250, n_points=40):
    """Calculate the partial dependence for dimensions `i` and `j` with
    respect to the objective value, as approximated by `model`.

    The partial dependence plot shows how the value of the dimensions
    `i` and `j` influence the `model` predictions after "averaging out"
    the influence of all other dimensions.

    Parameters
    ----------
    * `space` [`Space`]
        The parameter space over which the minimization was performed.

    * `model`
        Surrogate model for the objective function.

    * `i` [int]
        The first dimension for which to calculate the partial dependence.

    * `j` [int, default=None]
        The second dimension for which to calculate the partial dependence.
        To calculate the 1D partial dependence on `i` alone set `j=None`.

    * `sample_points` [np.array, shape=(n_points, n_dims), default=None]
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points`.

    * `n_samples` [int, default=100]
        Number of random samples to use for averaging the model function
        at each of the `n_points`. Only used when `sample_points=None`.

    * `n_points` [int, default=40]
        Number of points at which to evaluate the partial dependence
        along each dimension `i` and `j`.

    Returns
    -------
    For 1D partial dependence:

    * `xi`: [np.array]:
        The points at which the partial dependence was evaluated.

    * `yi`: [np.array]:
        The value of the model at each point `xi`.

    For 2D partial dependence:

    * `xi`: [np.array, shape=n_points]:
        The points at which the partial dependence was evaluated.
    * `yi`: [np.array, shape=n_points]:
        The points at which the partial dependence was evaluated.
    * `zi`: [np.array, shape=(n_points, n_points)]:
        The value of the model at each point `(xi, yi)`.

    For Categorical variables, the `xi` (and `yi` for 2D) returned are
    the indices of the variable in `Dimension.categories`.
    """
    # The idea is to step through one dimension, evaluating the model with
    # that dimension fixed and averaging over random values in all other
    # dimensions.  (Or step through 2 dimensions when i and j are given.)
    # Categorical dimensions make this interesting, because they are one-
    # hot-encoded, so there is a one-to-many mapping of input dimensions
    # to transformed (model) dimensions.

    if sample_points is None:
        sample_points = space.transform(space.rvs(n_samples=n_samples))

    # dim_locs[i] is the (column index of the) start of dim i in sample_points
    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    if j is None:
        xi, xi_transformed = _evenly_sample(space.dimensions[i], n_points)
        yi = []
        for x_ in xi_transformed:
            rvs_ = np.array(sample_points)      # copy
            rvs_[:, dim_locs[i]:dim_locs[i + 1]] = x_
            yi.append(np.mean(model.predict(rvs_)))

        return xi, yi

    else:
        xi, xi_transformed = _evenly_sample(space.dimensions[j], n_points)
        yi, yi_transformed = _evenly_sample(space.dimensions[i], n_points)

        zi = []
        for x_ in xi_transformed:
            row = []
            for y_ in yi_transformed:
                rvs_ = np.array(sample_points)      # copy
                rvs_[:, dim_locs[j]:dim_locs[j + 1]] = x_
                rvs_[:, dim_locs[i]:dim_locs[i + 1]] = y_
                row.append(np.mean(model.predict(rvs_)))
            zi.append(row)

        return xi, yi, np.array(zi).T


def plot_objective(result, levels=10, n_points=40, n_samples=250, size=2,
                   zscale='linear', dimensions=None):
    """Pairwise partial dependence plot of the objective function.

    The diagonal shows the partial dependence for dimension `i` with
    respect to the objective function. The off-diagonal shows the
    partial dependence for dimensions `i` and `j` with
    respect to the objective function. The objective function is
    approximated by `result.model.`

    Pairwise scatter plots of the points at which the objective
    function was directly evaluated are shown on the off-diagonal.
    A red point indicates the found minimum.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `levels` [int, default=10]
        Number of levels to draw on the contour plot, passed directly
        to `plt.contour()`.

    * `n_points` [int, default=40]
        Number of points at which to evaluate the partial dependence
        along each dimension.

    * `n_samples` [int, default=250]
        Number of random samples to use for averaging the model function
        at each of the `n_points`.

    * `size` [float, default=2]
        Height (in inches) of each facet.

    * `zscale` [str, default='linear']
        Scale to use for the z axis of the contour plots. Either 'linear'
        or 'log'.

    * `dimensions` [list of str, default=None] Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    space = result.space
    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))
    samples, minimum, _ = _map_categories(space, result.x_iters, result.x)

    if zscale == 'log':
        locator = LogLocator()
    elif zscale == 'linear':
        locator = None
    else:
        raise ValueError("Valid values for zscale are 'linear' and 'log',"
                         " not '%s'." % zscale)

    fig, ax = plt.subplots(space.n_dims, space.n_dims,
                           figsize=(size * space.n_dims, size * space.n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i == j:
                xi, yi = partial_dependence(space, result.models[-1], i,
                                            j=None,
                                            sample_points=rvs_transformed,
                                            n_points=n_points)

                ax[i, i].plot(xi, yi)
                ax[i, i].axvline(minimum[i], linestyle="--", color="r", lw=1)

            # lower triangle
            elif i > j:
                xi, yi, zi = partial_dependence(space, result.models[-1],
                                                i, j,
                                                rvs_transformed, n_points)
                ax[i, j].contourf(xi, yi, zi, levels,
                                  locator=locator, cmap='viridis_r')
                ax[i, j].scatter(samples[:, j], samples[:, i],
                                 c='k', s=10, lw=0.)
                ax[i, j].scatter(minimum[j], minimum[i],
                                 c=['r'], s=20, lw=0.)

    return _format_scatter_plot_axes(ax, space, ylabel="Partial dependence",
                                     dim_labels=dimensions)


def plot_evaluations(result, bins=20, dimensions=None):
    """Visualize the order in which points where sampled.

    The scatter plot matrix shows at which points in the search
    space and in which order samples were evaluated. Pairwise
    scatter plots are shown on the off-diagonal for each
    dimension of the search space. The order in which samples
    were evaluated is encoded in each point's color.
    The diagonal shows a histogram of sampled values for each
    dimension. A red point indicates the found minimum.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `bins` [int, bins=20]:
        Number of bins to use for histograms on the diagonal.

    * `dimensions` [list of str, default=None] Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    space = result.space
    # Convert categoricals to integers, so we can ensure consistent ordering.
    # Assign indices to categories in the order they appear in the Dimension.
    # Matplotlib's categorical plotting functions are only present in v 2.1+,
    # and may order categoricals differently in different plots anyway.
    samples, minimum, iscat = _map_categories(space, result.x_iters, result.x)
    order = range(samples.shape[0])
    fig, ax = plt.subplots(space.n_dims, space.n_dims,
                           figsize=(2 * space.n_dims, 2 * space.n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i == j:
                if iscat[j]:
                    bins_ = len(space.dimensions[j].categories)
                elif space.dimensions[j].prior == 'log-uniform':
                    low, high = space.bounds[j]
                    bins_ = np.logspace(np.log10(low), np.log10(high), bins)
                else:
                    bins_ = bins
                ax[i, i].hist(samples[:, j], bins=bins_, range=None if iscat[j]
                              else space.dimensions[j].bounds)

            # lower triangle
            elif i > j:
                ax[i, j].scatter(samples[:, j], samples[:, i],
                                 c=order, s=40, lw=0., cmap='viridis')
                ax[i, j].scatter(minimum[j], minimum[i],
                                 c=['r'], s=20, lw=0.)

    return _format_scatter_plot_axes(ax, space, ylabel="Number of samples",
                                     dim_labels=dimensions)


def _map_categories(space, points, minimum):
    """
    Map categorical values to integers in a set of points.

    Returns
    -------
    * `mapped_points` [np.array, shape=points.shape]:
        A copy of `points` with categoricals replaced with their indices in
        the corresponding `Dimension`.

    * `mapped_minimum` [np.array, shape=(space.n_dims,)]:
        A copy of `minimum` with categoricals replaced with their indices in
        the corresponding `Dimension`.

    * `iscat` [np.array, shape=(space.n_dims,)]:
       Boolean array indicating whether dimension `i` in the `space` is
       categorical.
    """
    points = np.asarray(points, dtype=object)  # Allow slicing, preserve cats
    iscat = np.repeat(False, space.n_dims)
    min_ = np.zeros(space.n_dims)
    pts_ = np.zeros(points.shape)
    for i, dim in enumerate(space.dimensions):
        if isinstance(dim, Categorical):
            iscat[i] = True
            catmap = dict(zip(dim.categories, count()))
            pts_[:, i] = [catmap[cat] for cat in points[:, i]]
            min_[i] = catmap[minimum[i]]
        else:
            pts_[:, i] = points[:, i]
            min_[i] = minimum[i]
    return pts_, min_, iscat


def _evenly_sample(dim, n_points):
    """Return `n_points` evenly spaced points from a Dimension.

    Parameters
    ----------
    * `dim` [`Dimension`]
        The Dimension to sample from.  Can be categorical; evenly-spaced
        category indices are chosen in order without replacement (result
        may be smaller than `n_points`).

    * `n_points` [int]
        The number of points to sample from `dim`.

    Returns
    -------
    * `xi`: [np.array]:
        The sampled points in the Dimension.  For Categorical
        dimensions, returns the index of the value in
        `dim.categories`.

    * `xi_transformed`: [np.array]:
        The transformed values of `xi`, for feeding to a model.
    """
    cats = np.array(getattr(dim, 'categories', []), dtype=object)
    if len(cats):   # Sample categoricals while maintaining order
        xi = np.linspace(0, len(cats) - 1, min(len(cats), n_points), dtype=int)
        xi_transformed = dim.transform(cats[xi])
    else:
        bounds = dim.bounds
        # XXX use linspace(*bounds, n_points) after python2 support ends
        xi = np.linspace(bounds[0], bounds[1], n_points)
        xi_transformed = dim.transform(xi)
    return xi, xi_transformed


def _cat_format(dimension, x, _):
    """Categorical axis tick formatter function.  Returns the name of category
    `x` in `dimension`.  Used with `matplotlib.ticker.FuncFormatter`."""
    return str(dimension.categories[int(x)])
