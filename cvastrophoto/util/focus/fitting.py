import tempfile
import numpy
import PIL

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.compose


try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def fit_focus(X, Y):
    """ Fit a focus curve

    Fits a regression model that fits the focus curve implied by X-Y

    :param X: Sequence of focus positions
    :param Y: Sequence of focus measurements, should match X in shape

    :returns: The trained sklearn model
    """
    model = sklearn.compose.TransformedTargetRegressor(
        sklearn.pipeline.Pipeline([
            ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2)),
            ('linear', sklearn.linear_model.RidgeCV(alphas=numpy.logspace(0, 1, 12), cv=2))
        ]),
        func=lambda Y:Y ** -0.25,
        inverse_func=lambda Y:Y ** -4,
    )
    X = numpy.asanyarray(X)
    Y = numpy.asanyarray(Y)
    X = X.reshape((X.size, 1))
    Y = Y.reshape((Y.size, 1))
    model.fit(X, Y)
    return model


def find_best_focus_from_model(model, fmin, fmax):
    """ Produce a full fitted curve and locate the vertex

    :param model: The sklearn model fitting the focus curve
    :param fmin: Minimum focus position to compute
    :param fmax: Maximum focus position to compute

    :returns: Tuple with (best_focus_pos, best_focus)
    """
    Xfull = numpy.arange(fmin, fmax)
    Xfull = Xfull.reshape((Xfull.size, 1))
    Yfull = model.predict(Xfull)
    best_focus_ix = Yfull[:,0].argmax()
    best_focus_pos = int(Xfull[best_focus_ix,0])
    best_focus = float(Yfull[best_focus_ix,0])

    return best_focus_pos, best_focus


def find_best_focus(samples):
    """ Find the critical focus point

    Takes samples in the form of (pos, fwhm, focus contrast) measurements,
    and tries to find the position where the critical focus point lies.

    :param samples: A sequence of samples
    :returns: A tuple with:

        * best_sample_fwhm: The sample with the best FWHM
        * best_sample_focus: The sample with the best focus contrast
        * best: The best sample overall (usually extrapolated from a curve fit)
        * model_focus: The curve fitting model for focus contrast
    """

    best_sample_focus = max(samples, key=lambda sample:sample[2])

    best_focus = best_sample_focus[2]
    best_focus_fwhm = best_sample_focus[1]

    X = numpy.array([sample[0] for sample in samples])
    Y = numpy.array([sample[2] for sample in samples])
    model_focus = fit_focus(X, Y)

    best_focus_pos, best_focus = find_best_focus_from_model(model_focus, int(X.min()), int(X.max()))

    # Check FWHM, only samples above median focus ranking (other samples tend to be inaccurate)
    median_focus = numpy.median([s[2] for s in samples])
    best_samples = [s for s in samples if s[2] >= median_focus]

    best_sample_fwhm = min(best_samples, key=lambda sample:sample[1])
    best_sample_focus = best = (best_focus_pos, best_focus_fwhm, best_focus)

    return best_sample_fwhm, best_sample_focus, best, model_focus


def plot_focus_curve(min_pos, max_pos, focus_model, samples, best_focus, best_fwhm, best):
    """ Produce a figure showing focus data

    :param int min_pos: Minimum focus position to graph
    :param int max_pos: Maximum focus position to graph
    :param focus_model: Trained sklearn model that fits the focus curve
    :param best_focus: Sample with best focus
    :param best_fwhm: Sample with best fwhm
    :param best: Sample with best overall
    :param samples: A sequence of (pos, fwhm, focus) measurements

    :returns: a PIL image with the graph
    """
    if plt is None:
        raise NotImplementedError("Can't plot without a working matplotlib")

    X = numpy.linspace(min_pos, max_pos, 200)
    X = X.reshape((X.size, 1))
    Y = focus_model.predict(X)

    # Temporary hack until we can have a proper UI for this
    samples = numpy.array(sorted(samples)).T
    best_focus = numpy.array([best_focus]).T
    best_fwhm = numpy.array([best_fwhm]).T
    best = numpy.array([best]).T
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].scatter(samples[0], samples[1], c="#008000", marker="+")
    ax[1].scatter(samples[0], samples[2], c="#000080", marker=".")
    ax[0].scatter(best_fwhm[0], best_fwhm[1], c="#00F000", marker="*")
    ax[1].scatter(best_focus[0], best_focus[2], c="#0000F0", marker="o")
    ax[0].scatter(best[0], best[1], c="#00F000", marker="1")
    ax[1].scatter(best[0], best[2], c="#0000F0", marker="1")
    ax[1].plot(X[:,0], Y[:,0], c="#000040", linestyle="dashed")
    ax[1].set_xlabel('position')
    ax[1].set_ylabel('contrast')
    ax[0].set_ylabel('FWHM')
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f)
        f.seek(0)
        img = PIL.Image.open(f)
        img.load()
    return img
