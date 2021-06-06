import tempfile
import numpy
import PIL
import logging
import functools

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.compose


try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


logger = logging.getLogger(__name__)


def masked_ufunc2(ufunc, x, y, where=True, out=None, dtype=numpy.float64):
    if out is None:
        out = x.astype(dtype)
    return ufunc(x, y, where=where, out=out)


def masked_ufunc1(ufunc, x, where=True, out=None, dtype=numpy.float64):
    if out is None:
        out = x.astype(dtype)
    return ufunc(x, where=where, out=out)


masked_power = functools.partial(masked_ufunc2, numpy.power)
masked_sqrt = functools.partial(masked_ufunc1, numpy.sqrt)
masked_square = functools.partial(masked_ufunc1, numpy.square)


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
        func=lambda Y:masked_power(Y, -0.25, where=Y > 0),
        inverse_func=lambda Y:masked_power(Y, -4, where=Y > 0),
    )
    X = numpy.asanyarray(X)
    Y = numpy.asanyarray(Y)
    X = X.reshape((X.size, 1))
    Y = Y.reshape((Y.size, 1))
    model.fit(X, Y)
    return model


def fit_fwhm(X, Y):
    """ Fit a FWHM curve

    Fits a regression model that fits the FWHM curve implied by X-Y

    :param X: Sequence of focus positions
    :param Y: Sequence of FWHM measurements, should match X in shape

    :returns: The trained sklearn model
    """
    model = sklearn.compose.TransformedTargetRegressor(
        sklearn.pipeline.Pipeline([
            ('poly', sklearn.preprocessing.PolynomialFeatures(degree=2)),
            ('linear', sklearn.linear_model.RidgeCV(alphas=numpy.logspace(0, 1, 12), cv=2))
        ]),
        func=lambda Y:-masked_square(-masked_square(Y, where=Y > 0), where=Y < 0),
        inverse_func=lambda Y:-masked_sqrt(-masked_sqrt(Y, where=Y > 0), where=Y < 0),
    )
    X = numpy.asanyarray(X)
    Y = numpy.asanyarray(Y)
    X = X.reshape((X.size, 1))
    Y = Y.reshape((Y.size, 1))
    model.fit(X, Y)
    return model


def find_best_focus_from_model(model, fmin, fmax, maximize=True):
    """ Produce a full fitted curve and locate the vertex

    :param model: The sklearn model fitting the focus curve
    :param fmin: Minimum focus position to compute
    :param fmax: Maximum focus position to compute

    :returns: Tuple with (best_focus_pos, best_focus)
    """
    Xfull = numpy.arange(fmin, fmax)
    Xfull = Xfull.reshape((Xfull.size, 1))
    Yfull = model.predict(Xfull)
    if maximize:
        best_focus_ix = Yfull[:,0].argmax()
    else:
        best_focus_ix = Yfull[:,0].argmin()
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
        * model_fwhm: The curve fitting model for FWHM
    """

    true_best_sample_focus = max(samples, key=lambda sample:sample[2])

    X = numpy.array([sample[0] for sample in samples])
    Yfocus = numpy.array([sample[2] for sample in samples])
    Yfwhm = numpy.array([sample[1] for sample in samples])

    try:
        model_focus = fit_focus(X, Yfocus)
    except Exception:
        logger.exception("Error fitting contrast model")
        model_focus = None

    try:
        model_fwhm = fit_fwhm(X, Yfwhm)
    except Exception:
        logger.exception("Error fitting FWHM model")
        model_fwhm = None

    best_focus_fwhm = None
    if model_focus is not None:
        best_focus_pos, best_focus = find_best_focus_from_model(model_focus, int(X.min()), int(X.max()), maximize=True)
    else:
        best_focus_pos = best_focus = None

    if model_fwhm is not None:
        best_fwhm_pos, best_fwhm = find_best_focus_from_model(model_fwhm, int(X.min()), int(X.max()), maximize=False)
        best_fwhm_focus = model_focus.predict([[best_fwhm_pos]])[0,0] if model_focus is not None else None
        best_focus_fwhm = model_fwhm.predict([[best_focus_pos]])[0,0] if best_focus_pos is not None else None
    else:
        best_fwhm_pos = best_fwhm = best_fwhm_focus = None

    best_sample_fwhm = (best_fwhm_pos, best_fwhm, best_fwhm_focus)
    best_sample_focus = (best_focus_pos, best_focus_fwhm, best_focus)

    X = X.reshape((X.size, 1))
    Yfwhm = Yfwhm.reshape((Yfwhm.size, 1))
    Yfocus = Yfocus.reshape((Yfocus.size, 1))

    score_fwhm = -1
    if model_fwhm is not None:
        try:
            score_fwhm = model_fwhm.score(X, Yfwhm)
        except ValueError:
            logger.exception("Error scoring FWHM model")

    score_focus = -1
    if model_focus is not None:
        try:
            score_focus = model_focus.score(X, Yfocus)
        except ValueError:
            logger.exception("Error scoring contrast model")

    if score_focus > 0.7 and score_fwhm > 0.7:
        logger.info(
            "Fit both curves (contrast=%g, FWHM=%g), averaging optimal points",
            score_focus, score_fwhm,
        )
        best_pos = int((score_focus * best_focus_pos + score_fwhm * best_fwhm_pos) / (score_focus + score_fwhm))
        best = (
            best_pos,
            model_fwhm.predict([[best_pos]])[0,0],
            model_focus.predict([[best_pos]])[0,0],
        )
    elif score_focus > 0 and score_focus > score_fwhm:
        logger.info(
            "Can only fit contrast curve (contrast=%g, FWHM=%g), using contrast model",
            score_focus, score_fwhm,
        )
        best = best_sample_focus
    elif score_fwhm > 0 and score_fwhm > score_focus:
        logger.info(
            "Can only fit FWHM curve (contrast=%g, FWHM=%g), using FWHM model",
            score_focus, score_fwhm,
        )
        best = best_sample_fwhm
    else:
        logger.warning(
            "Poor curve fitting (contrast=%g, FWHM=%g), using best contrast measure, ignoring predictive models",
            score_focus, score_fwhm,
        )
        best = true_best_sample_focus

    return best_sample_fwhm, best_sample_focus, best, model_focus, model_fwhm


def plot_focus_curve(min_pos, max_pos, focus_model, fwhm_model, samples, best_focus, best_fwhm, best):
    """ Produce a figure showing focus data

    :param int min_pos: Minimum focus position to graph
    :param int max_pos: Maximum focus position to graph
    :param focus_model: Trained sklearn model that fits the focus curve
    :param fwhm_model: Trained sklearn model that fits the FWHM curve
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

    Yfocus = focus_model.predict(X) if focus_model is not None else None
    Yfwhm = fwhm_model.predict(X) if fwhm_model is not None else None

    # Temporary hack until we can have a proper UI for this
    samples = numpy.array(sorted(samples)).T
    best_focus = numpy.array([best_focus]).T
    best_fwhm = numpy.array([best_fwhm]).T
    best = numpy.array([best]).T
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].scatter(samples[0], samples[1], c="#008000", marker="+")
    ax[1].scatter(samples[0], samples[2], c="#000080", marker=".")
    if best_fwhm is not None and all(best_fwhm):
        ax[0].scatter(best_fwhm[0], best_fwhm[1], c="#00F000", marker="*")
    if best_focus is not None and all(best_focus):
        ax[1].scatter(best_focus[0], best_focus[2], c="#0000F0", marker="o")
    ax[0].scatter(best[0], best[1], c="#00F000", marker="1")
    ax[1].scatter(best[0], best[2], c="#0000F0", marker="1")
    if Yfwhm is not None:
        ax[0].plot(X[:,0], Yfwhm[:,0], c="#004000", linestyle="dashed")
    if Yfocus is not None:
        ax[1].plot(X[:,0], Yfocus[:,0], c="#000040", linestyle="dashed")
    ax[1].set_xlabel('position')
    ax[1].set_ylabel('contrast')
    ax[0].set_ylabel('FWHM')
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f)
        f.seek(0)
        img = PIL.Image.open(f)
        img.load()
    return img
