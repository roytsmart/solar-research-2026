"""
Two Gaussians fitted to every line profile of a spectral cube.
"""

import numpy as np
import scipy.optimize
import scipy.special
import joblib
from ._caching import memory

__all__ = [
    "fit_gaussians",
    "seed_gaussians",
]


def _gaussian(
    params: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    A Gaussian integrated over each cell of a grid, and its derivatives.

    Integrated rather than sampled at the middle of each cell, since the
    lines are only a cell or two wide and a Gaussian sampled that coarsely
    is not the same shape as the light which fell in each cell.

    Parameters
    ----------
    params
        The area of the Gaussian, its center, and its standard deviation.
        The area is what the cells would add up to if the grid reached out
        forever, so it is in the units of the values of the cells.
    edges
        The edges of the cells, one more of them than there are cells.
    """
    area, center, sigma = params
    z = (edges - center) / sigma
    density = np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
    fraction = np.diff(scipy.special.ndtr(z))
    jacobian = np.stack(
        [
            fraction,
            -area * np.diff(density) / sigma,
            -area * np.diff(z * density) / sigma,
        ],
        axis=~0,
    )
    return area * fraction, jacobian


def _model(
    params: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Any number of Gaussians on a flat background, and the derivatives.

    Parameters
    ----------
    params
        Three parameters for each Gaussian, as in :func:`_gaussian`, followed
        by the background.
    edges
        The edges of the cells, one more of them than there are cells.
    """
    num = (params.size - 1) // 3
    result = np.full(edges.size - 1, params[~0])
    jacobian = []
    for i in range(num):
        value, jacobian_i = _gaussian(params[3 * i : 3 * i + 3], edges)
        result = result + value
        jacobian.append(jacobian_i)
    jacobian.append(np.ones((edges.size - 1, 1)))
    return result, np.concatenate(jacobian, axis=~0)


def _fit(
    profile: np.ndarray,
    edges: np.ndarray,
    guess: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    The least-squares fit of :func:`_model` to one profile.

    Parameters
    ----------
    profile
        The value of each cell.
    edges
        The edges of the cells.
    guess
        Where to start.
    lower
        The lowest each parameter may go.
    upper
        The highest each parameter may go.
    """
    result = scipy.optimize.least_squares(
        fun=lambda p: _model(p, edges)[0] - profile,
        x0=np.clip(guess, lower, upper),
        jac=lambda p: _model(p, edges)[1],
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
    )
    return result.x, 2 * result.cost


def _fit_profile(
    profile: np.ndarray,
    edges: np.ndarray,
    width_min: float,
    width_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One Gaussian and then two fitted to a profile, from several starts.

    A pair of Gaussians has more than one way to fit most profiles, and
    where the fit ends up depends on where it starts, so it is started from
    each of the ways an explosive event is expected to look and the best of
    them is kept: a core with a broad component under it, a core split into
    two halves, and a core with a second component out in either wing.
    Each start is built from the single Gaussian, which has only one answer.

    Parameters
    ----------
    profile
        The value of each cell.
    edges
        The edges of the cells.
    width_min
        The narrowest a Gaussian may be, as a standard deviation.
    width_max
        The broadest a Gaussian may be, as a standard deviation.

    Returns
    -------
    single
        The area, center and width of the single Gaussian, its background,
        and the sum of the squares of its residuals.
    double
        The area, center and width of each of the two Gaussians, their
        background, and the sum of the squares of the residuals.
    """
    nan = np.full(5, np.nan), np.full(8, np.nan)
    if not np.all(np.isfinite(profile)) or not np.any(profile > 0):
        return nan

    # Put into numbers of order one, which the solver treats better than the
    # numbers of order a trillion the product is in.
    scale = profile.max()
    profile = profile / scale

    centers = (edges[1:] + edges[:-1]) / 2
    lo, hi = edges[0], edges[~0]

    total = profile.sum()
    mean = (profile * centers).sum() / total
    sigma = np.sqrt((profile * (centers - mean) ** 2).sum() / total)
    sigma = np.clip(sigma, width_min, width_max)

    lower = np.array([0, lo, width_min, 0])
    upper = np.array([np.inf, hi, width_max, np.inf])
    single, rss_single = _fit(
        profile=profile,
        edges=edges,
        guess=np.array([total, mean, sigma, profile.min()]),
        lower=lower,
        upper=upper,
    )

    a, v, w, c = single
    guesses = [
        [0.8 * a, v, w, 0.2 * a, v, 3 * w, c],
        [0.5 * a, v - w, 0.7 * w, 0.5 * a, v + w, 0.7 * w, c],
        [0.8 * a, v, w, 0.2 * a, v - 3 * w, w, c],
        [0.8 * a, v, w, 0.2 * a, v + 3 * w, w, c],
    ]
    lower = np.array([0, lo, width_min, 0, lo, width_min, 0])
    upper = np.array([np.inf, hi, width_max, np.inf, hi, width_max, np.inf])
    double, rss_double = min(
        (
            _fit(
                profile=profile,
                edges=edges,
                guess=np.array(guess),
                lower=lower,
                upper=upper,
            )
            for guess in guesses
        ),
        key=lambda fit: fit[1],
    )

    # Back into the units of the product, which the centers and the widths
    # were never taken out of.
    single = single * scale
    single[1:3] = single[1:3] / scale
    double = double * scale
    for i in (1, 2, 4, 5):
        double[i] = double[i] / scale

    return (
        np.append(single, rss_single * scale**2),
        np.append(double, rss_double * scale**2),
    )


def _fit_frame(
    frame: np.ndarray,
    edges: np.ndarray,
    width_min: float,
    width_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    :func:`_fit_profile` over a batch of profiles.

    Parameters
    ----------
    frame
        The profiles, with the cells along the last axis.
    edges
        The edges of the cells.
    width_min
        The narrowest a Gaussian may be.
    width_max
        The broadest a Gaussian may be.
    """
    shape = frame.shape[:~0]
    profiles = frame.reshape(-1, frame.shape[~0])
    single, double = zip(
        *[_fit_profile(p, edges, width_min, width_max) for p in profiles]
    )
    return (
        np.stack(single).reshape(*shape, -1),
        np.stack(double).reshape(*shape, -1),
    )


@memory.cache
def fit_gaussians(
    radiance: np.ndarray,
    edges: np.ndarray,
    width_min: float,
    width_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    One Gaussian and a pair of Gaussians fitted to every profile of a cube.

    Each is a least-squares fit on a flat background, of Gaussians
    integrated over each cell rather than sampled at its middle, with no
    weighting: the noise of an inversion is not the noise of the photons it
    was made from, and nothing better is known about it.

    The pair is unordered: which of the two is which depends only on which
    start the best fit came from.

    Parameters
    ----------
    radiance
        The spectral radiance, with time along the first axis and the cells
        of the spectrum along the last.
    edges
        The edges of the cells, in velocity, one more than there are cells.
    width_min
        The narrowest a Gaussian may be, as a standard deviation in the units
        of `edges`.
    width_max
        The broadest a Gaussian may be, as a standard deviation.

    Returns
    -------
    single
        The shape of `radiance` with the spectrum replaced by five numbers:
        the area of the Gaussian, as a sum of cells in the units of
        `radiance`, its center and standard deviation in the units of
        `edges`, the background, and the sum of the squares of the
        residuals.
    double
        The same with eight: area, center and width of the first Gaussian,
        then of the second, the background, and the sum of squares.
    """
    # One row of one frame to each task, which is a few seconds of fitting
    # each: long against the overhead of handing it to a worker, and short
    # enough that no worker is left with a long tail when the rest are done.
    rows = radiance.reshape(-1, *radiance.shape[2:])
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_fit_frame)(row, edges, width_min, width_max) for row in rows
    )
    single, double = zip(*results)
    shape = radiance.shape[:~0]
    return (
        np.stack(single).reshape(*shape, -1),
        np.stack(double).reshape(*shape, -1),
    )


_offsets_neighbor = (
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
    (-1, 0, 0),
    (1, 0, 0),
)
"""
The places whose fits each fit is started again from, as offsets in time,
:math:`y` and :math:`x`: the four beside it, and itself in the frames before
and after.
"""


def _neighbors(double: np.ndarray, dirty: np.ndarray) -> np.ndarray:
    """
    The pair of each place's neighbors, as starts for fitting it again.

    Parameters
    ----------
    double
        The pairs, with time, :math:`y` and :math:`x` as the first three axes
        and their eight numbers along the last.
    dirty
        Which places changed in the last pass. A place none of whose
        neighbors changed would only be started again from where it was
        started last time, and is given no starts at all.

    Returns
    -------
    :
        The starts, with the neighbors along the second to last axis, and
        :obj:`numpy.nan` for a neighbor off the edge of the cube or not
        worth starting from.
    """
    shape = double.shape[:3]
    result = np.full((*shape, len(_offsets_neighbor), double.shape[~0]), np.nan)
    for i, offset in enumerate(_offsets_neighbor):
        target = []
        source = []
        for d, n in zip(offset, shape):
            target.append(slice(max(-d, 0), n - max(d, 0)))
            source.append(slice(max(d, 0), n - max(-d, 0)))
        target = tuple(target)
        source = tuple(source)
        start = np.where(dirty[source][..., np.newaxis], double[source], np.nan)
        result[target + (i,)] = start
    return result


def _refit_profile(
    profile: np.ndarray,
    edges: np.ndarray,
    width_min: float,
    width_max: float,
    current: np.ndarray,
    starts: np.ndarray,
    improvement_min: float = 1e-6,
) -> np.ndarray:
    """
    A pair of Gaussians fitted to a profile again from other starts, and
    the best of those and the pair it had.

    Parameters
    ----------
    profile
        The value of each cell.
    edges
        The edges of the cells.
    width_min
        The narrowest a Gaussian may be.
    width_max
        The broadest a Gaussian may be.
    current
        The pair the profile has, as the eight numbers of
        :func:`fit_gaussians`.
    starts
        The pairs to start from, one to a row, with rows of
        :obj:`numpy.nan` to skip.
    improvement_min
        How much smaller, as a fraction, the sum of the squares of the
        residuals of a new pair must be for it to replace the pair the
        profile has.

        Not zero, since a start from the neighbor of a place usually leads
        back to the pair the place already has, and then finishes a part in
        a billion better or worse than it did the first time, depending on
        where it stopped. Taking those would change most places on every
        pass and never be done. A pair which is actually a different way of
        fitting the profile does better by a percent or more.
    """
    result = current
    if not np.all(np.isfinite(profile)) or not np.any(profile > 0):
        return result

    scale = profile.max()
    profile = profile / scale

    lo, hi = edges[0], edges[~0]
    lower = np.array([0, lo, width_min, 0, lo, width_min, 0])
    upper = np.array([np.inf, hi, width_max, np.inf, hi, width_max, np.inf])

    # The areas and the background are in the units of the product, and
    # the fit is done in units of the peak.
    unscale = np.array([scale, 1, 1, scale, 1, 1, scale])

    for start in starts:
        if not np.all(np.isfinite(start)):
            continue
        params, rss = _fit(
            profile=profile,
            edges=edges,
            guess=start[:~0] / unscale,
            lower=lower,
            upper=upper,
        )
        rss = rss * scale**2
        if rss < (1 - improvement_min) * result[~0]:
            result = np.append(params * unscale, rss)

    return result


def _refit_row(
    row: np.ndarray,
    edges: np.ndarray,
    width_min: float,
    width_max: float,
    current: np.ndarray,
    starts: np.ndarray,
) -> np.ndarray:
    """
    :func:`_refit_profile` over a batch of profiles.

    Parameters
    ----------
    row
        The profiles, with the cells along the last axis.
    edges
        The edges of the cells.
    width_min
        The narrowest a Gaussian may be.
    width_max
        The broadest a Gaussian may be.
    current
        The pair each profile has.
    starts
        The starts of each profile.
    """
    return np.stack(
        [
            _refit_profile(p, edges, width_min, width_max, c, s)
            for p, c, s in zip(row, current, starts)
        ]
    )


@memory.cache
def seed_gaussians(
    radiance: np.ndarray,
    edges: np.ndarray,
    width_min: float,
    width_max: float,
    num_passes: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    :func:`fit_gaussians`, with each pair then fitted again from the pairs
    of its neighbors.

    A pair of Gaussians can fit most profiles in more than one way nearly as
    well, and two neighboring places whose profiles are nearly the same can
    be fitted in different ways, since each fit is on its own and ends up
    wherever its start leads it. Their maps are then speckled with places
    that have gone one way among places that have gone the other, which is
    the fit changing its mind rather than the Sun changing.

    So each place is fitted again, starting from the pair of each of its
    four neighbors and from its own pair in the frames before and after,
    and it takes whichever pair fits it best, its own included. A place
    only ever changes to a pair which fits it better, so this cannot make
    any fit worse; what it does is let a way of fitting a region which one
    place found spread to the others it suits. That is repeated until
    nothing changes, or for `num_passes` passes.

    Parameters
    ----------
    radiance
        The spectral radiance, with time along the first axis, then
        :math:`y` and :math:`x`, and the cells of the spectrum along the last.
    edges
        The edges of the cells, in velocity, one more than there are cells.
    width_min
        The narrowest a Gaussian may be, as a standard deviation in the units
        of `edges`.
    width_max
        The broadest a Gaussian may be, as a standard deviation.
    num_passes
        The most passes to make.

    Returns
    -------
    single
        The single Gaussian of :func:`fit_gaussians`, unchanged.
    double
        The pairs, as in :func:`fit_gaussians`.
    """
    single, double = fit_gaussians(radiance, edges, width_min, width_max)

    dirty = np.ones(double.shape[:3], dtype=bool)

    for _ in range(num_passes):
        starts = _neighbors(double, dirty)
        rows = [(t, y) for t in range(double.shape[0]) for y in range(double.shape[1])]
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(_refit_row)(
                radiance[t, y],
                edges,
                width_min,
                width_max,
                double[t, y],
                starts[t, y],
            )
            for t, y in rows
        )
        result = np.stack(results).reshape(double.shape)
        same = (result == double) | (np.isnan(result) & np.isnan(double))
        dirty = ~np.all(same, axis=~0)
        double = result
        if not dirty.any():
            break

    return single, double
