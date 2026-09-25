"""
An IRIS observation of a transition region explosive event,
built up one panel at a time.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.colorbar
import matplotlib.ticker
import astropy.units as u
import astropy.visualization
import astropy.constants
import colorsynth
import named_arrays as na
import iris
from .._observations import observation_iris
from ._color import velocity_color_default, percentile_default
from ._event import x_event_default, y_event_default
from ._layout import (
    figsize_default,
    rect_image_default,
    rect_colorbar_default,
    rect_spectrum_default,
    rect_profile_default,
    set_limits_sky,
)
from ._path import default_path

__all__ = [
    "iris_ee",
    "iris_ee_gallery",
]


def _show_raster(
    obs: "iris.sg.SpectrographObservation",
    index_time: int,
    ax: matplotlib.axes.Axes,
    cax: matplotlib.axes.Axes,
    velocity_color: u.Quantity,
    percentile: float,
) -> None:
    """
    Draw the raster as a false-color image, with its color key.

    Shared by :func:`iris_ee` and :func:`iris_ee_gallery`, so that the two
    draw the same picture of the sky and the same key beside it.

    Parameters
    ----------
    obs
        The observation to draw.
    index_time
        The index along the time axis to display.
    ax
        The axes to draw the image in.
    cax
        The axes to draw the color key in.
    velocity_color
        The Doppler velocity mapped to each end of the visible spectrum.
    percentile
        The percentile of the signal placed at the top of the brightness
        scale, separately at each wavelength.
    """
    fig = ax.figure

    # `show` adds a second axes to the key, for the velocity beside the
    # wavelength, and does not hand it back, so it is picked out by being the
    # one which was not there before.
    axes_before = set(fig.axes)

    obs.show(
        index_time=index_time,
        ax=ax,
        cax=cax,
        velocity_min=-velocity_color,
        velocity_max=+velocity_color,
        # Given rather than left to `show`, which would use its own
        # percentile, so that this figure and the blink are scaled alike.
        vmax=np.nanpercentile(
            obs.outputs,
            percentile,
            axis=(obs.axis_time, obs.axis_detector_x, obs.axis_detector_y),
        ),
    )

    (cax_twin,) = set(fig.axes) - axes_before

    # Said rather than left as a bare unit, so that the key reads the same
    # way as the key of the blink.
    cax.set_ylabel(f"wavelength ({na.unit(obs.inputs.wavelength_rest):latex_inline})")
    cax_twin.set_ylabel(f"velocity ({velocity_color.unit:latex_inline})")

    # The key is narrow and the radiance runs to six figures, so it can
    # only carry a couple of ticks, turned to keep them apart.
    cax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2))
    cax.tick_params(axis="x", labelrotation=45, labelsize="small")


def iris_ee(
    time: str = "2013-10-22 11:30",
    window: str = "Si IV 1394",
    index_time: int = 0,
    x_event: u.Quantity = x_event_default,
    y_event: u.Quantity = y_event_default,
    velocity_limit: u.Quantity = 250 * u.km / u.s,
    velocity_color: u.Quantity = velocity_color_default,
    velocity_limit_key: u.Quantity = velocity_color_default,
    percentile: float = percentile_default,
    velocity_gap: u.Quantity = 150 * u.km / u.s,
    position_gap: u.Quantity = 5 * u.arcsec,
    speed_sound: u.Quantity = 44 * u.km / u.s,
    linewidth: float = 2,
    headroom: float = 1.7,
    height_key: float = 0.5,
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 200,
    path: None | pathlib.Path = None,
) -> list[pathlib.Path]:
    """
    Save a sequence of three figures, each adding one more panel.

    The panels are, in the order they are revealed:

    1. The raster displayed as a false-color image, where the hue is the
       Doppler velocity.
    2. The spectrum along the slit, together with the red line marking the
       slit in the first panel.
    3. The spectral line profile at the explosive event, compared to the
       median profile of the whole raster, together with the red marker
       straddling the explosive event in the second panel.

    Each red marker therefore appears at the same time as the panel it points
    at, and never before it.

    Every figure has exactly the same layout,
    so that the panels appear one at a time without the earlier panels moving.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    index_time
        The index along the time axis to display.
    x_event
        The horizontal position of the explosive event, measured from the
        center of the field.
        The slit shown in the second panel is the one passing nearest to it.
    y_event
        The vertical position of the explosive event.
    velocity_limit
        The Doppler velocity range to display in the second and third panels.
    velocity_color
        The Doppler velocity mapped to each end of the visible spectrum in
        the first panel.
        The same value is used for the raster and for the color key,
        so that the key always describes the image.
    velocity_limit_key
        The Doppler velocity range to display in the color key.
    percentile
        The percentile of the signal placed at the top of the brightness
        scale of the first panel, separately at each wavelength.
    velocity_gap
        The half-width of the gap in the marker straddling the explosive
        event in the second panel.
    position_gap
        The half-height of the gap in the marker straddling the explosive
        event in the first panel.
    speed_sound
        The sound speed to mark in the third panel.
        Si IV forms near :math:`\\log T = 4.9`, where the sound speed is
        roughly 43 km/s for a fully ionized plasma with :math:`\\mu = 0.6`.
    linewidth
        The width of the spectral line profiles in the third panel.
    headroom
        The height of the third panel as a multiple of the height of the
        profile of the explosive event, which leaves room for the legend.
    height_key
        The height of the color key in the first figure,
        as a fraction of the height of the panels it stands in for.
    figsize
        The width and height of the figures in inches.
    dpi
        The resolution used for the parts of the figures which are too
        detailed to store as vectors.
    path
        The directory in which to save the figures.
        If :obj:`None`, they are saved alongside the other figures.
    """
    obs = observation_iris(
        time=time,
        window=window,
    )

    axis_time = obs.axis_time
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y

    # The event is given as a place on the sky, and the nearest cell of the
    # raster is the one shown. The raster is a parallelogram, so a cell has
    # to be searched for in both directions at once rather than found in each
    # separately: the horizontal coordinate changes along the slit as well as
    # across it.
    centers = obs.inputs.position[{axis_time: index_time}].cell_centers(
        (axis_x, axis_y)
    )
    distance = np.square(centers.x - x_event) + np.square(centers.y - y_event)
    nearest = np.argmin(distance)

    index_slit = {axis_time: index_time, axis_x: int(nearest[axis_x].ndarray)}
    index = index_slit | {axis_y: int(nearest[axis_y].ndarray)}

    if path is None:
        path = default_path
    path.mkdir(parents=True, exist_ok=True)

    velocity = obs.inputs.velocity
    position = obs.inputs.position

    unit_position = na.unit(position.x)
    x_slit = position.x[index].ndarray.value
    y_event = position.y[index].ndarray.value

    median = np.nanmedian(obs.outputs, axis=(axis_time, axis_x, axis_y))

    with astropy.visualization.quantity_support():

        # Placed rather than arranged, so that another figure can put its
        # image of the sky in exactly the same place, see :mod:`._layout`.
        fig = plt.figure(figsize=figsize)
        axs = [
            fig.add_axes(rect_image_default),
            fig.add_axes(rect_spectrum_default),
            fig.add_axes(rect_profile_default),
        ]
        cax = fig.add_axes(rect_colorbar_default)

        _show_raster(
            obs=obs,
            index_time=index_time,
            ax=axs[0],
            cax=cax,
            velocity_color=velocity_color,
            percentile=percentile,
        )

        # The spectrum along the slit.
        axs[1].sharey(axs[0])
        na.plt.pcolormesh(
            velocity[index_slit],
            position.y[index_slit],
            C=obs.outputs[index_slit].value,
            vmin=0,
            vmax=np.nanpercentile(obs.outputs[index_slit].value, 99.9),
            ax=axs[1],
        )
        axs[1].set_ylabel("")
        axs[1].tick_params(axis="y", labelleft=False)
        axs[1].set_title(f"$x = {x_slit:0.1f}$ {unit_position:latex_inline}")

        # The line profile at the explosive event.
        axs[2].sharex(axs[1])
        na.plt.stairs(
            velocity[index],
            obs.outputs[index],
            label=r"EE $1394\,\AA$",
            linewidth=linewidth,
            ax=axs[2],
        )
        na.plt.stairs(
            velocity,
            median,
            label=r"median $1394\,\AA$",
            linewidth=linewidth,
            ax=axs[2],
        )
        # Drawn underneath the profiles so that they read as a backdrop
        # against which the profiles are compared.
        axs[2].axvline(
            +speed_sound.value,
            color="red",
            linestyle="--",
            label="$c_s$",
            zorder=-1,
        )
        axs[2].axvline(
            -speed_sound.value,
            color="red",
            linestyle="--",
            zorder=-1,
        )
        axs[2].yaxis.tick_right()
        axs[2].yaxis.set_label_position("right")

        # Leave room above the profile of the event for the legend, as a
        # multiple of its height rather than as a number, so that it does not
        # depend on the units of the signal.
        axs[2].set_ylim(None, headroom * np.nanmax(obs.outputs[index].ndarray))
        axs[2].set_title(f"$y = {y_event:0.1f}$ {unit_position:latex_inline}")
        axs[2].legend()

        # The second and third panels share this axis, so setting it once
        # sets it for both.
        axs[1].set_xlim(-velocity_limit.value, +velocity_limit.value)

        set_limits_sky(axs[0], position)

    # Each color mesh has hundreds of thousands of cells, which would make an
    # enormous SVG if they were stored as vectors, so draw them as embedded
    # images instead. The axes, text, and lines stay vectors.
    for ax in fig.axes:
        for artist in ax.collections:
            artist.set_rasterized(True)

    # The positions are fixed, so hiding a panel cannot move the others, but
    # the figure still has to be drawn before anything can be measured off it.
    fig.canvas.draw()

    # Both markers are drawn as a pair of segments with a gap in the middle,
    # so that they point at the feature without covering it. Their endpoints
    # are in axes coordinates, so they can only be computed once the limits
    # are final and the figure has been drawn.
    def _fraction(
        ax: matplotlib.axes.Axes,
        point: tuple[float, float],
        component: int,
    ) -> float:
        """The position of a point in the coordinates of the given axes."""
        return ax.transAxes.inverted().transform(ax.transData.transform(point))[
            component
        ]

    gap_position = position_gap.to_value(na.unit(position.y))
    lines_slit = [
        axs[0].axvline(
            x=x_slit,
            ymin=ymin,
            ymax=ymax,
            color="red",
        )
        for ymin, ymax in [
            (0, _fraction(axs[0], (x_slit, y_event - gap_position), 1)),
            (_fraction(axs[0], (x_slit, y_event + gap_position), 1), 1),
        ]
    ]

    gap_velocity = velocity_gap.to_value(na.unit(velocity))
    lines_event = [
        axs[1].axhline(
            y=y_event,
            xmin=xmin,
            xmax=xmax,
            color="red",
        )
        for xmin, xmax in [
            (0, _fraction(axs[1], (-gap_velocity, y_event), 0)),
            (_fraction(axs[1], (+gap_velocity, y_event), 0), 1),
        ]
    ]

    # A key showing which Doppler velocity maps to which color in the raster,
    # which is only useful while the raster is the only panel on screen.
    # It is placed in the space that the later panels will occupy, using their
    # frozen positions, so that adding it cannot move anything else.
    bbox = axs[1].get_position().union([axs[1].get_position(), axs[2].get_position()])
    height = height_key * bbox.height

    # Half an inch of room on the left so that the vertical axis label does
    # not run into the raster, and the same on the right so that the labels
    # of the second vertical axis are not clipped by the edge of the figure.
    pad = 0.5 / figsize[0]

    # An inch of room underneath for the wavelength scale, which hangs below
    # the axes. The key sits at the bottom of the space left by the panels
    # which have not appeared yet, leaving the top of that space free.
    pad_lower = 1.0 / figsize[1]

    ax_key = fig.add_axes(
        (
            bbox.x0 + pad,
            bbox.y0 + pad_lower,
            bbox.width - 2 * pad,
            height,
        )
    )

    with astropy.visualization.quantity_support():

        velocity_center = velocity.cell_centers(obs.axis_wavelength).ndarray.squeeze()

        # The private function which `colorsynth` uses to map the spectral
        # axis onto the visible spectrum. It is given the same velocity range
        # as the raster, so that the key describes the colors in the image
        # rather than some other mapping.
        transform_wavelength = colorsynth._colorsynth._transform_wavelength(
            velocity_center,
            -1,
            -velocity_color,
            +velocity_color,
            None,
        )
        wavelength_visible = transform_wavelength(velocity_center)

        # The response of each color channel to a spectrum which is nonzero
        # at only one point along the spectral axis.
        spd = np.diagflat(np.ones(wavelength_visible.shape))
        XYZ = colorsynth.XYZcie1931_from_spd(
            spd,
            wavelength_visible[..., np.newaxis],
            axis=0,
        )
        XYZ = XYZ / XYZ.max(axis=1, keepdims=True)
        XYZ = XYZ * np.array([0.9505, 1.0000, 1.0890])[..., np.newaxis]
        red, green, blue = colorsynth.sRGB(XYZ, axis=0)

        for channel, color in zip([red, green, blue], ["red", "green", "blue"]):
            ax_key.plot(velocity_center, channel, color=color)

        ax_key_twin = ax_key.twinx()
        na.plt.stairs(
            velocity,
            median,
            label=r"median $1394\,\AA$",
            color="black",
            linewidth=linewidth,
            ax=ax_key_twin,
        )
        ax_key_twin.legend(loc="upper right")

        ax_key.set_xlim(-velocity_limit_key.value, +velocity_limit_key.value)
        ax_key.set_ylim(-0.2, None)
        ax_key.set_ylabel("sRGB response")
        ax_key_twin.set_ylim(0, None)

        # The same curves against the visible wavelength they are mapped onto.
        # The mapping is linear, so setting the limits is enough to line this
        # axis up with the one below it.
        unit_visible = wavelength_visible.unit
        ax_key_top = ax_key.twiny()
        ax_key_top.set_xlim(
            transform_wavelength(-velocity_limit_key).to_value(unit_visible),
            transform_wavelength(+velocity_limit_key).to_value(unit_visible),
        )
        ax_key_top.set_xlabel(f"visible wavelength ({unit_visible:latex_inline})")

        # A third scale, below the velocity, giving the wavelength of the
        # spectral line itself. The velocity and the wavelength are two
        # parameterizations of the same grid of pixels, so one is interpolated
        # onto the other rather than assuming a Doppler convention.
        unit_velocity = na.unit(velocity)
        unit_line = na.unit(obs.inputs.wavelength)
        ax_key_line = ax_key.twiny()
        ax_key_line.set_xlim(
            *np.interp(
                x=[-velocity_limit_key.to_value(unit_velocity)]
                + [+velocity_limit_key.to_value(unit_velocity)],
                xp=velocity[{axis_time: index_time}].ndarray.to_value(unit_velocity),
                fp=obs.inputs.wavelength[{axis_time: index_time}].ndarray.to_value(
                    unit_line
                ),
            )
        )
        ax_key_line.xaxis.set_ticks_position("bottom")
        ax_key_line.xaxis.set_label_position("bottom")
        ax_key_line.spines["bottom"].set_position(("outward", 36))
        for spine in ax_key_line.spines.values():
            spine.set_visible(False)
        ax_key_line.spines["bottom"].set_visible(True)
        ax_key_line.patch.set_visible(False)
        ax_key_line.set_xlabel(f"wavelength ({unit_line:latex_inline})")

        # These tick labels are long enough to run into each other at the
        # width of this axes, so use fewer of them than usual.
        ax_key_line.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))

    # Each panel appears along with the marker in the panel to its left which
    # points at it, so that a marker is never on screen before the panel
    # explaining it. The color key occupies the space of the panels which have
    # not appeared yet, so it goes away as soon as the first of them arrives.
    shown = [
        [ax_key, ax_key_twin, ax_key_top, ax_key_line],
        [axs[1], *lines_slit],
        [axs[2], *lines_event],
    ]
    hidden = [
        [],
        [ax_key, ax_key_twin, ax_key_top, ax_key_line],
        [],
    ]

    for group in shown[1:]:
        for artist in group:
            artist.set_visible(False)

    result = []
    for i, (group_shown, group_hidden) in enumerate(zip(shown, hidden)):
        for artist in group_shown:
            artist.set_visible(True)
        for artist in group_hidden:
            artist.set_visible(False)
        path_i = path / f"iris-ee-{i + 1}.svg"
        fig.savefig(path_i, dpi=dpi)
        result.append(path_i)

    plt.close(fig)

    return result


def _spds_gallery(
    wavelength: u.Quantity,
) -> dict[str, np.ndarray]:
    """
    The example spectra of the gallery in the :mod:`colorsynth` documentation.

    Parameters
    ----------
    wavelength
        The wavelengths to evaluate each spectrum at.
    """

    def gaussian(center: u.Quantity, width: u.Quantity) -> np.ndarray:
        """A Gaussian emission line with the given center and width."""
        return np.exp(-np.square((wavelength - center) / width)).value

    def blackbody(temperature: u.Quantity) -> np.ndarray:
        """A Planck spectrum at the given temperature, normalized to its peak."""
        h = astropy.constants.h
        c = astropy.constants.c
        k_B = astropy.constants.k_B
        spd = 1 / wavelength**5 / np.expm1(h * c / (wavelength * k_B * temperature))
        return (spd / spd.max()).to_value(u.dimensionless_unscaled)

    return {
        "blue emission line": gaussian(450 * u.nm, 10 * u.nm),
        "green emission line": gaussian(540 * u.nm, 10 * u.nm),
        "red emission line": gaussian(620 * u.nm, 10 * u.nm),
        "blue + red doublet": gaussian(450 * u.nm, 10 * u.nm)
        + gaussian(640 * u.nm, 10 * u.nm),
        "broad emission line": gaussian(550 * u.nm, 80 * u.nm),
        "flat spectrum": np.ones(wavelength.shape),
        "3000 K blackbody": blackbody(3000 * u.K),
        "20000 K blackbody": blackbody(20000 * u.K),
    }


def iris_ee_gallery(
    time: str = "2013-10-22 11:30",
    window: str = "Si IV 1394",
    index_time: int = 0,
    velocity_color: u.Quantity = velocity_color_default,
    percentile: float = percentile_default,
    top_gallery: float = 0.77,
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 200,
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    """
    The first figure of :func:`iris_ee`, with the key to its colors replaced
    by examples of how a spectrum becomes a color.

    The image of the sky and its key are drawn exactly as in the first figure
    of :func:`iris_ee`, so that the two can follow one another on a slide
    without the sky moving. Where that figure plots the response of the red,
    green and blue channels, this one shows the gallery from the
    :mod:`colorsynth` documentation: eight spectra, each filled with the color
    :func:`colorsynth.rgb` gives it. Narrow lines give saturated hues, a blue
    and a red line together give a purple no single wavelength can, and broad
    spectra wash out toward white.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    index_time
        The index along the time axis to display.
    velocity_color
        The Doppler velocity mapped to each end of the visible spectrum in
        the image of the sky.
    percentile
        The percentile of the signal placed at the top of the brightness
        scale of the image, separately at each wavelength.
    top_gallery
        The top of the gallery, as a fraction of the figure, which leaves the
        top right of the slide free for its label.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution used for the parts of the figure which are too
        detailed to store as vectors.
    path
        The location to save the figure.
        If :obj:`None`, it is saved alongside the other figures.
    """
    obs = observation_iris(
        time=time,
        window=window,
    )

    if path is None:
        path = default_path / "iris-ee-gallery.svg"
    path.parent.mkdir(parents=True, exist_ok=True)

    # The gallery occupies the box that the key to the colors occupies in
    # the first figure of `iris_ee`, extended upward to just below the label.
    pad = 0.5 / figsize[0]
    pad_lower = 1.0 / figsize[1]
    left = rect_spectrum_default[0] + pad
    right = rect_profile_default[0] + rect_profile_default[2] - pad
    bottom = rect_spectrum_default[1] + pad_lower

    wavelength = np.linspace(380, 700, num=321) * u.nm
    spds = _spds_gallery(wavelength)

    hspace = 0.6

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect_image_default)
        cax = fig.add_axes(rect_colorbar_default)

        _show_raster(
            obs=obs,
            index_time=index_time,
            ax=ax,
            cax=cax,
            velocity_color=velocity_color,
            percentile=percentile,
        )
        set_limits_sky(ax, obs.inputs.position)

        gridspec = fig.add_gridspec(
            nrows=4,
            ncols=2,
            left=left,
            right=right,
            bottom=bottom,
            top=top_gallery,
            hspace=hspace,
            wspace=0.1,
        )
        axs = gridspec.subplots(sharex=True, sharey=True)

        for ax_spd, (label, spd) in zip(axs.flat, spds.items()):
            color = colorsynth.rgb(
                spd,
                wavelength,
                axis=-1,
                spd_min=0,
                spd_max=spd.max(),
            )
            ax_spd.set_facecolor("0.85")
            ax_spd.fill_between(wavelength, spd, color=color)
            ax_spd.plot(wavelength, spd, color="black", linewidth=0.5)
            ax_spd.set_title(label, fontsize="small")
            ax_spd.set_ylim(0, 1.1)
            ax_spd.tick_params(labelsize="x-small")
            ax_spd.set_xlabel("")
            ax_spd.set_ylabel("")

        for ax_spd in axs[~0]:
            ax_spd.set_xlabel(
                f"wavelength ({wavelength.unit:latex_inline})",
                fontsize="small",
            )

        # One label for the whole column, halfway down it, rather than one
        # on every row.
        axs[1, 0].set_ylabel("relative intensity", fontsize="small")
        axs[1, 0].yaxis.set_label_coords(-0.25, -hspace / 2)

    for artist in ax.collections + cax.collections:
        artist.set_rasterized(True)

    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    return path


if __name__ == "__main__":

    for p in iris_ee():
        print(p)
